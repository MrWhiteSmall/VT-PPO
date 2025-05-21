import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor
import torch.nn.functional as F
from torch.distributions import Normal
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from safetensors.torch import save_file

from model.attn_processor import SkipAttnProcessor,LoRACrossAttnProcessor,PSLB
from model.utils import get_trainable_module, init_adapter
from cat_utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)

torch._dynamo.config.suppress_errors = True

class DDIMSchedulerExtended(DDIMScheduler):
  """Extension of diffusers.DDIMScheduler."""

  def _get_variance_logprob(self, timestep, prev_timestep):
    alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
    mask_a = (prev_timestep >= 0).int().to(timestep.device)
    mask_b = 1 - mask_a
    alpha_prod_t_prev = (
        self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
        + self.final_alpha_cumprod.to(timestep.device) * mask_b
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (
        1 - alpha_prod_t / alpha_prod_t_prev
    )

    return variance

  # new step function that can take multiple timesteps and middle step images as
  # input
  def step_logprob(
      self,
      model_output,
      timestep,
      sample,
      eta = 1.0,
      use_clipped_model_output = False,
      generator=None,
      variance_noise = None,
      return_dict = True,
  ):  # pylint: disable=g-bare-generic
    """Predict the sample at the previous timestep by reversing the SDE.

    Core function to propagate the diffusion process from the learned model
    outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion
          model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`): current instance of sample being created
          by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected"
          `model_output` from the clipped predicted original sample. Necessary
          because predicted original sample is clipped to [-1, 1] when
          `self.config.clip_sample` is `True`. If no clipping has happened,
          "corrected" `model_output` would coincide with the one provided as
          input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for
          the variance using `generator`, we can directly provide the noise for
          the variance itself. This is useful for methods such as
          CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than
          DDIMSchedulerOutput class

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is
        True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.

        log_prob (`torch.FloatTensor`): log probability of the sample.
    """
    if self.num_inference_steps is None:
      raise ValueError(
          "Number of inference steps is 'None', you need to run 'set_timesteps'"
          " after creating the scheduler"
      )

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )

    # 2. compute alphas, betas
    self.alphas_cumprod = self.alphas_cumprod.to(timestep.device)
    alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
    # alpha_prod_t = alpha_prod_t.to(torch.float16)
    mask_a = (prev_timestep >= 0).int().to(timestep.device)
    mask_b = 1 - mask_a
    alpha_prod_t_prev = (
        self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
        + self.final_alpha_cumprod.to(timestep.device) * mask_b
    )
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
      pred_original_sample = (
          sample - beta_prod_t ** (0.5) * model_output
      ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
      pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
      pred_original_sample = (alpha_prod_t**0.5) * sample - (
          beta_prod_t**0.5
      ) * model_output
      # predict V
      model_output = (alpha_prod_t**0.5) * model_output + (
          beta_prod_t**0.5
      ) * sample
    else:
      raise ValueError(
          f"prediction_type given as {self.config.prediction_type} must be one"
          " of `epsilon`, `sample`, or `v_prediction`"
      )

    # 4. Clip "predicted x_0"
    if self.config.clip_sample:
      pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance_logprob(timestep, prev_timestep).to(
        dtype=sample.dtype
    )
    std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)

    if use_clipped_model_output:
      # the model_output is always re-derived from the clipped x_0 in Glide
      model_output = (
          sample - alpha_prod_t ** (0.5) * pred_original_sample
      ) / beta_prod_t ** (0.5)

    # pylint: disable=line-too-long
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * model_output

    # pylint: disable=line-too-long
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample
        + pred_sample_direction
    )

    if eta > 0:
      device = model_output.device
      if variance_noise is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and variance_noise. Please make sure"
            " that either `generator` or `variance_noise` stays `None`."
        )

      if variance_noise is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=device,
            dtype=model_output.dtype,
        )
      variance = std_dev_t * variance_noise
      dist = Normal(prev_sample, std_dev_t)
      prev_sample = prev_sample.detach().clone() + variance
      log_prob = (
          dist.log_prob(prev_sample.detach().clone())
          .mean(dim=-1)
          .mean(dim=-1)
          .mean(dim=-1)
          .detach()
          .cpu()
      )
    if not return_dict:
      return (prev_sample,)

    return (
        DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        ),
        log_prob,
    )

  def step_forward_logprob(
      self,
      model_output,
      timestep,
      sample,
      next_sample,
      eta = 1.0,
      use_clipped_model_output = False,
      generator=None,
      variance_noise = None,
      return_dict = True,
  ):  # pylint: disable=g-bare-generic

    if self.num_inference_steps is None:
      raise ValueError(
          "Number of inference steps is 'None', you need to run 'set_timesteps'"
          " after creating the scheduler"
      )

    # pylint: disable=line-too-long
    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
    mask_a = (prev_timestep >= 0).int().to(timestep.device)
    mask_b = 1 - mask_a
    alpha_prod_t_prev = (
        self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
        + self.final_alpha_cumprod.to(timestep.device) * mask_b
    )
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
      pred_original_sample = (
          sample - beta_prod_t ** (0.5) * model_output
      ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
      pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
      pred_original_sample = (alpha_prod_t**0.5) * sample - (
          beta_prod_t**0.5
      ) * model_output
      # predict V
      model_output = (alpha_prod_t**0.5) * model_output + (
          beta_prod_t**0.5
      ) * sample
    else:
      raise ValueError(
          f"prediction_type given as {self.config.prediction_type} must be one"
          " of `epsilon`, `sample`, or `v_prediction`"
      )

    # 4. Clip "predicted x_0"
    if self.config.clip_sample:
      pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance_logprob(timestep, prev_timestep).to(
        dtype=sample.dtype
    )
    std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)

    if use_clipped_model_output:
      # the model_output is always re-derived from the clipped x_0 in Glide
      model_output = (
          sample - alpha_prod_t ** (0.5) * pred_original_sample
      ) / beta_prod_t ** (0.5)

    # pylint: disable=line-too-long
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * model_output

    # pylint: disable=line-too-long
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample
        + pred_sample_direction
    )

    if eta > 0:
      dist = Normal(prev_sample, std_dev_t)
      log_prob = (
          dist.log_prob(next_sample.detach().clone())
          .mean(dim=-1)
          .mean(dim=-1)
          .mean(dim=-1)
      )

    return log_prob
    
    

SCHEDULER_CONFIG = ''

class CatVTONPipeline:
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt, 
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device='cuda',
        compile=False,
        skip_safety_check=False,
        use_tf32=True,
        is_train=False,
    ):
        self.device = device
        self.is_train = is_train
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.noise_scheduler = DDIMSchedulerExtended.from_config(self.noise_scheduler.config)

        self.vae = AutoencoderKL.from_pretrained("/root/lsj/checkpoints/vae").to(device, dtype=weight_dtype)
        # if not skip_safety_check:
        self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=weight_dtype)
        # Error no file named diffusion_pytorch_model.bin found in directory /root/lsj/checkpoints/ootd
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        # self.unet = self.unet.float()
        self.unet_copy = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        init_adapter(self.unet, cross_attn_cls=PSLB)  # Skip Cross-Attention
        init_adapter(self.unet_copy, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
        
        # print(self.unet)
        # print(self.unet_copy)
        
        self.attn_modules = get_trainable_module(self.unet, "attention2")
        self.attn_modules1 = get_trainable_module(self.unet, "attention")
        self.attn_modules_copy = get_trainable_module(self.unet_copy, "attention")
        self.attn_modules.to(device, dtype=weight_dtype)
        self.attn_modules1.to(device, dtype=weight_dtype)
        self.attn_modules_copy.to(device, dtype=weight_dtype)
        # self.attn_modules_copy = self.attn_modules_copy.to(device, dtype=weight_dtype)
        self.optimizer = torch.optim.AdamW(
                list(self.attn_modules.parameters()),
                lr=1e-5,
                betas=(0.9, 0.99),
                weight_decay=1e-2,
                eps=1e-08,
        )
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        self.auto_attn_ckpt_load_copy(attn_ckpt, attn_ckpt_version)
        
        self.vae.requires_grad_(False)
        # self.feature_extractor.requires_grad_(False)
        self.safety_checker.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.unet_copy.requires_grad_(False)
        self.attn_modules1.requires_grad_(False)
        self.attn_modules_copy.requires_grad_(False)
        self.attn_modules.requires_grad_(True)
        
        # Pytorch 2.0 Compile
        self.unet_copy.eval()

        self.vae = torch.compile(self.vae, mode="reduce-overhead")
        # self.unet = torch.compile(self.unet)
            
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        # if use_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        
        self.attn_modules.train()

        
        self.collecting = []

    def forward_collect_traj_ddim(self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        encoding_hidden_states=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        output_type = 'pil',
        **kwargs):
        
        # 统一使用设备管理
        # device = self.device
        # dtype = self.weight_dtype
        
        self.latents_list = []
        self.log_prob_list = []
        self.kl_list = []
        
        concat_dim = -2  # FIXME: y axis concat
        # Prepare inputs to Tensor
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        # Mask image
        masked_image = image * (mask < 0.5)
        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        # ori_img_latent = compute_vae_encodings(image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        
        
        mask = mask.float()
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        mask_latent = mask_latent.to(torch.bfloat16)
        del image, mask, condition_image
        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        # true_x0 = torch.cat([ori_img_latent, condition_latent], dim=concat_dim)\
        #             .to(self.device, dtype=self.weight_dtype)# 1 4 256 96
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # ori_masked_latents = torch.clone(mask_latent_concat)
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        unconditional_prompt_embeds = torch.zeros_like(condition_latent)
        guided_prompt_embeds = condition_latent
        ## uncondi && condi
        self.unconditional_prompt_embeds = unconditional_prompt_embeds.detach().cpu()
        self.guided_prompt_embeds       = guided_prompt_embeds.detach().cpu()
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, 
                               unconditional_prompt_embeds], 
                              dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)

        self.latents_list.append((
            latents.detach().clone().cpu()
        ))

        if self.is_train:
            with tqdm.tqdm(total=num_inference_steps) as progress_bar:
                # print('inference')
                for i, t in enumerate(timesteps):
                    # t = t.to(self.device, dtype=self.weight_dtype)
                    # expand the latents if we are doing classifier free guidance
                    non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                    non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
                    # prepare the input for the inpainting model
                    '''
                    2 4 /4 /4 non_inpainting_latent_model_input
                    2 1 /4 /4 mask_latent_concat
                    2 4 /4 /4 masked_latent_concat
                    '''
                    inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, 
                                                            mask_latent_concat, 
                                                            masked_latent_concat], dim=1)
                    inpainting_latent_model_input = \
                        inpainting_latent_model_input.to(self.device, dtype=self.weight_dtype) # torch.Size([2, 9, 128, 48])
                    # print(inpainting_latent_model_input.shape,inpainting_latent_model_input.dtype)
                    # 验证一下，输入是不是一样的
                    # self.collecting.append(
                    #     (inpainting_latent_model_input.detach().clone().cpu(),
                    #      latents.detach().clone().cpu(),
                    #      mask_latent_concat.detach().clone().cpu(),
                    #      masked_latent_concat.detach().clone().cpu(),
                    #      encoding_hidden_states.detach().clone().cpu(),
                    #      t)
                                        #    )                                                            
                    # print(t.shape)                                                            
                    # predict the noise residual
                    # with torch.no_grad():
                    noise_pred= self.unet(
                        inpainting_latent_model_input,
                        t,
                        encoder_hidden_states=encoding_hidden_states, # FIXME
                        return_dict=False,
                    )[0]
                    # old_noise_pred= self.unet_copy(
                    #     inpainting_latent_model_input,
                    #     t,
                    #     encoder_hidden_states=None, # FIXME
                    #     return_dict=False,
                    # )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    # compute the previous noisy sample x_t -> x_t-1
                    # latents = self.noise_scheduler.step(
                    #     noise_pred, t, latents, **extra_step_kwargs
                    # ).prev_sample
                    unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
                    unet_t = unsqueeze3x(torch.tensor([t])).to(noise_pred.device) # shape = torch.Size([1, 1, 1, 1])
                    # unet_times = unsqueeze3x(torch.tensor([t] * noise_pred.shape[0])).to(noise_pred.device)
                    # prev_latents = latents.clone()
                    # noise_pred : torch.Size([1, 4, 256, 96])
                    
                    # 这里的log 计算的是 p(x_t-1|x_t,z_t)
                    #       公式中 通过 xt zt 计算 可能的 x_t-1 并计算转移过去的概率
                    pesudo_latents, log_prob = self.noise_scheduler.step_logprob(
                        noise_pred, unet_t, latents, **extra_step_kwargs
                    )
                    pesudo_latents = pesudo_latents.prev_sample
                    # latents = latents.to(inpainting_latent_model_input.dtype)
                    
                    # 这个才是真的 latents
                    # 上面是加了一个随机噪声而已  这里的latent用于 下一个函数 p( x_t-1|x_t )
                    #       公式中 并不知道 x_t-1如何计算，所以直接给出当前的预测内容
                    #           后续中 x_t-1 就是时间步计算所需的 label
                    latents = self.noise_scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs
                    ).prev_sample
                    self.latents_list.append((
                                              latents
                                              ).detach().clone().cpu())
                    self.log_prob_list.append(log_prob.detach().clone().cpu())

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.noise_scheduler.order == 0
                    ):
                        progress_bar.update()
                        
            
                latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
                if output_type == "latent":
                    image = latents
                # Decode the final latents
                elif output_type == "pil":
                    latents = latents.detach()
                    latents = 1 / self.vae.config.scaling_factor * latents
                    image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                
                    image = numpy_to_pil(image)
                else:
                    latents = 1 / self.vae.config.scaling_factor * latents
                    image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                
                # Safety Check
                # if not self.skip_safety_check:
                #     current_script_directory = os.path.dirname(os.path.realpath(__file__))
                #     nsfw_image = os.path.join(os.path.dirname(current_script_directory), 'resource', 'img', 'NSFW.jpg')
                #     nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
                #     image_np = np.array(image)
                #     _, has_nsfw_concept = self.run_safety_checker(image=image_np)
                #     for i, not_safe in enumerate(has_nsfw_concept):
                #         if not_safe:
                #             image[i] = nsfw_image
            kl_sum = 0
            kl_path = []
            # for i in range(len(self.kl_list)):
            #     kl_sum += self.kl_list[i]
            # kl_path.append(kl_sum.clone())
            # for i in range(1, len(self.kl_list)):
            #     kl_sum -= self.kl_list[i - 1]
            #     kl_path.append(kl_sum.clone())
            
            return (
                image,
                self.latents_list,
                self.unconditional_prompt_embeds.detach().cpu(),
                self.guided_prompt_embeds.detach().cpu(),
                self.log_prob_list,
                kl_path,
                
                timesteps.detach().clone().cpu(), # list
                mask_latent_concat.detach().clone().cpu(), 
                masked_latent_concat.detach().clone().cpu(),
            )
    
                
        """
                
        if not self.is_train:
            total_loss = 0
            bsz = latents.shape[0]
            # 随机时间步
            # 输入：随机事件步的噪声（noise+step_noise)
            # 训练：与初始的noise做loss
            t = torch.randint(0, num_inference_steps, 
                            (bsz*2,), device=self.device)
            t = t.long()
            # expand the latents if we are doing classifier free guidance
            non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
            non_inpainting_latent_model_input = \
                self.noise_scheduler.scale_model_input(
                    non_inpainting_latent_model_input, 
                    t)
            # prepare the input for the inpainting model
            ''' 
            2 4 /4 /4 non_inpainting_latent_model_input
            2 1 /4 /4 mask_latent_concat
            2 4 /4 /4 masked_latent_concat
            '''
            inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, 
                                                        mask_latent_concat, 
                                                        masked_latent_concat], dim=1)
            
            inpainting_latent_model_input = inpainting_latent_model_input.to(self.device, 
                                                                             dtype=self.weight_dtype) # torch.Size([2, 9, 128, 48])
            # print('train')
            # print(inpainting_latent_model_input.shape)                                                            
            # print(t.shape)   
            # predict the noise residual
            # noise_pred_tmp= self.unet_copy(
            #     inpainting_latent_model_input,
            #     t.to(self.device,dtype=self.weight_dtype),
            #     encoder_hidden_states=None, # FIXME
            #     return_dict=False,
            # )[0]   # torch.Size([2, 4, 256, 96])
            noise_pred= self.unet(
                inpainting_latent_model_input,
                t.to(self.device,dtype=self.weight_dtype),
                encoder_hidden_states=encoding_hidden_states, # FIXME
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                
            # train loss(noise_pred , latents)
            '''
            input size (torch.Size([1, 4, 256, 96]))
            target size (torch.Size([1, 4, 256, 96]))
            '''
            loss = F.mse_loss(noise_pred.float(), 
                            latents.float(), reduction="mean")
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.noise_scheduler.step(
            #     noise_pred, t, latents, **extra_step_kwargs
            # ).prev_sample
        
            return total_loss
           
        """
     
    
    # Feed transitions pairs and old model
    def forward_calculate_logprob(
        self,
        
        latents=None,
        ts=None,
        next_latents=None,
        
        mask_latent_concat=None, 
        masked_latent_concat=None,
        encoding_hidden_states=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        
        eta = 1.0,
        
    ):
        latents = latents.to(self.device, dtype=self.weight_dtype)
        # ts = ts.to(self.device, dtype=self.weight_dtype)
        mask_latent_concat = mask_latent_concat.to(self.device, dtype=self.weight_dtype)
        masked_latent_concat = masked_latent_concat.to(self.device, dtype=self.weight_dtype)
        encoding_hidden_states = encoding_hidden_states.to(self.device, dtype=self.weight_dtype)
   
        do_classifier_free_guidance = (guidance_scale > 1.0)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        # expand the latents if we are doing classifier free guidance
        non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
        non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, 
                                                                                   ts)
        # prepare the input for the inpainting model
        '''
        2 4 /4 /4 non_inpainting_latent_model_input
        2 1 /4 /4 mask_latent_concat
        2 4 /4 /4 masked_latent_concat
        '''
        inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, 
                                                mask_latent_concat, 
                                                masked_latent_concat], dim=1)
        inpainting_latent_model_input = \
                inpainting_latent_model_input.to(self.device, dtype=self.weight_dtype) # torch.Size([2, 9, 128, 48])
        # print(inpainting_latent_model_input.shape)                                                            
        # print(ts.shape)                                                            
        # predict the noise residual
        
        # 在 self.collecting 寻找 t==ts 查看此时 inpainting_latent_model_input 是否相同
        # find_resv = next((x for x in self.collecting if x[-1] == ts),None)
        # if inpainting_latent_model_input.detach().clone().cpu() == find_resv[0].detach().clone().cpu():
        #     print('inpainting_latent_model_input is same')
        # else:
        #     print('inpainting_latent_model_input is different')
        noise_pred= self.unet(
            inpainting_latent_model_input,
            ts,
            encoder_hidden_states=encoding_hidden_states, # FIXME
            return_dict=False,
        )[0]
        with torch.no_grad():
            # tocuda -> process -> tocpu
            self.unet_copy.to(self.device, dtype=self.weight_dtype)
            old_noise_pred = self.unet_copy(
                inpainting_latent_model_input,
                ts,
                encoder_hidden_states=None, # FIXME
                return_dict=False,
            )[0]
            self.unet_copy.to('cpu', dtype=torch.float32)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
            old_noise_pred = old_noise_pred_uncond + guidance_scale * (
                old_noise_pred_text - old_noise_pred_uncond
            )
        # now we get the predicted noise
        kl_regularizer = (noise_pred - old_noise_pred) ** 2
        
        # compute the previous noisy sample x_t -> x_t-1
        unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
        unet_t = unsqueeze3x(torch.tensor([ts])).to(noise_pred.device)
        
        log_prob = self.noise_scheduler.step_forward_logprob(
            noise_pred, unet_t, latents, 
            next_latents,
            **extra_step_kwargs
        )
        # latents = latents.prev_sample
        
        
        return log_prob, kl_regularizer
    
    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            print(os.path.join(attn_ckpt, sub_folder, 'attention'))
            load_checkpoint_in_model(self.attn_modules1, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules1, os.path.join(repo_path, sub_folder, 'attention'))
            
    def auto_attn_ckpt_load_copy(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            print(os.path.join(attn_ckpt, sub_folder, 'attention'))
            load_checkpoint_in_model(self.attn_modules_copy, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules_copy, os.path.join(repo_path, sub_folder, 'attention'))
            
    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
        return image, has_nsfw_concept
    
    def check_inputs(self, image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # @torch.no_grad()
    def __call__(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        if self.is_train:
            # num_train_epochs = 100
            # save_ckp_dir = lambda epoch : f"./ckp/unet-epoch{str(epoch)}.safetensors"
            # for epoch in tqdm(range(num_train_epochs)):
            total_loss = self.main(
                image=image,
                condition_image=condition_image,
                mask=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                eta=eta,)
            # if (epoch%10 == 0 and epoch != 0 ) or epoch == (num_train_epochs-1):
            #     state_dict_attn_modules = self.attn_modules.state_dict()
            # for key in state_dict_attn_modules.keys():
            #     state_dict_attn_modules[key] = state_dict_attn_modules[key].to('cpu')
            #     save_file(state_dict_attn_modules, save_ckp_dir(epoch))
            return total_loss
        else:
            with torch.no_grad():
                image = self.main(
                    image=image,
                    condition_image=condition_image,
                    mask=mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    eta=eta,)
        return image
      