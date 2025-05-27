import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
# from diffusers.pipelines.stable_diffusion.safety_checker import \
#     StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor
# import torch.nn.functional as F
# from torch.distributions import Normal
# from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
# from safetensors.torch import save_file

from model.attn_processor import SkipAttnProcessor,LoRACrossAttnProcessor
# from model.attn_processor_self import MyLoraAttnProcessor2_0
from model.utils import get_trainable_module, init_adapter
from cat_utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)

# from diffusers.models.attention_processor import AttnProcessor2_0

torch._dynamo.config.suppress_errors = True

SCHEDULER_CONFIG = ''

Lora_Path_Dir = ''

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
        # self.noise_scheduler = DDIMSchedulerExtended.from_config(self.noise_scheduler.config)

        self.vae = AutoencoderKL.from_pretrained("/root/lsj/checkpoints/vae").to(device, dtype=weight_dtype)
        # if not skip_safety_check:
        # self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=weight_dtype)
        # Error no file named diffusion_pytorch_model.bin found in directory /root/lsj/checkpoints/ootd
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet")
        
        # 初始化 attn2 为 skipattn
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
        # 获取attention1
        self.attn_modules = get_trainable_module(self.unet, "attention")
        # 在原有attention1中加载 q k v 的权重
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        
        
        #替换 attn1 为 mylora attn1 
        # attn_procs = {}
        # unet = self.unet
        # # unet_sd = unet.state_dict()
        # for name in unet.attn_processors.keys():
        #     #如果是自注意力注意力attn1，那么设置为空，否则设置为交叉注意力的维度
        #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        #     #这里记录此时这个快的通道式
        #     if name.startswith("mid_block"):
        #     #'block_out_channels', [320, 640, 1280, 1280]
        #         hidden_size = unet.config.block_out_channels[-1]
        #     elif name.startswith("up_blocks"):
        #     #name中的，up_block.的后一个位置就是表示是第几个上块
        #         block_id = int(name[len("up_blocks.")])
        #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        #     elif name.startswith("down_blocks"):
        #         block_id = int(name[len("down_blocks.")])
        #         hidden_size = unet.config.block_out_channels[block_id]
        #     if cross_attention_dim is None: # self attn
        #         attn_procs[name] = MyLoraAttnProcessor2_0(hidden_size=hidden_size, 
        #                                                   rank=4,scale=0.2)
        #     else:
        #         attn_procs[name] = SkipAttnProcessor(hidden_size=hidden_size, 
        #                                       cross_attention_dim=cross_attention_dim,)
        # #最后这里将unet的注意力处理器设置为自己重构后的注意力字典
        # unet.set_attn_processor(attn_procs)
        
        # self.attn_modules_lora = get_trainable_module(self.unet, "attention_lora")
        
        self.unet.requires_grad_(False)
        self.attn_modules.requires_grad_(False)
        # self.attn_modules_lora.requires_grad_(False)
        
        self.unet.to(device, dtype=weight_dtype)

        # 检查是否只有LoRA参数可训练
        # for name, param in unet.named_parameters():
        #     if "lora_" in name and not param.requires_grad:
        #         print(f"ERROR: LoRA参数 {name} 未启用梯度！")
        #     if "to_q.weight" in name and param.requires_grad:
        #         print(f"ERROR: 原始权重 {name} 未被冻结！")

        # 训练时仅优化LoRA参数
        # self.optimizer = torch.optim.AdamW(
        #         list(self.attn_modules_lora.parameters()),
        #         lr=1e-5,
        #         betas=(0.9, 0.99),
        #         weight_decay=1e-2,
        #         eps=1e-08,
        # )
        # 保存LoRA权重（仅需保存新增参数）
        # torch.save(
        #     {n: p for n, p in self.attn_modules_lora.named_parameters() if "lora_" in n},
        #     "lora_weights.pt"
        # )
        # 3. 加载LoRA权重
        # lora_weights = torch.load("lora_weights.pt")
        # self.unet.load_state_dict(lora_weights, strict=False)

        
        

        self.vae = torch.compile(self.vae, mode="reduce-overhead")
            
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        
        self.vae.requires_grad_(False)
        # self.safety_checker.requires_grad_(False)
        

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
                    self.collecting.append(
                        (inpainting_latent_model_input.detach().clone().cpu(),
                         latents.detach().clone().cpu(),
                         mask_latent_concat.detach().clone().cpu(),
                         masked_latent_concat.detach().clone().cpu(),
                         encoding_hidden_states.detach().clone().cpu(),
                         t)
                                           )                                                            
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
                    latents, log_prob = self.noise_scheduler.step_logprob(
                        noise_pred, unet_t, latents, **extra_step_kwargs
                    )
                    latents = latents.prev_sample
                    # latents = latents.to(inpainting_latent_model_input.dtype)
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
    
                 
    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            print(os.path.join(attn_ckpt, sub_folder, 'attention'))
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
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

    @torch.no_grad()
    def __call__(self, 
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

        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
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
                # with torch.no_grad():
                noise_pred= self.unet(
                    inpainting_latent_model_input,
                    t,
                    encoder_hidden_states=None, # FIXME
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # unet_times = unsqueeze3x(torch.tensor([t] * noise_pred.shape[0])).to(noise_pred.device)
                # prev_latents = latents.clone()
                # noise_pred : torch.Size([1, 4, 256, 96])
                latents = self.noise_scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs
                    ).prev_sample
                # latents = latents.to(inpainting_latent_model_input.dtype)

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
                
            del latents, masked_latent_concat, mask_latent_concat, encoding_hidden_states
            return image
        

class VTPPO:
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
        # self.noise_scheduler = DDIMSchedulerExtended.from_config(self.noise_scheduler.config)

        self.vae = AutoencoderKL.from_pretrained("/root/lsj/checkpoints/vae").to(device, dtype=weight_dtype)
        # if not skip_safety_check:
        self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=weight_dtype)
        # Error no file named diffusion_pytorch_model.bin found in directory /root/lsj/checkpoints/ootd
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        # self.unet = self.unet.float()
        # self.unet_copy = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        init_adapter(self.unet, cross_attn_cls=LoRACrossAttnProcessor)  # Skip Cross-Attention
        # init_adapter(self.unet_copy, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
        
        self.attn_modules = get_trainable_module(self.unet, "attention2")
        self.attn_modules.to(device, dtype=weight_dtype)
        # self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        # load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, 
        #                                                          sub_folder, 'attention'))
        
        self.vae = torch.compile(self.vae, mode="reduce-overhead")
            
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        # if use_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        
        self.vae.requires_grad_(False)
        # self.safety_checker.requires_grad_(False)
        self.unet.requires_grad_(False)
        # self.unet_copy.requires_grad_(False)
        # self.attn_modules_copy.requires_grad_(False)
        self.attn_modules.requires_grad_(False)
        # self.attn_modules.train()
        # for param in self.attn_modules.parameters():
        #     param.requires_grad = True

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
                    self.collecting.append(
                        (inpainting_latent_model_input.detach().clone().cpu(),
                         latents.detach().clone().cpu(),
                         mask_latent_concat.detach().clone().cpu(),
                         masked_latent_concat.detach().clone().cpu(),
                         encoding_hidden_states.detach().clone().cpu(),
                         t)
                                           )                                                            
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
                    latents, log_prob = self.noise_scheduler.step_logprob(
                        noise_pred, unet_t, latents, **extra_step_kwargs
                    )
                    latents = latents.prev_sample
                    # latents = latents.to(inpainting_latent_model_input.dtype)
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
    
                 
    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            print(os.path.join(attn_ckpt, sub_folder, 'attention'))
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
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

    @torch.no_grad()
    def __call__(self, 
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

        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
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
                # with torch.no_grad():
                noise_pred= self.unet(
                    inpainting_latent_model_input,
                    t,
                    encoder_hidden_states=encoding_hidden_states, # FIXME
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # unet_times = unsqueeze3x(torch.tensor([t] * noise_pred.shape[0])).to(noise_pred.device)
                # prev_latents = latents.clone()
                # noise_pred : torch.Size([1, 4, 256, 96])
                latents = self.noise_scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs
                    ).prev_sample
                # latents = latents.to(inpainting_latent_model_input.dtype)

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
                
            del latents, masked_latent_concat, mask_latent_concat, encoding_hidden_states
            return image
     