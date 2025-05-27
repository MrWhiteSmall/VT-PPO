from MODEL_CKP import *
import argparse
import os
import torch

'''
超参实验
1 lora scale 的超参（实验≥5 次）
2 reward 的相似度计算的超参（实验≥5 次）
3 loss policy 中α β两个 loss 的权重实验（实验≥5 次）
3 online ppo 训练采样次数的超参，默认是 5 次，然后梯度累计反传  以及  单次采样步数的超参 默认是 12（实验≥8 次）

'''
ls=0.1
wc=.5
wl=.2
wh=.3
policy_alpha=100 # reward weight
policy_beta=0.01 # KL weight
sample_times=5 # s
sample_steps_per_time=12 # st

outdir = f'output_ls{ls}_wc{wc}_wl{wl}_wh{wh}_pa{policy_alpha}_pb{policy_beta}_s{sample_times}_st{sample_steps_per_time}'
mixed_precision_types = ["no", "fp16", "bf16"]
mixed_precision = mixed_precision_types[1]
weight_dtypes={
          "no": torch.float32,
          "fp16": torch.float16,
          "bf16": torch.bfloat16,
      }
weight_dtype = weight_dtypes[mixed_precision]

def parse_args():
    parser = argparse.ArgumentParser(description="Training and evaluation script")
    # 模型路径相关参数
    model_path_group = parser.add_argument_group("Model Paths")
    model_path_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    model_path_group.add_argument(
        "--base_model_path",
        type=str,
        default=base_model_path,
        help="The path to the base model to use for evaluation."
    )
    model_path_group.add_argument(
        "--resume_path",
        type=str,
        default=resume_path,
        help="Path to the checkpoint of trained model."
    )
    model_path_group.add_argument(
        "--sft_path",
        type=str,
        default="./checkpoints/models/finetune_b512_lr2e-05_max10000_w0.01",
        help="Path to the pretrained supervised finetuned model."
    )
    model_path_group.add_argument(
        "--reward_model_path",
        type=str,
        default="./checkpoints/reward/reward_model_5007.pkl",
        help="Path to the pretrained reward model."
    )
    model_path_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models."
    )
    model_path_group.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        help="Revision of pretrained non-ema model identifier."
    )

    # 数据集相关参数
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on."
    )
    data_group.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset."
    )
    data_group.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data."
    )
    data_group.add_argument(
        "--data_root_path", 
        type=str, 
        help="Path to the dataset to evaluate."
    )
    data_group.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image."
    )
    data_group.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption."
    )
    data_group.add_argument(
        "--prompt_path",
        type=str,
        default="./dataset/drawbench/data_meta.json",
        help="Path to the prompt dataset."
    )
    data_group.add_argument(
        "--prompt_category",
        type=str,
        default="all",
        help="Categories to use from prompt dataset."
    )
    data_group.add_argument(
        "--person_path",
        type=str,
        help="Path to person images."
    )
    data_group.add_argument(
        "--cloth_path",
        type=str,
        help="Path to cloth images."
    )
    data_group.add_argument(
        "--cloth_type",
        type=str,
        help="Type of cloth images."
    )

    # 训练超参数
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging, truncate the number of training examples."
    )
    training_group.add_argument(
        "--seed", 
        type=int, 
        default=555,
        help="A seed for reproducible training."
    )
    training_group.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images."
    )
    training_group.add_argument(
        "--width",
        type=int,
        default=768,
        help="Width resolution for input images."
    )
    training_group.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height resolution for input images."
    )
    training_group.add_argument(
        "--center_crop",
        action="store_true",
        default=True,
        help="Whether to center crop the input images."
    )
    training_group.add_argument(
        "--random_flip",
        action="store_true",
        default=True,
        help="Whether to randomly flip images horizontally."
    )
    training_group.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for training."
    )
    training_group.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=100,
        help="Total number of training epochs."
    )
    training_group.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps."
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=12,
        help="Number of updates steps to accumulate before backward pass."
    )
    training_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Whether to use gradient checkpointing."
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for policy."
    )
    training_group.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by number of GPUs/batch size."
    )
    training_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use.'
    )
    training_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    training_group.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether to use 8-bit Adam."
    )
    training_group.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    training_group.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    training_group.add_argument(
        "--adam_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use."
    )
    training_group.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer."
    )
    training_group.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=1.0,
        help="Max gradient norm."
    )
    training_group.add_argument(
        "--clip_norm", 
        type=float, 
        default=0.1,
        help="Norm for gradient clipping."
    )

    # 推理相关参数
    inference_group = parser.add_argument_group("Inference")
    inference_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to perform."
    )
    inference_group.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="The scale of classifier-free guidance for inference."
    )
    inference_group.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per prompt."
    )
    inference_group.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of validation images to generate."
    )

    # 强化学习相关参数
    rl_group = parser.add_argument_group("Reinforcement Learning")
    rl_group.add_argument(
        "--p_step",
        type=int,
        default=5,
        help="Number of steps to update the policy per sampling step."
    )
    rl_group.add_argument(
        "--p_batch_size",
        type=int,
        default=1,
        help="Batch size for policy update per gpu."
    )
    rl_group.add_argument(
        "--g_step", 
        type=int, 
        default=1,
        help="The number of sampling steps."
    )
    rl_group.add_argument(
        "--g_batch_size",
        type=int,
        default=6,
        help="Batch size of prompts for sampling per gpu."
    )
    rl_group.add_argument(
        "--reward_weight", 
        type=float, 
        default=100,
        help="Weight of reward loss."
    )
    rl_group.add_argument(
        "--reward_flag",
        type=int,
        default=0,
        help="0: ImageReward, 1: Custom reward model."
    )
    rl_group.add_argument(
        "--reward_filter",
        type=int,
        default=0,
        help="0: raw value, 1: took positive."
    )
    rl_group.add_argument(
        "--kl_weight", 
        type=float, 
        default=0.01,
        help="Weight of kl loss."
    )
    rl_group.add_argument(
        "--kl_warmup", 
        type=int, 
        default=-1,
        help="Warm up for kl weight."
    )
    rl_group.add_argument(
        "--buffer_size", 
        type=int, 
        default=1000,
        help="Size of replay buffer."
    )
    rl_group.add_argument(
        "--v_batch_size", 
        type=int, 
        default=49,
        help="Batch size for value function update per gpu."
    )
    rl_group.add_argument(
        "--v_lr", 
        type=float, 
        default=1e-4,
        help="Learning rate for value fn."
    )
    rl_group.add_argument(
        "--v_step", 
        type=int, 
        default=5,
        help="Number of steps to update the value function per sampling step."
    )
    rl_group.add_argument(
        "--ratio_clip",
        type=float,
        default=1e-4,
        help="Ratio clip for PPO."
    )

    # 输出和日志
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default=outdir,
        help="The output directory where results will be written."
    )
    output_group.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory."
    )
    output_group.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report results and logs to."
    )
    output_group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where downloaded models and datasets will be stored."
    )

    # 检查点和恢复
    checkpoint_group = parser.add_argument_group("Checkpoint")
    checkpoint_group.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50, # 2000
        help="Save a checkpoint every X updates."
    )
    checkpoint_group.add_argument(
        "--save_interval",
        type=int,
        default=100, # 
        help="Save model every X steps."
    )
    checkpoint_group.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store."
    )
    checkpoint_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint."
    )

    # 硬件和性能
    performance_group = parser.add_argument_group("Performance")
    performance_group.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help="Whether to allow TF32 on Ampere GPUs."
    )
    performance_group.add_argument(
        "--mixed_precision",
        type=str,
        default=mixed_precision,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision."
    )
    performance_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading."
    )
    performance_group.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether to use xformers."
    )
    performance_group.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank."
    )

    # 其他功能
    feature_group = parser.add_argument_group("Features")
    feature_group.add_argument(
        "--use_ema", 
        action="store_true",
        default=False,
        help="Whether to use EMA model."
    )
    feature_group.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hub."
    )
    feature_group.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub."
    )
    feature_group.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync."
    )
    feature_group.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    feature_group.add_argument(
        "--eval_pair",
        action="store_true",
        help="Whether to evaluate the pair."
    )
    feature_group.add_argument(
        "--concat_eval_results",
        action="store_true",
        help="Whether to concatenate all conditions into one image."
    )
    feature_group.add_argument(
        "--concat_axis",
        type=str,
        choices=["x", "y", "random"],
        default="y",
        help="The axis to concat the cloth feature."
    )
    feature_group.add_argument(
        "--enable_condition_noise",
        action="store_true",
        default=True,
        help="Whether to enable condition noise."
    )

    # 实验性/调试参数
    experimental_group = parser.add_argument_group("Experimental")
    experimental_group.add_argument(
        "--v_flag",
        type=int,
        default=1,
        help="Experimental flag."
    )
    experimental_group.add_argument(
        "--single_flag",
        type=int,
        default=1,
        help="Experimental flag."
    )
    experimental_group.add_argument(
        "--single_prompt",
        type=str,
        default="A green colored rabbit.",
        help="Single prompt for debugging."
    )
    experimental_group.add_argument(
        "--sft_initialization",
        type=int,
        default=0,
        help="Whether to use SFT initialization."
    )
    experimental_group.add_argument(
        "--multi_gpu",
        type=int,
        default=0,
        help="Whether to use multi-GPU."
    )
    experimental_group.add_argument(
        "--lora_rank", 
        type=int, 
        default=4,
        help="Rank for LoRA."
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args