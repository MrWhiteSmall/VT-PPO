IMAG_CKP='/root/lsj/IMAGDressing-main/ckpt/IMAGDressing-v1_512.pt'
IP_CKP='/root/lsj/IMAGDressing-main/ckpt/ip-adapter-faceid-plusv2_sd15.bin'

VAE_PATH = '/root/lsj/checkpoints/sd-vae-ft-mse'
CLIP_PATH = '/root/lsj/checkpoints/clip-vit-large-patch14'
CKPT_PREFIX = '/root/lsj/checkpoints/ootd'
IP_A_PATH = '/root/lsj/checkpoints/IP-Adapter'
REALISTIC_PATH = '/root/lsj/checkpoints/Realistic_vision'
CONTROL_INPAINT_PATH = '/root/lsj/checkpoints/control_v11p_sd15_inpaint'
CONTROL_POSE_PATH = '/root/lsj/checkpoints/control_v11p_sd15_openpose'
POSE_DETECTOR_PATH = '/root/lsj/checkpoints/ControlNet'

REWARD_PATH='/root/lsj/huggingface/ImageReward/ImageReward.pt'
REWARD_CONFIG_PATH='/root/lsj/huggingface/ImageReward/med_config.json'

g_batch_size = 1

from os.path import join as osj
root='/root/lsj/google-research-master/dpok'
data_root='/root/datasets/VITON-HD_ori/'

base_model_path = "/root/lsj/checkpoints/ootd"
resume_path = "/root/lsj/checkpoints/catvton"
repo_path="/root/lsj/checkpoints/catvton"


dataset_name = 'vitonhd'
prompt_path="/root/datasets/VITON-HD_ori/test/test.json"