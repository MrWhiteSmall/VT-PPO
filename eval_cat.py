import os
from PIL import Image
import torch
# import safetensors.torch
# from diffusers.image_processor import VaeImageProcessor

from model.pipeline_cat_origin import CatVTONPipeline
# from model.cloth_masker import AutoMasker

from eval_parse import parse_args

'''
fp32 10.8GB
fp16 5.2GB
'''

def _get_batch(data_iter_loader, data_iterator, prompt_list, args):
  """Creates a batch."""
  batch = next(data_iter_loader, None)
  if batch is None:
    batch = next(
        iter(
            data_iterator(prompt_list, batch_size=args.g_batch_size)
        )
    )
  batch_list = []
  for i in range(len(batch)):
    # batch text只取前两句话
    batch[i]['text'] = '.'.join(batch[i]['text'].split('.')[:1])
    batch[i]['cloth_text'] = '.'.join(batch[i]['cloth_text'].split('.')[:1])
    batch_list.extend([batch[i] for _ in range(args.num_samples)])
  batch = batch_list
  return batch

from os.path import join as osj
from util_mask_for_reward import parsing_model
from util_mask import get_mask_location_all
from MODEL_CKP import data_root
from MODEL_CKP import dataset_name,prompt_path
import json,random
from tqdm import tqdm
# data_root = ''
output_dir = './cat_res_dc'
def get_image_from_cat(args, 
                       pipe, 
                        batch):
    model_input_shape = (args.width , args.height) # w h
    """Collects trajectories."""
    person_path = osj(data_root,batch[0]['image_file']) 
    cloth_path = osj(data_root,batch[0]['cloth_file'])
    text      = batch[0]['text']
    # mask_path = batch['mask']
    person_image, cloth_image = \
            [Image.open(path) 
            for path in [person_path,cloth_path]]
            
    # 如果存在mask_path就直接读取
    if os.path.exists(os.path.join(data_root, "cloth_mask", person_path.split("/")[-1].replace(".jpg", ".png"))):
        cloth_mask = Image.open(os.path.join(data_root, "cloth_mask", person_path.split("/")[-1].replace(".jpg", ".png")))
    else:
        pose_img_pil = person_image
        target_shape = (384 , 512) # w h
        model_img_pil = pose_img_pil.resize(target_shape)
        model_parse, _ = parsing_model(model_img_pil.resize(target_shape))
        mask_clothing,mask_clothing_upper,mask_clothing_lower, \
        mask_limbs, mask_hands = get_mask_location_all(model_parse)
        # 保存一下cloth mask 下次可以直接读取
        save_dir = os.path.join(data_root, "cloth_mask")
        os.makedirs(save_dir, exist_ok=True)
        
        cloth_mask = mask_clothing_upper
        cloth_mask.save(os.path.join(save_dir, person_path.split("/")[-1].replace(".jpg", ".png")))
    person_image = person_image.resize(model_input_shape)
    cloth_image = cloth_image.resize(model_input_shape)
    mask = cloth_mask.resize(model_input_shape)
    
    generator = torch.Generator(device='cuda').manual_seed(20250509)
    

    with torch.no_grad():
      image = pipe(
            person_image,
            cloth_image,
            mask,
            encoding_hidden_states=None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,)[0]
        ## try condition -> embedding
        ## try prompt -> embedding
        
    image = image.resize(model_input_shape)
    
    # 创建新图片
    new_img = Image.new('RGB', (model_input_shape[0]*3,model_input_shape[1]))

    # 依次粘贴三张图片
    new_img.paste(image, (0, 0))
    new_img.paste(person_image, (model_input_shape[0], 0))
    new_img.paste(cloth_image, (model_input_shape[0]*2, 0))
    
    # assert isinstance(image,Image.Image)
    output_path = osj(
                output_dir,   'h'+os.path.basename(person_path)+'_'+\
                            'c'+os.path.basename(cloth_path))
    os.makedirs(output_dir,exist_ok=True)
    new_img.save(output_path)
    
def main():
    args = parse_args()
    pipe = CatVTONPipeline(
      base_ckpt=args.base_model_path,
      attn_ckpt=args.resume_path,
      attn_ckpt_version=dataset_name,
      weight_dtype={
          "no": torch.float32,
          "fp16": torch.float16,
          "bf16": torch.bfloat16,
      }[args.mixed_precision],
      device="cuda",
      skip_safety_check=True,
      is_train=True,
    )
    with open(prompt_path) as json_file:
        prompt_dict = json.load(json_file)
    prompt_list = []
    for prompt in prompt_dict:
        prompt_list.append(prompt)
    def _my_data_iterator(data, batch_size):
        # Shuffle the data randomly
        random.shuffle(data)

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
        yield batch
    data_iterator = _my_data_iterator(prompt_list, batch_size=1)
    data_iter_loader = iter(data_iterator)
    # for count in range(0, args.max_train_steps // args.p_step): # 10000 // 5
    for i in tqdm(range(len(prompt_list))):
        # fix batchnorm
        batch = _get_batch(
            data_iter_loader, _my_data_iterator, prompt_list, args
        )
        get_image_from_cat(args,pipe,batch)
        
if __name__=='__main__':
    main()