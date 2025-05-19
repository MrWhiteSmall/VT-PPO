import numpy as np
import cv2,os,shutil
from PIL import Image, ImageDraw
from preprocess.humanparsing.run_parsing import Parsing
from tqdm import tqdm
# Example usage

parsing_model = Parsing(0)
target_shape = (384 , 512)

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
    'neck':18,
}
'''
def get_parse_mask_for_similarity(model_path):
    if model_path exits:
        get parse path
        if parse_path exits:
            if mask clo limbs hands path exits:
                return parse, mask clo limbs hands
            else:
                mask_clothing, mask_limbs, mask_hands = get_masks_for_similarity(parse)
                save mask clo limbs hands
                return parse, mask clo limbs hands
        else:
            parse = get_parse_by_modelpath(model_path)
            save parse
            mask_clothing, mask_limbs, mask_hands = get_masks_for_similarity(parse)
            save mask clo limbs hands
            return parse, mask clo limbs hands
''' 
def get_parse_mask_for_similarity(model_path):
    mask_clo_path, mask_limbs_path, mask_hands_path, mask_parse_path = \
                                            get_mask_path_by_modelpath(model_path)
    if os.path.exists(mask_parse_path):
        model_parse = Image.open(mask_parse_path)
        if os.path.exists(mask_clo_path) and os.path.exists(mask_limbs_path) and os.path.exists(mask_hands_path):
            mask_clothing = Image.open(mask_clo_path)
            mask_limbs = Image.open(mask_limbs_path)
            mask_hands = Image.open(mask_hands_path)
            return model_parse, mask_clothing, mask_limbs, mask_hands
    else:
        model_parse = get_parse_by_modelpath(model_path)
        if not os.path.exists(os.path.dirname(mask_parse_path)):
            os.makedirs(os.path.dirname(mask_parse_path))
        model_parse.save(mask_parse_path)
    mask_clothing, mask_limbs, mask_hands = get_masks_for_similarity(model_parse=model_parse)
    if not os.path.exists(os.path.dirname(mask_clo_path)):
        os.makedirs(os.path.dirname(mask_clo_path))
    if not os.path.exists(os.path.dirname(mask_limbs_path)):
        os.makedirs(os.path.dirname(mask_limbs_path))
    if not os.path.exists(os.path.dirname(mask_hands_path)):
        os.makedirs(os.path.dirname(mask_hands_path))
    mask_clothing.save(mask_clo_path)
    mask_limbs.save(mask_limbs_path)
    mask_hands.save(mask_hands_path)
    return model_parse, mask_clothing, mask_limbs, mask_hands

def get_parse_by_modelpath(model_path):
    img_pil = Image.open(model_path).convert("RGB")

    model_img_pil = img_pil.resize(target_shape)
    model_parse, _ = parsing_model(model_img_pil.resize(target_shape))
    return model_parse

def get_mask_path_by_modelpath(model_path):
    mask_clo_path = os.path.join(model_path.replace('image', 'mask_cloth').replace('.jpg', '_mask_clothing.png'))
    mask_limbs_path = os.path.join(model_path.replace('image', 'mask_limbs').replace('.jpg', '_mask_limbs.png'))
    mask_hands_path = os.path.join(model_path.replace('image', 'mask_hands').replace('.jpg', '_mask_hands.png'))
    mask_parse_path = os.path.join(model_path.replace('image', 'parse-human_0_18').replace('.jpg', '.png'))
    return mask_clo_path, mask_limbs_path, mask_hands_path, mask_parse_path
def get_masks_for_similarity(model_parse: Image.Image=None,
                             model_parse_path='', width=384, height=512):
    """Generate clothing, limbs, and hands masks from parsed image.
    
    Args:
        model_parse: Parsed segmentation image (PIL Image)
        width, height: Target dimensions
        
    Returns:
        Tuple of (mask_clothing, mask_limbs, mask_hands) as PIL Images
    """
    if os.path.exists(model_parse_path):
        model_parse = Image.open(model_parse_path)
    else:
        model_parse = model_parse
    # Resize and convert to numpy array
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    
    # Initialize empty masks
    mask_clothing = np.zeros_like(parse_array, dtype=np.float32)
    mask_limbs = np.zeros_like(parse_array, dtype=np.float32)
    mask_hands = np.zeros_like(parse_array, dtype=np.float32)
    
    # Define label groups for each mask
    clothing_labels = [
        "upper_clothes", "skirt", "pants", "dress", 
        "belt", "scarf"
    ]
    
    limbs_labels = [
        "left_leg", "right_leg", 
        "left_shoe", "right_shoe",
        "head","neck",'hair',"sunglasses",
    ]
    
    hands_labels = [
        "left_arm", "right_arm"  # Note: Add these to your label_map if needed
    ]
    
    # Build masks by aggregating relevant labels
    for label_name, label_value in label_map.items():
        if label_name in clothing_labels:
            mask_clothing += (parse_array == label_value).astype(np.float32)
        if label_name in limbs_labels:
            mask_limbs += (parse_array == label_value).astype(np.float32)
        if label_name in hands_labels:
            mask_hands += (parse_array == label_value).astype(np.float32)
    
    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    def refine_mask(mask):
        """Apply morphological refinement to mask"""
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    # Refine each mask
    mask_clothing = refine_mask(mask_clothing)
    mask_limbs = refine_mask(mask_limbs)
    mask_hands = refine_mask(mask_hands)
    
    # Convert to PIL Images
    mask_clothing = Image.fromarray((mask_clothing * 255).astype(np.uint8))
    mask_limbs = Image.fromarray((mask_limbs * 255).astype(np.uint8))
    mask_hands = Image.fromarray((mask_hands * 255).astype(np.uint8))
    
    # for i in range(19):
    #     Image.fromarray(((
    #                       (parse_array == i).astype(np.float32)
    #                       ) * 255).astype(np.uint8))\
    #                           .save(f'mask_{i}.png')
    return mask_clothing, mask_limbs, mask_hands

if __name__ == "__main__":

    
    
    categories = ['test','train']
    dataroot = "/root/datasets/VITON-HD_ori/"
    for category in tqdm(categories):
        save_parse_dir = os.path.join(dataroot, category, 'parse-human_0_18/')
        save_clo_mask_dir = os.path.join(dataroot, category, 'mask_cloth/')
        save_limbs_mask_dir = os.path.join(dataroot, category, 'mask_limbs/')
        save_hands_mask_dir = os.path.join(dataroot, category, 'mask_hands/')
        if os.path.exists(save_parse_dir):
            shutil.rmtree(save_parse_dir)
        if os.path.exists(save_clo_mask_dir):
            shutil.rmtree(save_clo_mask_dir)
        if os.path.exists(save_limbs_mask_dir):
            shutil.rmtree(save_limbs_mask_dir)
        if os.path.exists(save_hands_mask_dir):
            shutil.rmtree(save_hands_mask_dir)
        os.makedirs(save_parse_dir)
        os.makedirs(save_clo_mask_dir)
        os.makedirs(save_limbs_mask_dir)
        os.makedirs(save_hands_mask_dir)
        
        model_dir = os.path.join(dataroot, category, 'image/')
        for img_name in tqdm(os.listdir(model_dir)):
            if img_name.endswith('.jpg'):
                model_path = os.path.join(model_dir, img_name)
                # Load image
                img_pil = Image.open(model_path).convert("RGB")
    
                model_img_pil = img_pil.resize(target_shape)
                model_parse, _ = parsing_model(model_img_pil.resize(target_shape))
                mask_clothing, mask_limbs, mask_hands = get_masks_for_similarity(model_parse=model_parse)
                model_parse.save(os.path.join(save_parse_dir, 
                                            os.path.basename(model_path).replace('.jpg', '.png')))
                mask_clothing.save(os.path.join(save_clo_mask_dir, 
                                            os.path.basename(model_path).replace('.jpg', '_mask_clothing.png')))
                mask_limbs.save(os.path.join(save_limbs_mask_dir,
                                            os.path.basename(model_path).replace('.jpg', '_mask_limbs.png')))
                mask_hands.save(os.path.join(save_hands_mask_dir,
                                            os.path.basename(model_path).replace('.jpg', '_mask_hands.png')))

            
    
    
    categories = ['upper', 'lower', 'dresses']
    dataroot = "/root/datasets/DressCode_1024/"
    for category in tqdm(categories):
        save_parse_dir = os.path.join(dataroot, category, 'parse-human_0_18/')
        save_clo_mask_dir = os.path.join(dataroot, category, 'mask_cloth/')
        save_limbs_mask_dir = os.path.join(dataroot, category, 'mask_limbs/')
        save_hands_mask_dir = os.path.join(dataroot, category, 'mask_hands/')
        if os.path.exists(save_parse_dir):
            shutil.rmtree(save_parse_dir)
        if os.path.exists(save_clo_mask_dir):
            shutil.rmtree(save_clo_mask_dir)
        if os.path.exists(save_limbs_mask_dir):
            shutil.rmtree(save_limbs_mask_dir)
        if os.path.exists(save_hands_mask_dir):
            shutil.rmtree(save_hands_mask_dir)
        os.makedirs(save_parse_dir)
        os.makedirs(save_clo_mask_dir)
        os.makedirs(save_limbs_mask_dir)
        os.makedirs(save_hands_mask_dir)
        
        model_dir = os.path.join(dataroot, category, 'image/')
        for img_name in tqdm(os.listdir(model_dir)):
            if img_name.endswith('.jpg'):
                model_path = os.path.join(model_dir, img_name)
                # Load image
                img_pil = Image.open(model_path).convert("RGB")
    
                model_img_pil = img_pil.resize(target_shape)
                model_parse, _ = parsing_model(model_img_pil.resize(target_shape))
                mask_clothing, mask_limbs, mask_hands = get_masks_for_similarity(model_parse=model_parse)
                model_parse.save(os.path.join(save_parse_dir, 
                                            os.path.basename(model_path).replace('.jpg', '.png')))
                mask_clothing.save(os.path.join(save_clo_mask_dir, 
                                            os.path.basename(model_path).replace('.jpg', '_mask_clothing.png')))
                mask_limbs.save(os.path.join(save_limbs_mask_dir,
                                            os.path.basename(model_path).replace('.jpg', '_mask_limbs.png')))
                mask_hands.save(os.path.join(save_hands_mask_dir,
                                            os.path.basename(model_path).replace('.jpg', '_mask_hands.png')))




def test():
    model_path = "/root/datasets/DressCode_1024/upper/image/000001_0.jpg"
    save_parse_dir = "/root/datasets/DressCode_1024/upper/parse-human_0_17/"
    save_clo_mask_dir = "/root/datasets/DressCode_1024/upper/mask_cloth/"
    save_limbs_mask_dir = "/root/datasets/DressCode_1024/upper/mask_limbs/"
    save_hands_mask_dir = "/root/datasets/DressCode_1024/upper/mask_hands/"
    if os.path.exists(save_parse_dir):
        shutil.rmtree(save_parse_dir)
    if os.path.exists(save_clo_mask_dir):
        shutil.rmtree(save_clo_mask_dir)
    if os.path.exists(save_limbs_mask_dir):
        shutil.rmtree(save_limbs_mask_dir)
    if os.path.exists(save_hands_mask_dir):
        shutil.rmtree(save_hands_mask_dir)
    os.makedirs(save_parse_dir)
    os.makedirs(save_clo_mask_dir)
    os.makedirs(save_limbs_mask_dir)
    os.makedirs(save_hands_mask_dir)
    # Load image
    img_pil = Image.open(model_path).convert("RGB")
    
    parsing_model = Parsing(0)
    target_shape = (384 , 512) # w h
    model_img_pil = img_pil.resize(target_shape)
    model_parse, _ = parsing_model(model_img_pil.resize(target_shape))
    mask_clothing, mask_limbs, mask_hands = get_masks_for_similarity(model_parse=model_parse)
    mask_clothing.save('mask_clothing.png')
    mask_limbs.save('mask_limbs.png')
    mask_hands.save('mask_hands.png')
    model_parse.save(os.path.join(save_parse_dir, 
                                  os.path.basename(model_path).replace('.jpg', '.png')))
    mask_clothing.save(os.path.join(save_clo_mask_dir, 
                                  os.path.basename(model_path).replace('.jpg', '_mask_clothing.png')))
    mask_limbs.save(os.path.join(save_limbs_mask_dir,
                                  os.path.basename(model_path).replace('.jpg', '_mask_limbs.png')))
    mask_hands.save(os.path.join(save_hands_mask_dir,
                                  os.path.basename(model_path).replace('.jpg', '_mask_hands.png')))
    # Save or display masks                