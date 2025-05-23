import pdb

import numpy as np
import cv2,os
from PIL import Image, ImageDraw

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
}
def get_mask_location_all(model_parse: Image.Image=None,
                             model_parse_path='', width=384, height=512):
    if os.path.exists(model_parse_path):
        model_parse = Image.open(model_parse_path)
    else:
        model_parse = model_parse
    # Resize and convert to numpy array
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    
    # Initialize empty masks
    mask_clothing = np.zeros_like(parse_array, dtype=np.float32)
    mask_clothing_upper = np.zeros_like(parse_array, dtype=np.float32)
    mask_clothing_lower = np.zeros_like(parse_array, dtype=np.float32)
    mask_limbs = np.zeros_like(parse_array, dtype=np.float32)
    mask_hands = np.zeros_like(parse_array, dtype=np.float32)
    
    # Define label groups for each mask
    clothing_labels = [
        "upper_clothes", "skirt", "pants", "dress", 
        "belt", "scarf"
    ]
    upper_labels = [
        "upper_clothes", "dress", 
    ]
    lower_labels = [
        "skirt", "pants" 
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
        if label_name in upper_labels:
            mask_clothing_upper += (parse_array == label_value).astype(np.float32)
        if label_name in lower_labels:
            mask_clothing_lower += (parse_array == label_value).astype(np.float32)
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
    mask_clothing_upper = refine_mask(mask_clothing_upper)
    mask_clothing_lower = refine_mask(mask_clothing_lower)
    mask_limbs = refine_mask(mask_limbs)
    mask_hands = refine_mask(mask_hands)
    
    # Convert to PIL Images
    mask_clothing = Image.fromarray((mask_clothing * 255).astype(np.uint8))
    mask_clothing_upper = Image.fromarray((mask_clothing_upper * 255).astype(np.uint8))
    mask_clothing_lower = Image.fromarray((mask_clothing_lower * 255).astype(np.uint8))
    mask_limbs = Image.fromarray((mask_limbs * 255).astype(np.uint8))
    mask_hands = Image.fromarray((mask_hands * 255).astype(np.uint8))
    
    return mask_clothing,mask_clothing_upper,mask_clothing_lower, \
        mask_limbs, mask_hands

   
def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

def get_mask_location(model_parse: Image.Image, keypoint: dict, width=384,height=512):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    # print(np.array(parse_array).shape) # (512, 384)
    # print(np.unique(parse_array)) # [ 0  2  4  6 11 14 15 18]

    arm_width = 60

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)
    arms = arms_left + arms_right


    parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
    parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                    (parse_array == label_map["pants"]).astype(np.float32)
    parser_mask_fixed += parser_mask_fixed_lower_cloth
    parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    
    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)

    shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
    shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
    elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
    elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
    wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
    wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
    ARM_LINE_WIDTH = int(arm_width / 512 * height)
    size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
    size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,
                    shoulder_right[1] + ARM_LINE_WIDTH // 2]
    

    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
        im_arms_right = arms_right
    else:
        wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
        arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

    if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
        im_arms_left = arms_left
    else:
        wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
        arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

    hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
    hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
    parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    # parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    neck_mask = (parse_array == 18).astype(np.float32)
    # neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
    neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
    parse_mask = np.logical_or(parse_mask, neck_mask)
    # arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
    arm_mask = np.logical_or(im_arms_left, im_arms_right) # 不使用 腐蚀
    parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    # 不选中的：手掌 头 颈
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    
    # parse_mask_total 这里 没有 cloth 和 arm
    
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray


 


def get_mask_location_all2(model_parse: Image.Image, width=384,height=512):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    # print(np.array(parse_array).shape) # (512, 384)
    # print(np.unique(parse_array)) # [ 0  2  4  6 11 14 15 18]

    # arm_width = 60
    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)
    parse_human = 1 - parser_mask_changeable

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 2).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)
                 
    parse_cloth = 1-(parser_mask_changeable+parse_head)

    '''
    膨胀（dilate） 和 腐蚀（erode）
    
    开操作 是先进行腐蚀操作，再进行膨胀操作的形态学操作。
    腐蚀：先对输入图像进行腐蚀操作，去除小的噪声和细小的白色区域。
    膨胀：然后对经过腐蚀后的图像进行膨胀操作，恢复目标区域的大小，但噪声和小物体已经被去除。
    
    闭操作能填补图像中的小黑洞或细小的黑色区域，同时保留大部分的目标区域。
    '''
    
    
    
    # human_mask = cv2.dilate(parse_human, 
    #                         np.ones((5, 5), np.uint16), 
    #                         iterations=3)
    # head_mask = cv2.dilate(parse_head, 
    #                         np.ones((5, 5), np.uint16), 
    #                         iterations=3)
    # cloth_mask = cv2.dilate(parse_cloth, 
    #                         np.ones((5, 5), np.uint16), 
    #                         iterations=3)
    # 定义结构元素，通常是一个方形或圆形的内核
    # kernel = np.ones((3, 3), np.uint8)

    # 使用开操作：先腐蚀，再膨胀   填补小白
    # inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_OPEN, kernel)
    # 使用闭操作：先膨胀，再腐蚀    填补小黑
    # inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_CLOSE, kernel)
    
    def get_mask(inpaint_mask):
        img = np.where(inpaint_mask, 255, 0)
        dst = hole_fill(img.astype(np.uint8))
        dst = refine_mask(dst)
        inpaint_mask = dst / 255 * 1
        mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
        return mask
    # parse_mask_total 这里 没有 cloth 和 arm
    
    # mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)
    human_mask = get_mask(parse_human)
    head_mask = get_mask(parse_head)
    cloth_mask = get_mask(parse_cloth)

    return human_mask,head_mask,cloth_mask