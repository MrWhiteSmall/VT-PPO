from util_mask_for_reward import get_parse_mask_for_similarity,target_shape
import PIL,os
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import cv2
from torchvision.transforms import ToTensor
from sklearn.metrics.pairwise import cosine_similarity

weights={'clothing': 0.5, 'limbs': 0.2, 'hands': 0.3}
def get_bounding_rectangles(mask: np.ndarray):
    """
    获取二值掩码中所有白色区域的外接矩形坐标
    
    Args:
        mask: 二值化掩码图像（0为黑色背景，255为白色目标区域）
        
    Returns:
        List[Tuple[x, y, w, h]]: 外接矩形列表，每个矩形格式为(x, y, width, height)
    """
    # 确保输入是单通道二值图像
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理（确保只有0和255）
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取所有外接矩形
    rectangles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rectangles.append((x, y, w, h))
    
    return rectangles
# 计算各区域的相似度
def masked_cosine(img1, img2, mask:PIL.Image):
    """计算掩码区域的余弦相似度"""
    '''
    直接计算 误差太大
    mask中先把1的区域抠出来变成外接矩形 mask_rect=[]
    for m_res in mask_rect:
        分别把img1 img2的对应区域抠出来
        计算余弦相似度
    '''
    mask_rect = get_bounding_rectangles(np.array(mask))
    all_sim = []
    
    # mask_tensor = ToTensor()(mask).unsqueeze(0)  # [1,1,H,W]
    # masked1 = img1 * mask_tensor  # [1,3,H,W]
    # masked2 = img2 * mask_tensor  # [1,3,H,W]
    
    # # 展平后计算余弦相似度
    # flat1 = masked1.view(1, -1).numpy()  # [1, 3*H*W]
    # flat2 = masked2.view(1, -1).numpy()
    # sim = cosine_similarity(flat1, flat2)[0][0]
    
    # 计算每个外接矩形的相似度
    for rect in mask_rect:
        x, y, w, h = rect
        # 提取对应区域
        img1_region = img1[:, :, y:y+h, x:x+w].contiguous()
        img2_region = img2[:, :, y:y+h, x:x+w].contiguous()
        
        # 计算余弦相似度
        flat1 = img1_region.view(1, -1).numpy()
        flat2 = img2_region.view(1, -1).numpy()
        sim = 1-cosine_similarity(flat1, flat2)[0][0]
        all_sim.append(sim)
    # 每个sim需要根据区域大小进行加权
    weights = []
    for rect in mask_rect:
        x, y, w, h = rect
        area = w * h
        weights.append(area)
    # 归一化权重
    weights = np.array(weights) / np.sum(weights)
    # 计算加权平均相似度
    all_sim_np = np.array(all_sim)
    similarity = np.dot(weights, all_sim_np)

    # 计算所有区域加权求和后的相似度
    return similarity if all_sim else 0.0


def get_similarity(model_path='',
                   model_pred:PIL.Image=None,):
    assert os.path.exists(model_path) , "model_path must be provided"
    assert model_pred is not None, "model_pred must be provided"
    output_dir = os.path.dirname(os.path.dirname(model_path))
    name = os.path.basename(model_path).split('.')[0]
    
    model = Image.open(model_path).convert("RGB")
    model = model.resize(target_shape)
    model_pred = model_pred.resize(target_shape)
    model_parse, mask_clothing, mask_limbs, mask_hands = \
                            get_parse_mask_for_similarity(model_path)
    del model_parse
    '''
    model · mask_clothing (sim) model_pred · mask_clothing
    model · mask_limbs (sim) model_pred · mask_limbs
    model · mask_hands (sim) model_pred · mask_hands
    
    return similarity
    '''
    
    model_tensor = ToTensor()(model).unsqueeze(0)  # [1,3,H,W]
    model_pred_tensor = ToTensor()(model_pred).unsqueeze(0)  # [1,3,H,W]
    

    # 各区域相似度计算
    sim_clothing = masked_cosine(model_tensor, model_pred_tensor, mask_clothing)
    sim_limbs = masked_cosine(model_tensor, model_pred_tensor, mask_limbs)
    sim_hands = masked_cosine(model_tensor, model_pred_tensor, mask_hands)
    
    # 加权综合相似度
    clo_sim  = weights['clothing'] * sim_clothing
    limbs_sim = weights['limbs'] * sim_limbs
    hands_sim = weights['hands'] * sim_hands
    total_sim = clo_sim + limbs_sim + hands_sim
    # print(total_sim)
    # 数值截断保证在[0,1]范围内
    
    '''
    展示 并  保存  output_dir/name_{sim}.jpg
    modelimg model_clothing_img model_limbs_img model_hands_img
    model_pred_img model_pred_clothing_img model_pred_limbs_img model_pred_hands_img
    二行四列展示
    最后一行  下面写上 对应的相似度
    
    先把图片转换为图片数值分布
    然后PIL拼接到一起 二行四列
    然后再新增一行数据到最下面
    然后保存
    '''
    # Prepare images (convert to numpy arrays)
    def apply_mask(img_tensor, mask):
        mask_tensor = ToTensor()(mask).unsqueeze(0)  # [1,1,H,W]
        return (img_tensor * mask_tensor).squeeze().permute(1,2,0).numpy()
    
    images = {
        'Original': model_tensor.squeeze().permute(1,2,0).numpy(),
        'Original Clothing': apply_mask(model_tensor, mask_clothing),
        'Original Limbs': apply_mask(model_tensor, mask_limbs),
        'Original Hands': apply_mask(model_tensor, mask_hands),
        'Predicted': model_pred_tensor.squeeze().permute(1,2,0).numpy(),
        'Predicted Clothing': apply_mask(model_pred_tensor, mask_clothing),
        'Predicted Limbs': apply_mask(model_pred_tensor, mask_limbs),
        'Predicted Hands': apply_mask(model_pred_tensor, mask_hands)
    }

    # Create 2x4 grid
    rows = []
    for i in range(2):
        row_images = []
        for j in range(4):
            idx = i*4 + j
            key = list(images.keys())[idx]
            img = (images[key] * 255).astype('uint8')
            row_images.append(Image.fromarray(img))
        rows.append(np.hstack([np.array(img) for img in row_images]))
    
    # Combine rows
    grid = np.vstack(rows)
    result = Image.fromarray(grid)

    # Add text
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    text = (f"Similarity: Clothing={clo_sim:.2f}(x{weights['clothing']}) | "
            f"Limbs={limbs_sim:.2f}(x{weights['limbs']}) | "
            f"Hands={hands_sim:.2f}(x{weights['hands']}) | "
            f"Total={total_sim:.2f}")
    draw.text((10, grid.shape[0]-30), text, fill='white', font=font)

    # Save
    output_dir = os.path.join(output_dir, 'similarity_results_grid')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{total_sim:.2f}.jpg")
    result.save(save_path)
    # print(f"Saved to {save_path}")
    
    return max(0.0, min(1.0, total_sim))

if __name__ == "__main__":
    model_path = '/root/datasets/DressCode_1024/upper/image/000001_0.jpg'
    # model_pred = Image.open('path/to/model_pred').convert("RGB")
    model_pred = Image.open(model_path).convert("RGB")
    similarity = get_similarity(model_path, model_pred)
    print(f"Similarity: {similarity:.4f}")