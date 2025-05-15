import os
from PIL import Image
from torch.utils.data import Dataset
from diffusers.image_processor import VaeImageProcessor

gpu='cuda:0'

'''
def __init__(
    self,
    do_resize: bool = True,
    vae_scale_factor: int = 8,
    resample: str = "lanczos",
    do_normalize: bool = True,
    do_convert_rgb: bool = False,
):
'''
class InferenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
    
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8) 
        self.mask_processor = VaeImageProcessor(
                            do_resize=True,          # 显式启用缩放（默认True，可省略）
                            vae_scale_factor=8,      # 保持和旧版一致（默认已是8）
                            do_normalize=False,      # 覆盖默认值True
                            do_convert_rgb=False,    # 等效于旧版的do_convert_grayscale=True
                            resample="lanczos"       # 明确指定重采样方法（默认已是lanczos）
                            ) 
        self.data = self.load_data()
    
    def load_data(self):
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        person, cloth, mask = [Image.open(data[key]) for key in ['person', 'cloth', 'mask']]
        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        }

class VITONHDTestDataset(InferenceDataset):
    def load_data(self):
        assert os.path.exists(pair_txt:=os.path.join(self.args.data_root_path, 'test_pairs_unpaired.txt')), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        self.args.data_root_path = os.path.join(self.args.data_root_path, "test")
        output_dir = os.path.join(self.args.output_dir, "vitonhd", 'unpaired' if not self.args.eval_pair else 'paired')
        data = []
        for line in lines:
            person_img, cloth_img = line.strip().split(" ")
            if os.path.exists(os.path.join(output_dir, person_img)):
                continue
            if self.args.eval_pair:
                cloth_img = person_img
            data.append({
                'person_name': person_img,
                'person': os.path.join(self.args.data_root_path, 'image', person_img),
                'cloth': os.path.join(self.args.data_root_path, 'cloth', cloth_img),
                'mask': os.path.join(self.args.data_root_path, 'agnostic-mask', person_img.replace('.jpg', '_mask.png')),
            })
        return data

class DressCodeTestDataset(InferenceDataset):
    def load_data(self):
        data = []
        for sub_folder in ['upper', 'lower', 'dresses']:
            assert os.path.exists(os.path.join(self.args.data_root_path, sub_folder)), f"Folder {sub_folder} does not exist."
            pair_txt = os.path.join(self.args.data_root_path, 'test_pairs_paired.txt' if self.args.eval_pair else 'test_pairs_unpaired.txt')
            assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
            # with open(pair_txt, 'r') as f:
            #     lines = f.readlines()
            with open(pair_txt, 'r') as f:
                lines = f.readlines()
                lines = [line for line in lines if line.strip().split(" ")[-1]==sub_folder]
                print(sub_folder,len(lines))   

            output_dir = os.path.join(self.args.output_dir, f"dresscode-{self.args.height}", 
                                      'unpaired' if not self.args.eval_pair else 'paired', sub_folder)
            for line in lines:
                person_img, cloth_img,_ = line.strip().split(" ")
                person_img = person_img.replace('.png','.jpg')
                cloth_img = cloth_img.replace('.png','.jpg')
                if os.path.exists(os.path.join(output_dir, person_img)):
                    continue
                data.append({
                    'person_name': os.path.join(sub_folder, person_img),
                    'person': os.path.join(self.args.data_root_path, sub_folder, 'image', person_img),
                    'cloth': os.path.join(self.args.data_root_path, sub_folder, 'cloth', cloth_img),
                    'mask': os.path.join(self.args.data_root_path, sub_folder, 'agnostic_masks', person_img.replace('.jpg', '.png'))
                })
        return data
               