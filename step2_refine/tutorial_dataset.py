import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm

BASE_SIZE = 64  # Base unit size (must be divisible by 64 for UNet compatibility)


def pad_to_size(img, target_size, pad_value=0):
    """Pad image to target_size with center alignment"""
    h, w = img.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if len(img.shape) == 3:
        return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=pad_value)
    return np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=pad_value)


class MyDataset(Dataset):
    def __init__(self, root_dir='/root/autodl-tmp/dreamnav/', dataset_type='try_train', size_mult=1, image_size=None):
        self.data = []
        if image_size is None or int(image_size) <= 0:
            self.image_size = BASE_SIZE * int(size_mult)
        else:
            self.image_size = int(image_size)
        if self.image_size % 64 != 0:
            raise ValueError(f"image_size must be divisible by 64 for SD/ControlNet, got {self.image_size}")
        print(f"[Stage] Dataset image size: {self.image_size}x{self.image_size} (latent {self.image_size // 8}x{self.image_size // 8})", flush=True)
        self.root = os.path.join(root_dir, dataset_type) + '/'
        print(f"[Stage] Scanning train dataset: {self.root}", flush=True)
        building_ids = sorted(os.listdir(self.root))
        for building_id in tqdm(building_ids, desc=f"Scanning {dataset_type} buildings", dynamic_ncols=True):
            now_dir_path = self.root + building_id + '/'
            for item_id in sorted(os.listdir(now_dir_path)):
                json_path = now_dir_path + item_id
                with open(json_path, "r", encoding="utf-8") as f:
                    now_json = json.load(f)
                image_patha = root_dir + '/tours/' + now_json["image_a"]
                image_pathb = root_dir + '/tours/' + now_json["image_b"]
                self.data.append([image_patha, image_pathb, now_json["heading_num"], now_json["range_num"]])
        print(f"[Stage] Train dataset loaded with {len(self.data)} samples", flush=True)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        source_filename = item[0]
        target_filename = item[1]
        heading_num = item[2]
        range_num = item[3]

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Resize images to image_size x image_size (size_mult * 64)
        source = cv2.resize(source, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, heading_num=heading_num, range_num=range_num, hint=source)

