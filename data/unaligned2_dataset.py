import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
import cv2
import numpy as np
import torch
import json
from torch.utils.data import Dataset

class Unaligned2Dataset(BaseDataset):
    """
    Modified dataset class for paired high/low quality images.
    Directory structure:
    - Training: 
        train_dataset/
            class_1/
                high_quality/  # Domain B 
                low_quality/   # Domain A 
            ...
            class_5/
                high_quality/
                low_quality/
    - Testing:
        test_dataset/
            high_quality/  # Domain B 
            low_quality/   # Domain A 
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.phase = opt.phase
        self.data_root = opt.dataroot if opt.phase == "train" else opt.testdataroot
        
        # Collect all image paths
        self.A_paths = []  # low quality (changed from high)
        self.B_paths = []  # high quality (changed from low)
        self.labels = []   # only for training

        if self.phase == "train":
            # Training data: organized by class folders
            self.classes = sorted([d for d in os.listdir(self.data_root) 
                                 if os.path.isdir(os.path.join(self.data_root, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            for class_name in self.classes:
                class_path = os.path.join(self.data_root, class_name)
                label = self.class_to_idx[class_name] + 1
                
                high_dir = os.path.join(class_path, 'high_quality')  # Domain B
                low_dir = os.path.join(class_path, 'low_quality')    # Domain A
                
                low_images = sorted([os.path.join(low_dir, f) 
                                   for f in os.listdir(low_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                high_images = sorted([os.path.join(high_dir, f) 
                                    for f in os.listdir(high_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                # Pair images (assuming equal count per class)
                for low_img, high_img in zip(low_images, high_images):
                    self.A_paths.append(low_img)
                    self.B_paths.append(high_img)
                    self.labels.append(label)
        else:
            # Testing data: flat directory structure
            high_dir = os.path.join(self.data_root, 'high_quality')  # Domain B
            low_dir = os.path.join(self.data_root, 'low_quality')    # Domain A
            
            self.A_paths = sorted([os.path.join(low_dir, f) 
                                 for f in os.listdir(low_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            self.B_paths = sorted([os.path.join(high_dir, f) 
                                 for f in os.listdir(high_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Ensure paired images (optional: add checks if filenames should match)
            assert len(self.A_paths) == len(self.B_paths), \
                   "Mismatched number of high/low quality images in test data."

        # Data normalization info (shared for train/test)
        if opt.data_norm == 'ab_seperate':
            norm_path = os.path.join(opt.dataroot, "data.json")  # Assume stats from training
            if os.path.exists(norm_path):
                with open(norm_path, "r") as f:
                    self.norm_stats = json.load(f)
            else:
                self.norm_stats = None

    def __getitem__(self, index):
        A_path = self.A_paths[index]  # low quality
        B_path = self.B_paths[index]  # high quality
        
        # Load images (grayscale)
        A_img = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE)
        B_img = cv2.imread(B_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if self.opt.resizeBig:
            A_img = cv2.resize(A_img, (512, 512))
            B_img = cv2.resize(B_img, (512, 512))
        
        # Add channel dimension
        A_img_arr = np.array(A_img).reshape((1,) + A_img.shape)
        B_img_arr = np.array(B_img).reshape((1,) + B_img.shape)
        
        # Random flip (training only)
        if self.phase == "train" and self.opt.lr_flip and random.random() < 0.5:
            A_img_arr = np.flip(A_img_arr, axis=2)
            B_img_arr = np.flip(B_img_arr, axis=2)
        
        # Normalize
        if self.opt.data_norm == 'basic':
            A_img_arr = ((A_img_arr / 255.) * 2) - 1
            B_img_arr = ((B_img_arr / 255.) * 2) - 1
        elif self.opt.data_norm == 'ab_seperate' and self.norm_stats:
            A_img_arr = ((A_img_arr / 255.) - self.norm_stats['TrainA'][0]) / self.norm_stats['TrainA'][1]
            B_img_arr = ((B_img_arr / 255.) - self.norm_stats['TrainB'][0]) / self.norm_stats['TrainB'][1]
        
        # Convert to tensor
        A = torch.from_numpy(A_img_arr).float()
        B = torch.from_numpy(B_img_arr).float()
        
        if self.phase == "train":
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}, self.labels[index]
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)
