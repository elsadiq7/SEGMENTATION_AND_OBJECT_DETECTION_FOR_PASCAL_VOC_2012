"""
PASCAL VOC Dataset Utilities
===========================

This module provides dataset handling utilities for both object detection and 
semantic segmentation tasks using the PASCAL VOC dataset.

Key Components:
--------------
1. Object Detection Dataset (VOCDataset_OD)
   - Custom dataset class for object detection
   - Bounding box and label handling
   - Image resizing and preprocessing

2. Segmentation Dataset (VocDataset)
   - Custom dataset class for semantic segmentation
   - Multi-channel mask conversion
   - Color map handling

3. Utility Functions
   - Data downloading
   - Custom collate function for batching
   - Image and mask preprocessing

Dependencies:
------------
Standard Library:
    - os: File and directory operations

Third-party Libraries:
    - cv2: Image processing
    - numpy: Numerical operations
    - matplotlib: Visualization
    - PIL: Image handling

PyTorch Ecosystem:
    - torch: Deep learning framework
    - torchvision: Vision datasets and transforms

Author: Abdulrahman 
Version: 1.0
"""

# Standard library imports
import os

# Third-party library imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader

# Torchvision imports
from torchvision import transforms
from torchvision.datasets import VOCSegmentation, VOCDetection
from torchvision.transforms import functional as F

def download_data_OD(path):
    """
    Download and load the Pascal VOC dataset for object detection.
    
    Args:
        path (str): Directory path to store the dataset
        download (bool): Whether to download the dataset if not present
    
    Returns:
        None
        
    Features:
        - Checks for existing dataset
        - Downloads both training and validation sets
        - Handles download status automatically
    """
    download=True;
    # If downloading, remove the existing directory
    if os.path.exists(path):
        print(f"Data is already existed: {path}")
        download=False

    # Load Pascal VOC dataset
    VOCDetection(root=path, year='2012', image_set='train',download=download)
    VOCDetection(root=path, year='2012', image_set='val',download=download)


def voc_collate_fn(batch):
    """
    Collate function for VOC object detection dataset.
    
    Args:
        batch (List[tuple]): List of (image, target) pairs
        
    Returns:
        tuple: (list of images, list of targets)
        
    Features:
        - Maintains original target dictionary structure
        - Handles variable number of objects per image
        - Preserves individual image dimensions
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)  # Each target is a dictionary with bounding boxes & labels
    
    return images, targets  # Keep lists instead of stacking tensors















class VOCDataset_OD(Dataset):
    """
    Custom Dataset class for Pascal VOC Object Detection.
    
    Args:
        path (str): Directory containing the VOC dataset
        year (str): Dataset year (default: "2012")
        image_set (str): Dataset split ("train" or "val")
        resize (tuple): Image resize dimensions (height, width)
        download (bool): Whether to download dataset
        
    Features:
        - Handles image loading and preprocessing
        - Scales bounding boxes to resized dimensions
        - Converts annotations to tensor format
        - Supports 20 VOC classes
    """
    def __init__(self, path, year="2012", image_set="train", resize=(256, 256), download=False):
        self.dataset = VOCDetection(root=path, year=year, image_set=image_set, download=download)
        self.resize = resize  # (height, width)

        self.class_to_idx = {
            "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
            "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10,
            "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
            "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20
        }

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, target_dict) where target_dict contains:
                - boxes (torch.Tensor): Scaled bounding boxes
                - labels (torch.Tensor): Class labels
                
        Features:
            - Converts RGB images to tensors
            - Scales bounding boxes to match resized images
            - Handles multiple objects per image
        """
        img, target = self.dataset[idx]
        img = img.convert("RGB")

        annotation = target["annotation"]
        orig_width = int(annotation["size"]["width"])
        orig_height = int(annotation["size"]["height"])

        # Extract bounding boxes and labels
        labels = []
        boxes = []
        for obj in annotation["object"]:
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            label = self.class_to_idx[obj["name"]]

            # Scale bounding boxes to new image size
            new_xmin = (float(xmin) / float(orig_width)) * float(self.resize[1])
            new_ymin = (float(ymin) / float(orig_height)) * float(self.resize[0])
            new_xmax = (float(xmax) / float(orig_width)) * float(self.resize[1])
            new_ymax = (float(ymax) / float(orig_height)) * float(self.resize[0])
            boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
            labels.append(label)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target_dict = {"boxes": boxes, "labels": labels}

        # Apply transformations
        img = F.resize(img, self.resize)
        img = F.to_tensor(img)

        return img, target_dict









class VocDataset(Dataset):
    """
    Custom Dataset class for Pascal VOC Semantic Segmentation.
    
    Args:
        dir (str): Root directory containing VOC dataset
        color_map (List): RGB values for each class
        resize_dim (tuple): Target image dimensions (height, width)
        transform (callable): Optional transforms to be applied
        
    Features:
        - Supports semantic segmentation tasks
        - Handles RGB to multi-channel mask conversion
        - Maintains class color mapping
        - Supports custom transformations
    """
    def __init__(self, dir, color_map, resize_dim=(256, 256), transform=None):
        """
        PASCAL VOC 2012 Semantic Segmentation Dataset.

        Args:
            dir (str): Path to VOC dataset root directory.
            color_map (list): List of RGB values for each class in the dataset.
            resize_dim (tuple): Target image resize dimensions (height, width).
            transform (callable, optional): Optional transformation to be applied to both image and mask.
        """
        self.root = os.path.join(dir, 'VOCdevkit/VOC2012')
        self.target_dir = os.path.join(self.root, 'SegmentationObject')  # Changed to instance segmentation
        self.images_dir = os.path.join(self.root, 'JPEGImages')

        # Load image file names
        file_list = os.path.join(self.root, 'ImageSets/Segmentation/trainval.txt')
        with open(file_list, "r") as f:
            self.files = [line.strip() for line in f]

        self.color_map = np.array(color_map, dtype=np.uint8)
        self.resize_dim = resize_dim
        self.transform = transform

    def convert_to_segmentation_mask(self, mask):
        """
        Convert RGB mask to multi-channel segmentation mask.
        
        Args:
            mask (np.ndarray): RGB segmentation mask array
            
        Returns:
            np.ndarray: Multi-channel segmentation mask (H, W, num_classes)
            
        Features:
            - Efficient numpy-based conversion
            - Handles all dataset classes
            - Maintains spatial dimensions
        """
        # Use numpy's advanced indexing for efficient processing
        segmentation_mask = np.zeros((mask.shape[0], mask.shape[1], len(self.color_map)), dtype=np.float32)
        for i, color in enumerate(self.color_map):
            segmentation_mask[:, :, i] = np.all(mask == color, axis=-1).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, segmentation_mask_tensor)
            
        Features:
            - Loads and processes both image and mask
            - Applies consistent resizing
            - Converts to appropriate tensor format
            - Handles optional transformations
        """
        image_id = self.files[index]
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        label_path = os.path.join(self.target_dir, f"{image_id}.png")

        # Load and process image
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, self.resize_dim)
        image = transforms.ToTensor()(image)  # Converts to range [0, 1] and channels-first format

        # Load and process label mask
        label = cv.imread(label_path)
        label = cv.cvtColor(label, cv.COLOR_BGR2RGB)
        label = cv.resize(label, self.resize_dim, interpolation=cv.INTER_NEAREST)
        label = self.convert_to_segmentation_mask(label)

        label = torch.tensor(label, dtype=torch.float32).permute(2, 0, 1)  # Channels-first format

        # Apply any additional transformations if provided
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
        
    def __len__(self):
        return len(self.files)
 




