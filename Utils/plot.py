"""
Visualization Utilities for Object Detection and Segmentation
=========================================================

This module provides comprehensive visualization tools for displaying and analyzing
results from object detection and segmentation models, including bounding boxes,
segmentation masks, and training metrics.

Key Features:
------------
- Bounding box visualization with ground truth and predictions
- Segmentation mask overlay
- Training history plots
- Batch visualization
- Non-Maximum Suppression (NMS) visualization
- Detailed metric plotting

Dependencies:
------------
- torch
- torchvision
- matplotlib
- numpy
- PIL
- logging

Usage:
------
This module can be used to:
1. Visualize detection results with bounding boxes
2. Display segmentation masks and predictions
3. Plot training metrics and learning curves
4. Analyze model performance visually

Author:Abdulrahman Elsadiq
Version: 1.0
"""

import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage
import torchvision.ops as ops
import numpy as np
from typing import Optional, Union, List, Dict
import warnings
import logging

# Configure matplotlib
# plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150

# Constants
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)

NUM_CLASSES = len(VOC_COLORMAP)
class_names = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
}


import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage
import random
import numpy as np 
import torch
import torchvision.ops as ops
import torch
import random
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage

import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage

# Pascal VOC class names mapping
class_names = {
    1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
    6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
    16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"
}

# Define Pascal VOC colors for classes
voc_colors = [
    "red", "blue", "green", "yellow", "purple", "orange", "cyan", "pink",
    "lime", "brown", "magenta", "gray", "navy", "maroon", "olive",
    "teal", "aqua", "gold", "violet", "indigo"
]

def get_color(label):
    """Assigns a unique color based on class label."""
    return voc_colors[label % len(voc_colors)]


def plot_batch_with_bboxes(images, targets, predictions=None):
    batch_size = min(len(images), 4)
    cols = 2 if predictions else 1  # Two columns if predictions are provided
    
    fig, axes = plt.subplots(batch_size, cols, figsize=(15, batch_size * 6), dpi=150)
    
    # Ensure axes is always iterable
    if batch_size == 1:
        axes = np.array([axes])
    if predictions:
        axes = axes.reshape(batch_size, 2)  # Reshape for proper indexing
    
    for i in range(batch_size):
        img = (images[i] * 255).clamp(0, 255).to(torch.uint8)
        
        # Process ground truth
        gt_boxes = targets[i]["boxes"]
        gt_labels = targets[i]["labels"]
        gt_label_names = [class_names.get(label.item(), f"class_{label.item()}") for label in gt_labels]
        img_with_gt = draw_bounding_boxes(img, gt_boxes, labels=gt_label_names, colors=['white'] * len(gt_labels), width=4)
        
        axes[i, 0].imshow(ToPILImage()(img_with_gt))
        axes[i, 0].set_title("Ground Truth", fontsize=14, pad=10)
        axes[i, 0].axis("off")
        
        # Process predictions if available
        if predictions:
            pred_boxes = predictions[i]["boxes"]
            pred_labels = predictions[i]["labels"]
            pred_scores = predictions[i]["scores"]
            pred_label_names = [f"{class_names.get(label.item(), f'class_{label.item()}')} ({score:.2f})" for label, score in zip(pred_labels, pred_scores)]
            img_with_pred = draw_bounding_boxes(img, pred_boxes, labels=pred_label_names, colors=['red'] * len(pred_labels), width=4)
            
            axes[i, 1].imshow(ToPILImage()(img_with_pred))
            axes[i, 1].set_title("Predictions", fontsize=14, pad=10)
            axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.show()


def apply_nms(predictions, iou_threshold=0.5, score_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
    
    Args:
        predictions (List[Dict]): List of prediction dictionaries containing:
            - 'boxes' (torch.Tensor): Predicted bounding boxes (N, 4)
            - 'scores' (torch.Tensor): Confidence scores
            - 'labels' (torch.Tensor): Class labels
        iou_threshold (float): IoU threshold for NMS (default: 0.5)
        score_threshold (float): Confidence score threshold (default: 0.5)
    
    Returns:
        List[Dict]: Filtered predictions with overlapping boxes removed
        
    Features:
        - Class-wise NMS application
        - Confidence score filtering
        - Sorting by confidence scores
        - Handles empty predictions gracefully
    """
    filtered_preds = []
    
    for pred in predictions:
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        
        # Filter by confidence score
        high_scores_idx = scores > score_threshold
        boxes = boxes[high_scores_idx]
        scores = scores[high_scores_idx]
        labels = labels[high_scores_idx]
        
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        
        # Process each class separately
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_labels = labels[class_mask]
            
            # Apply NMS with stricter threshold for better filtering
            keep_idx = ops.nms(class_boxes, class_scores, iou_threshold)
            
            filtered_boxes.append(class_boxes[keep_idx])
            filtered_scores.append(class_scores[keep_idx])
            filtered_labels.append(class_labels[keep_idx])
        
        # Combine results
        if filtered_boxes:
            filtered_boxes = torch.cat(filtered_boxes)
            filtered_scores = torch.cat(filtered_scores)
            filtered_labels = torch.cat(filtered_labels)
            
            # Sort by confidence score
            sorted_idx = torch.argsort(filtered_scores, descending=True)
            
            filtered_preds.append({
                "boxes": filtered_boxes[sorted_idx],
                "scores": filtered_scores[sorted_idx],
                "labels": filtered_labels[sorted_idx]
            })
        else:
            filtered_preds.append({
                "boxes": torch.empty((0, 4), device=boxes.device),
                "scores": torch.empty(0, device=scores.device),
                "labels": torch.empty(0, device=labels.device)
            })

    return filtered_preds

def visualize_test_batch(model, test_loader, device, iou_threshold=0.5, score_threshold=0.7):
    """
    Visualizes model predictions on a test batch with NMS applied.
    
    Args:
        model (torch.nn.Module): Trained object detection model
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        device (str): Device to run inference on ('cuda' or 'cpu')
        iou_threshold (float): IoU threshold for NMS (default: 0.5)
        score_threshold (float): Confidence score threshold (default: 0.7)
    
    Returns:
        None: Displays the visualization using matplotlib
        
    Features:
        - Automatic batch processing
        - Integrated NMS filtering
        - Side-by-side comparison of ground truth and predictions
        - GPU/CPU compatibility
    """
    model.eval()
    images, targets = next(iter(test_loader))
    images = [img.to(device) for img in images]
    
    with torch.no_grad():
        predictions = model(images)
    
    pred_list = []
    for pred in predictions:
        pred_list.append({
            "boxes": pred["boxes"].cpu(),
            "labels": pred["labels"].cpu(),
            "scores": pred["scores"].cpu()
        })
    
    filtered_predictions = apply_nms(
        pred_list, 
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )
    
    targets = [{
        "boxes": t["boxes"].cpu(), 
        "labels": t["labels"].cpu()
    } for t in targets]
    
    plot_batch_with_bboxes(images, targets, filtered_predictions)

###################################################################################################################

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from typing import Optional, Union
import warnings

# Define VOC colormap as a constant
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], 
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], 
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
], dtype=np.uint8)

NUM_CLASSES = len(VOC_COLORMAP)

def validate_mask(mask: np.ndarray) -> None:
    """
    Validate the input mask format and values.
    
    Args:
        mask (np.ndarray): Input segmentation mask
    
    Raises:
        ValueError: If mask format or values are invalid
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Mask must be a numpy array, got {type(mask)}")
    
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    
    if mask.min() < 0 or mask.max() >= NUM_CLASSES:
        raise ValueError(f"Mask values must be between 0 and {NUM_CLASSES-1}, "
                        f"got min={mask.min()}, max={mask.max()}")

def decode_segmentation_mask(mask: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Convert a segmentation mask with class indices to a color image.
    
    Args:
        mask (np.ndarray): Input segmentation mask (H, W)
        debug (bool): Whether to print debug information
    
    Returns:
        np.ndarray: Colored mask (H, W, 3)
    """
    validate_mask(mask)
    
    if debug:
        print(f"Mask shape: {mask.shape}")
        print(f"Unique classes in mask: {np.unique(mask)}")
    
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id in range(NUM_CLASSES):
        class_mask = (mask == class_id)
        color_mask[class_mask] = VOC_COLORMAP[class_id]
    
    if debug:
        print(f"Unique colors in output: {np.unique(color_mask.reshape(-1, 3), axis=0)}")
    
    return color_mask

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to range [0, 1].
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        np.ndarray: Normalized image
    """
    if image.max() > 1.0:
        return image / 255.0
    return image

def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay segmentation mask on image with transparency.
    
    Args:
        image (np.ndarray): Original image (H, W, 3), values in [0,1]
        mask (np.ndarray): Segmentation mask (H, W, 3)
        alpha (float): Transparency (0 = invisible, 1 = fully visible)
    
    Returns:
        np.ndarray: Blended image
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    image = normalize_image(image)
    mask = normalize_image(mask)
    
    overlay = (1 - alpha) * image + alpha * mask
    return np.clip(overlay, 0, 1)

def visualize_samples_seg(dataloader, num_samples=5, model=None, device="cuda", alpha=0.5, debug=False):
    """
    Visualizes segmentation masks and model predictions for random samples.
    
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing (image, mask) pairs
        num_samples (int): Number of samples to visualize (default: 5)
        model (torch.nn.Module, optional): Trained segmentation model
        device (str): Device for model inference (default: "cuda")
        alpha (float): Transparency for mask overlay (default: 0.5)
        debug (bool): Enable debug printing (default: False)
    
    Returns:
        None: Displays the visualization using matplotlib
        
    Features:
        - Random sample selection
        - Support for both binary and multi-class segmentation
        - Optional model prediction visualization
        - Customizable transparency
        - Debug mode for troubleshooting
    """
    if not dataloader.dataset:
        raise ValueError("Empty dataloader")
    
    num_cols = 4 if model else 3
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(12, num_samples * 3))
    
    if model:
        model.eval()
        if debug:
            print(f"Model device: {next(model.parameters()).device}")
    
    try:
        # Get batch of data
        images, masks = next(iter(dataloader))
        
        if debug:
            print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
        
        # Ensure we have enough samples
        batch_size = len(images)
        if batch_size < num_samples:
            warnings.warn(f"Requested {num_samples} samples but batch only has {batch_size}")
            num_samples = batch_size
        
        indices = random.sample(range(batch_size), num_samples)
        
        for i, idx in enumerate(indices):
            image, mask = images[idx], masks[idx]
            
            # Convert image to numpy
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = normalize_image(image_np)
            
            # Convert mask to proper format
            if mask.ndim == 3 and mask.shape[0] == NUM_CLASSES:
                mask_np = mask.argmax(0).cpu().numpy()
            else:
                mask_np = mask.cpu().numpy()
            
            # Generate colored mask
            color_mask = decode_segmentation_mask(mask_np, debug=debug)
            blended_image = overlay_mask_on_image(image_np, color_mask, alpha)
            
            # Plot original image
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f"Image {idx}")
            axes[i, 0].axis("off")
            
            # Plot ground truth mask
            axes[i, 1].imshow(color_mask)
            axes[i, 1].set_title(f"Ground Truth {idx}")
            axes[i, 1].axis("off")
            
            # Plot overlay
            axes[i, 2].imshow(blended_image)
            axes[i, 2].set_title(f"Overlay {idx}")
            axes[i, 2].axis("off")
            
            # Generate and plot predictions if model is provided
            if model:
                with torch.no_grad():
                    input_tensor = image.unsqueeze(0).to(device)
                    pred_logits = model(input_tensor)
                    pred_mask = pred_logits.argmax(1).squeeze(0).cpu().numpy()
                
                pred_color_mask = decode_segmentation_mask(pred_mask, debug=debug)
                pred_blended = overlay_mask_on_image(image_np, pred_color_mask, alpha)
                
                axes[i, 3].imshow(pred_blended)
                axes[i, 3].set_title(f"Prediction {idx}")
                axes[i, 3].axis("off")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Error during visualization: {str(e)}")


def plot_detailed_history(train_history, test_history, save_path=None, metrics_to_plot=None):
    """
    Creates detailed plots of training and validation metrics history.
    
    Args:
        train_history (List[Dict]): Training metrics for each epoch
        test_history (List[Dict]): Validation metrics for each epoch
        save_path (str, optional): Path to save the plot
        metrics_to_plot (List[str], optional): Specific metrics to plot
    
    Returns:
        None: Displays and optionally saves the plot
        
    Features:
        - Multiple metric support
        - Automatic best value highlighting
        - Value annotations
        - Customizable metrics selection
        - High-resolution export option
        - Grid and styling optimization
    """
    epochs = range(1, len(train_history) + 1)
    
    # Get available metrics
    available_metrics = train_history[0].keys()
    if metrics_to_plot is None:
        metrics_to_plot = available_metrics
    else:
        # Verify requested metrics are available
        for metric in metrics_to_plot:
            if metric not in available_metrics:
                raise ValueError(f"Metric '{metric}' not found in history")
    
    num_metrics = len(metrics_to_plot)
    
    # Create figure
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5*num_metrics))
    if num_metrics == 1:
        axes = [axes]
    
    
    # Plot each metric
    for idx, metric in enumerate(metrics_to_plot):
        # Move tensors to CPU and convert to numpy
        train_values = [epoch[metric].cpu().numpy() if torch.is_tensor(epoch[metric]) 
                       else epoch[metric] for epoch in train_history]
        test_values = [epoch[metric].cpu().numpy() if torch.is_tensor(epoch[metric]) 
                      else epoch[metric] for epoch in test_history]
        
        # Plot lines
        axes[idx].plot(epochs, train_values, 'b-', label=f'Training {metric}', linewidth=2)
        axes[idx].plot(epochs, test_values, 'r-', label=f'Validation {metric}', linewidth=2)
        
        # Add scatter points
        axes[idx].scatter(epochs, train_values, c='blue', s=50)
        axes[idx].scatter(epochs, test_values, c='red', s=50)
        
        # Styling
        axes[idx].set_title(f'Training and Validation {metric.upper()}', fontsize=14, pad=15)
        axes[idx].set_xlabel('Epochs', fontsize=12)
        axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[idx].legend(fontsize=12)
        axes[idx].grid(True)
        
        # Add best values
        best_train = min(train_values) if 'loss' in metric.lower() else max(train_values)
        best_test = min(test_values) if 'loss' in metric.lower() else max(test_values)
        
        axes[idx].axhline(y=best_train, color='b', linestyle='--', alpha=0.3)
        axes[idx].axhline(y=best_test, color='r', linestyle='--', alpha=0.3)
        
        # Add value annotations
        for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
            axes[idx].annotate(f'{float(train_val):.4f}', 
                             (epochs[i], train_val),
                             textcoords="offset points",
                             xytext=(0,10),
                             ha='center',
                             fontsize=8)
            axes[idx].annotate(f'{float(test_val):.4f}',
                             (epochs[i], test_val),
                             textcoords="offset points",
                             xytext=(0,-15),
                             ha='center',
                             fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()




