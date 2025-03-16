"""
Object Detection and Segmentation Training Module
==============================================

This module provides comprehensive training utilities for both object detection 
and segmentation models, including training loops, evaluation metrics, and logging.

Key Components:
--------------
- Training loops for object detection and segmentation
- Metric computation and evaluation
- Model checkpointing
- Progress tracking and logging
- Learning rate scheduling

Dependencies:
------------
- torch
- torchvision
- numpy
- logging
- tqdm
- segmentation_models_pytorch as smp

Author: [Abdulrahman Elsadiq]
Version: 1.0
"""




import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from Utils.load_data import VOCDataset_OD, voc_collate_fn
from torchvision.ops import box_iou
from sklearn.metrics import f1_score
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
# Configure logging
import os

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Training function
def train(model, train_loader, test_loader, optimizer, scheduler, device, num_epochs=10):
    """
    Main training loop for object detection model with logging capabilities.
    
    Args:
        model (torch.nn.Module): The detection model to train
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler: Learning rate scheduler
        device (str): Device to run training on ('cuda' or 'cpu')
        num_epochs (int): Number of training epochs (default: 10)
    
    Features:
        - Batch-wise loss logging
        - Periodic evaluation (every 5 epochs)
        - Learning rate scheduling
        - Detailed logging to file
    """
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        model.train()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass & loss calculation
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            logging.info(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")
        logging.info(f"Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_iou, label_acc, f1_val = evaluate(model, test_loader, device)
            print(f"[Test] Epoch {epoch} - IoU: {avg_iou:.4f}, Acc: {label_acc:.4f}, F1: {f1_val:.4f}")
            logging.info(f"[Test] Epoch {epoch} - IoU: {avg_iou:.4f}, Acc: {label_acc:.4f}, F1: {f1_val:.4f}")

    print("Training completed!")
    logging.info("Training completed!")



def compute_metrics(predictions, targets, iou_threshold=0.5):
    """
    Computes evaluation metrics for object detection predictions.
    
    Args:
        predictions (List[Dict]): Model predictions containing boxes, labels, and scores
        targets (List[Dict]): Ground truth annotations
        iou_threshold (float): IoU threshold for matching predictions to ground truth
    
    Returns:
        tuple: (average_iou, accuracy, f1_score)
            - average_iou (float): Mean IoU for matched boxes
            - accuracy (float): Classification accuracy for matched boxes
            - f1_score (float): Weighted F1-score for matched boxes
    
    Features:
        - Handles empty predictions/targets
        - Box-level IoU computation
        - Label matching and accuracy calculation
        - Weighted F1-score computation
    """
    total_iou = 0.0
    total_valid_matches = 0
    pred_labels_list = []
    target_labels_list = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"].detach().cpu()
        pred_labels = pred["labels"].detach().cpu().numpy()
        target_boxes = target["boxes"].detach().cpu()
        target_labels = target["labels"].detach().cpu().numpy()

        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            continue  # Skip empty cases

        # Compute IoU
        iou_matrix = box_iou(pred_boxes, target_boxes)
        matched_iou, matched_indices = iou_matrix.max(dim=1)

        # Filter valid matches
        valid_matches = matched_iou >= iou_threshold
        if valid_matches.sum().item() > 0:
            total_iou += matched_iou[valid_matches].sum().item()
            total_valid_matches += valid_matches.sum().item()

        # Store labels for accuracy & F1
        matched_pred_labels = pred_labels[valid_matches.numpy()]
        matched_target_labels = target_labels[matched_indices[valid_matches].numpy()]
        pred_labels_list.extend(matched_pred_labels)
        target_labels_list.extend(matched_target_labels)

    # Compute final IoU
    avg_iou = total_iou / total_valid_matches if total_valid_matches > 0 else 0

    # Compute Accuracy and F1-score
    if not target_labels_list:
        return avg_iou, 0, 0

    accuracy = sum(p == t for p, t in zip(pred_labels_list, target_labels_list)) / len(target_labels_list)
    f1 = f1_score(target_labels_list, pred_labels_list, average='weighted')

    return avg_iou, accuracy, f1

# Evaluation function
def evaluate(model, data_loader, device):
    """
    Evaluates object detection model on validation data.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (DataLoader): Validation data loader
        device (str): Device for evaluation
    
    Returns:
        tuple: (avg_iou, avg_acc, avg_f1)
            - avg_iou (float): Average IoU across all batches
            - avg_acc (float): Average accuracy
            - avg_f1 (float): Average F1-score
    """    
    model.eval()
    total_iou, total_acc, total_f1, num_batches = 0, 0, 0, 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            # Compute IoU, Accuracy, and F1-score together
            batch_iou, batch_acc, batch_f1 = compute_metrics(predictions, targets)

            # Accumulate values
            total_iou += batch_iou
            total_acc += batch_acc
            total_f1 += batch_f1
            num_batches += 1

    # Avoid division by zero
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    avg_f1 = total_f1 / num_batches if num_batches > 0 else 0

    model.train()
    return avg_iou, avg_acc, avg_f1






#######################################




def compute_metrics_seg(outputs, targets, mode='multilabel', threshold=0.5):
    """
    Computes metrics for segmentation tasks.
    
    Args:
        outputs (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth masks
        mode (str): Evaluation mode ('multilabel' or 'binary')
        threshold (float): Classification threshold
    
    Returns:
        dict: Dictionary containing various metrics:
            - iou_score: Intersection over Union
            - f1_score: F1 score
            - f2_score: F2 score
            - accuracy: Pixel accuracy
            - recall: Recall score
    """
    # Compute statistics
    tp, fp, fn, tn = smp.metrics.get_stats(
        outputs, 
        targets.long(), 
        mode=mode, 
        threshold=threshold
    )

    # Compute metrics
    metrics = {
        'iou_score': smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise"),
        'f1_score': smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise"),
        'f2_score': smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro-imagewise"),
        'accuracy': smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise"),
        'recall': smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    }
    
    return metrics

def test_seg(model, test_loader, criterion, device):
    """
    Validation function for segmentation model.
    
    Args:
        model (torch.nn.Module): Segmentation model
        test_loader (DataLoader): Validation data loader
        criterion: Loss function
        device (str): Device for computation
    
    Returns:
        tuple: (avg_loss, avg_metrics)
            - avg_loss (float): Average loss on validation set
            - avg_metrics (dict): Dictionary of averaged evaluation metrics
    """
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    metrics_sum = {'iou_score': 0, 'f1_score': 0, 'f2_score': 0, 'accuracy': 0, 'recall': 0}
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Compute metrics
            batch_metrics = compute_metrics_seg(outputs, targets)
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
            
            running_loss += loss.detach()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute averages
    avg_loss = running_loss.item() / len(test_loader)
    avg_metrics = {k: v/len(test_loader) for k, v in metrics_sum.items()}
    
    return avg_loss, avg_metrics

def train_seg(model, train_loader, criterion, optimizer, scheduler, device, epoch, num_epochs):
    """
    Training function for one epoch of segmentation training.
    
    Args:
        model (torch.nn.Module): Segmentation model
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device (str): Device for training
        epoch (int): Current epoch number
        num_epochs (int): Total number of epochs
    
    Returns:
        tuple: (avg_loss, avg_metrics)
            - avg_loss (float): Average training loss
            - avg_metrics (dict): Dictionary of averaged training metrics
    
    Features:
        - Progress bar with live updates
        - Batch-wise metric computation
        - Learning rate scheduling
    """

    model.train()
    running_loss = torch.tensor(0.0, device=device)
    metrics_sum = {'iou_score': 0, 'f1_score': 0, 'f2_score': 0, 'accuracy': 0, 'recall': 0}
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            batch_metrics = compute_metrics_seg(outputs, targets)
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
        
        running_loss += loss.detach()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
   
    
    # Compute averages
    avg_loss = running_loss.item() / len(train_loader)
    avg_metrics = {k: v/len(train_loader) for k, v in metrics_sum.items()}
    
    return avg_loss, avg_metrics



def train_model_seg(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                   device, num_epochs, save_path='models/best_model_seg.pth'):
    """
    Complete training pipeline for segmentation model.
    
    Args:
        model (torch.nn.Module): Segmentation model
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device (str): Device for training
        num_epochs (int): Number of training epochs
        save_path (str): Path to save best model
    
    Returns:
        tuple: (train_history, test_history)
            - train_history (list): Training metrics for each epoch
            - test_history (list): Validation metrics for each epoch
    
    Features:
        - Best model saving based on IoU
        - Comprehensive metric tracking
        - Learning rate scheduling
        - Detailed progress printing
    """

    best_iou = float('0')
    train_history = []
    test_history = []
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_metrics = train_seg(
            model, train_loader, criterion, optimizer, 
            scheduler, device, epoch, num_epochs
        )
        
        # Testing phase
        test_loss, test_metrics = test_seg(
            model, test_loader, criterion, device
        )
        test_iou=test_metrics["iou_score"]
        scheduler.step(test_iou)
        
        # Save results
        train_history.append({'loss': train_loss, **train_metrics})
        test_history.append({'loss': test_loss, **test_metrics})
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("\nTraining Metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Save best model
        if test_iou > best_iou:
            best_iou = test_iou
            model.save_pretrained('./models/seg_model', metrics={'iou': best_iou.item() if isinstance(best_iou, torch.Tensor) else best_iou}, dataset='pascal_voc')

            print(f"\nSaved best model with loss: {best_iou:.4f}")
        
        print("-" * 60)
    
    return train_history, test_history











