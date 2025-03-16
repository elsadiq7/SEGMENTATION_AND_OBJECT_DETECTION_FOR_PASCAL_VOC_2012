import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.ops as ops
import torchvision.models.detection as detection
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage

# Define Constants
NUM_CLASSES = 21  # Change based on dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate a colormap for segmentation classes
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], 
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], 
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
], dtype=np.uint8)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

@st.cache_resource
def load_models():
    """Load detection and segmentation models."""
    try:
        # Load Object Detection Model
        detection_model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
        detection_model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            in_features, NUM_CLASSES
        )
        detection_model=torch.load("./models/fasterrcnn_model.pth")
        detection_model.to(DEVICE)
        detection_model.eval()

        # Load Segmentation Model
        segmentation_model = smp.from_pretrained("./models/seg_model")
        segmentation_model.to(DEVICE)
        segmentation_model.eval()

        return detection_model, segmentation_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def validate_mask(mask: np.ndarray) -> None:
    """Validate the input segmentation mask format."""
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Mask must be a numpy array, got {type(mask)}")
    
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    
    if mask.min() < 0 or mask.max() >= NUM_CLASSES:
        raise ValueError(f"Mask values must be between 0 and {NUM_CLASSES-1}, "
                         f"got min={mask.min()}, max={mask.max()}")

def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a segmentation mask with class indices to a color image."""
    validate_mask(mask)
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id in range(NUM_CLASSES):
        class_mask = (mask == class_id)
        color_mask[class_mask] = VOC_COLORMAP[class_id]
    
    return color_mask

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image values to range [0, 255]."""
    if image.max() > 1.0:
        return image.astype(np.uint8)
    return (image * 255).astype(np.uint8)

def apply_nms(predictions, iou_threshold=0.5, score_threshold=0.5):
    """Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes."""
    filtered_preds = []
    
    for pred in predictions:
        boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
        high_scores_idx = scores > score_threshold
        boxes, scores, labels = boxes[high_scores_idx], scores[high_scores_idx], labels[high_scores_idx]

        if len(boxes) > 0:
            keep_idx = ops.nms(boxes, scores, iou_threshold)
            filtered_preds.append({
                "boxes": boxes[keep_idx],
                "scores": scores[keep_idx],
                "labels": labels[keep_idx]
            })
        else:
            filtered_preds.append({"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0)})
    
    return filtered_preds

def detect_objects(model, image):
    """Run object detection on the uploaded image."""
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(input_tensor)

    return apply_nms(predictions)

def segment_image(model, image):
    """Run segmentation model and return correctly decoded mask."""
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask = model(image_tensor).squeeze(0).cpu().numpy()  # (C, H, W)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)  # Convert logits to class indices

    decoded_mask = decode_segmentation_mask(mask)
    return decoded_mask

def overlay_segmentation(image, mask, alpha=0.5):
    """Overlay decoded segmentation mask on the image."""
    image_np = np.array(image)

    # Normalize and blend image
    normalized_img = normalize_image(image_np)
    blended = cv2.addWeighted(normalized_img, 1 - alpha, mask, alpha, 0)

    return Image.fromarray(blended)

def plot_image_with_bboxes(image, predictions):
    """Displays the image with detected objects' bounding boxes."""
    img = (image * 255).clamp(0, 255).to(torch.uint8)

    if predictions and predictions[0]["boxes"].size(0) > 0:
        pred_boxes = predictions[0]["boxes"]
        pred_labels = predictions[0]["labels"]
        pred_scores = predictions[0]["scores"]

        pred_label_names = [f"{VOC_CLASSES[label]} ({score:.2f})" for label, score in zip(pred_labels, pred_scores)]

        # Use a clearer font color and increase font size
        img_with_pred = draw_bounding_boxes(
            img,
            pred_boxes,
            labels=pred_label_names,
            colors=['white'] * len(pred_labels),  # Change color to white
            width=3,
            font_size=30  # Increase font size
        )
        return ToPILImage()(img_with_pred)
    else:
        st.warning("No objects detected.")
        return ToPILImage()(img)
def process_image(image, detection_model, segmentation_model):
    """Processes the uploaded image."""
    try:
        transform = transforms.ToTensor()
        image_tensor = transform(image)

        # Run object detection
        predictions = detect_objects(detection_model, image)

        # Run segmentation
        mask = segment_image(segmentation_model, image)

        # Overlay segmentation on image
        segmented_image = overlay_segmentation(image, mask)

        # Apply bounding boxes on segmented image
        final_image = plot_image_with_bboxes(transforms.ToTensor()(segmented_image), predictions)

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(final_image, caption="Segmented Image with Bounding Boxes", use_column_width=True)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def main():
    st.title("Object Detection and Segmentation")

    detection_model, segmentation_model = load_models()
    if not detection_model or not segmentation_model:
        st.error("Failed to load models.")
        return

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)

        if st.button("Process Image"):
            process_image(image, detection_model, segmentation_model)

        if st.button("Choose Another Image"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()