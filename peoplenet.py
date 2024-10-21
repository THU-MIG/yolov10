from scipy import ndimage
from skimage.feature import peak_local_max
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import tritonclient.http as httpclient
import time

def read_image(image_path: str) -> np.ndarray:
    """
    Read an image using OpenCV.

    Args:
        image_path (str): Path to the image file

    Returns:
        np.ndarray: Image array in BGR format
    """
    return cv2.imread(image_path)


def preprocess(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Preprocess the input image for PeopleNet model.

    Args:
        image (np.ndarray): Input image array in BGR format

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: Preprocessed image array and original dimensions
    """
    original_height, original_width = image.shape[:2]

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to 960x544
    image = cv2.resize(image, (960, 544))

    # Normalize the image
    image = image.astype(np.float32) / 255.0

    # Transpose from (H, W, C) to (C, H, W)
    image = image.transpose(2, 0, 1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image, (original_width, original_height)


def run_inference(triton_client: httpclient.InferenceServerClient, preprocessed_image: np.ndarray, model_name: str) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Run inference using Triton Inference Server.

    Args:
        triton_client (httpclient.InferenceServerClient): Triton client object
        preprocessed_image (np.ndarray): Preprocessed image array
        model_name (str): Name of the model on Triton server

    Returns:
        Tuple[np.ndarray, np.ndarray]: Coverage and bounding box output tensors
    """
    # Prepare the input data
    input_name = "input_1:0"
    inputs = [httpclient.InferInput(input_name, list(preprocessed_image.shape), datatype="FP32")]
    inputs[0].set_data_from_numpy(preprocessed_image)

    # Run inference
    outputs = [
        httpclient.InferRequestedOutput("output_cov/Sigmoid:0"),
        httpclient.InferRequestedOutput("output_bbox/BiasAdd:0")
    ]
    response = triton_client.infer(model_name, inputs, outputs=outputs)

    # Get the output data
    cov = response.as_numpy("output_cov/Sigmoid:0")
    bbox = response.as_numpy("output_bbox/BiasAdd:0")

    return cov, bbox


def postprocess(
        cov: np.ndarray,
        bbox: np.ndarray,
        original_dims: Tuple[int, int],
        confidence_thresholds: Dict[str, float] = {
            'Person': 0.3,  # Lower threshold to catch more people
            'Face': 0.5,
            'Bag': 0.8  # Higher threshold to reduce false positives
        },
        min_distance: Dict[str, int] = {
            'Person': 100,  # Larger distance for person detection
            'Face': 30,
            'Bag': 50
        },
        min_size: Dict[str, int] = {
            'Person': 20,  # Larger minimum size for person detection
            'Face': 10,
            'Bag': 15
        },
        box_scale_factor: Dict[str, float] = {
            'Person': 1.3,  # Larger scaling for person boxes
            'Face': 1.1,
            'Bag': 1.0
        }
) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
    """
    Enhanced postprocessing with better person detection and overlap handling.
    """
    classes = ['Person', 'Face', 'Bag']  # Reorder to prioritize person detection
    results = []

    orig_height, orig_width = original_dims[1], original_dims[0]

    for class_name in classes:
        class_idx = ['Bag', 'Face', 'Person'].index(class_name)  # Map to original model output order

        # Get class-specific parameters
        threshold = confidence_thresholds[class_name]
        distance = min_distance[class_name]
        size = min_size[class_name]
        scale = box_scale_factor[class_name]

        # Extract heatmap for current class
        heatmap = cov[0, class_idx]

        # Resize heatmap to original image dimensions
        heatmap = cv2.resize(heatmap, (orig_width, orig_height))

        # Normalize heatmap
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # For person class, apply additional processing
        if class_name == 'Person':
            # Enhance person detection sensitivity
            heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
            heatmap = np.power(heatmap, 0.7)  # Reduce sensitivity to confidence threshold

        # Find local maxima
        coordinates = peak_local_max(
            heatmap,
            min_distance=distance,
            threshold_abs=threshold,
            exclude_border=False
        )

        # Process each peak
        for coord in coordinates:
            y, x = coord

            # Grow region around peak
            binary = heatmap > (heatmap[y, x] * 0.3)  # More lenient region growing
            labeled, _ = ndimage.label(binary)
            region = labeled == labeled[y, x]

            # Find region bounds
            ys, xs = np.where(region)
            if len(ys) > 0 and len(xs) > 0:
                x1, x2 = np.min(xs), np.max(xs)
                y1, y2 = np.min(ys), np.max(ys)

                # Scale the box
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Apply class-specific scaling
                width *= scale
                height *= scale

                # Ensure box stays within image bounds
                width = min(width, orig_width - center_x, center_x)
                height = min(height, orig_height - center_y, center_y)

                # Filter small detections
                if width >= size and height >= size:
                    # Get confidence from peak value
                    confidence = heatmap[y, x]

                    # Add detection
                    results.append((
                        class_name,
                        (center_x, center_y, width, height),
                        confidence
                    ))

    # Resolve overlapping detections
    results = resolve_overlapping_detections(results)

    return results


def resolve_overlapping_detections(
        detections: List[Tuple[str, Tuple[float, float, float, float], float]],
        iou_threshold: float = 0.3
) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
    """
    Resolve overlapping detections with class priority rules.
    """
    if not detections:
        return []

    # Sort by class priority (Person > Face > Bag) and confidence
    class_priority = {'Person': 0, 'Face': 1, 'Bag': 2}
    detections = sorted(detections,
                        key=lambda x: (class_priority[x[0]], -x[2]))

    final_detections = []

    while detections:
        current = detections.pop(0)
        current_box = current[1]  # (x, y, w, h)

        # Check overlap with existing final detections
        overlapping = False
        for existing in final_detections:
            existing_box = existing[1]
            if calculate_iou(current_box, existing_box) > iou_threshold:
                overlapping = True
                break

        if not overlapping:
            final_detections.append(current)

    return final_detections


def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate IoU between two boxes in center format (x, y, w, h).
    """
    # Convert to corner format
    x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate areas
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    # Calculate IoU
    return intersection / (area1 + area2 - intersection)


def apply_class_rules(
        detections: List[Tuple[str, Tuple[float, float, float, float], float]],
        image_dims: Tuple[int, int]
) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
    """
    Apply class-specific rules to improve detection accuracy.
    """
    filtered_detections = []

    # Group detections by location for conflict resolution
    location_groups = {}
    for detection in detections:
        class_name, (x, y, w, h), conf = detection
        key = f"{int(x / 50)},{int(y / 50)}"  # Group by grid cells
        if key not in location_groups:
            location_groups[key] = []
        location_groups[key].append(detection)

    # Process each location group
    for group in location_groups.values():
        if len(group) > 1:
            # If multiple detections in same area, prefer Person/Face over Bag
            person_detections = [d for d in group if d[0] in ['Person', 'Face']]
            if person_detections:
                filtered_detections.extend(person_detections)
                continue

        filtered_detections.extend(group)

    return filtered_detections


def merge_overlapping_detections(
        detections: List[Tuple[str, Tuple[float, float, float, float], float]],
        iou_threshold: float = 0.5
) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
    """
    Merge overlapping detections using IoU.
    """
    if not detections:
        return []

    # Convert to corner format for IoU calculation
    boxes = []
    for class_name, (x, y, w, h), conf in detections:
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append((class_name, (x1, y1, x2, y2), conf))

    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x[2], reverse=True)

    merged = []
    while boxes:
        current = boxes.pop(0)
        boxes_to_merge = [current]

        i = 0
        while i < len(boxes):
            if (boxes[i][0] == current[0] and  # Same class
                    box_iou(current[1], boxes[i][1]) > iou_threshold):
                boxes_to_merge.append(boxes.pop(i))
            else:
                i += 1

        # Merge boxes
        merged_box = merge_box_list(boxes_to_merge)
        merged.append(merged_box)

    # Convert back to center format
    final_results = []
    for class_name, (x1, y1, x2, y2), conf in merged:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        final_results.append((class_name, (center_x, center_y, width, height), conf))

    return final_results


def box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two boxes in corner format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / float(area1 + area2 - intersection)


def merge_box_list(boxes: List[Tuple[str, Tuple[float, float, float, float], float]]) -> Tuple[
    str, Tuple[float, float, float, float], float]:
    """Merge a list of boxes into a single box."""
    class_name = boxes[0][0]  # Use class of highest confidence box
    x1 = min(box[1][0] for box in boxes)
    y1 = min(box[1][1] for box in boxes)
    x2 = max(box[1][2] for box in boxes)
    y2 = max(box[1][3] for box in boxes)
    conf = max(box[2] for box in boxes)  # Take highest confidence

    return (class_name, (x1, y1, x2, y2), conf)


def visualize_heatmap(image: np.ndarray, cov: np.ndarray, class_idx: int = 0) -> np.ndarray:
    """
    Create a heatmap visualization overlay.

    Args:
        image (np.ndarray): Original image
        cov (np.ndarray): Coverage array
        class_idx (int): Class index to visualize

    Returns:
        np.ndarray: Image with heatmap overlay
    """
    # Extract and resize heatmap
    heatmap = cov[0, class_idx]
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Convert to color heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    return cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)


class TritonClient:
    def __init__(self, url: str, model_name: str):
        """
        Initialize Triton client with connection monitoring.

        Args:
            url (str): Triton server URL
            model_name (str): Name of the model to use
        """
        self.url = url
        self.model_name = model_name
        self.client = None
        self.connect()

    def connect(self):
        """Establish connection to Triton server with error handling."""
        try:
            self.client = httpclient.InferenceServerClient(url=self.url)
            # Test connection
            self.client.is_server_ready()
            print(f"Successfully connected to Triton server at {self.url}")
        except Exception as e:
            print(f"Failed to connect to Triton server: {str(e)}")
            raise

    def run_inference(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run inference with timing and error handling.

        Args:
            input_data (np.ndarray): Preprocessed input image

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: bbox output, coverage output, and inference time
        """
        try:
            # Prepare input tensor
            input_tensor = httpclient.InferInput("input_1:0", input_data.shape, "FP32")
            input_tensor.set_data_from_numpy(input_data)

            # Prepare output tensors
            outputs = [
                httpclient.InferRequestedOutput("output_bbox/BiasAdd:0"),
                httpclient.InferRequestedOutput("output_cov/Sigmoid:0")
            ]

            # Run inference with timing
            start_time = time.time()
            response = self.client.infer(self.model_name, [input_tensor], outputs=outputs)
            inference_time = time.time() - start_time

            # Get outputs
            output_bbox = response.as_numpy("output_bbox/BiasAdd:0")
            output_cov = response.as_numpy("output_cov/Sigmoid:0")

            return output_bbox, output_cov, inference_time

        except Exception as e:
            print(f"Inference failed: {str(e)}")
            # Try to reconnect once
            try:
                self.connect()
                print("Retrying inference after reconnection...")
                return self.run_inference(input_data)
            except:
                raise


def draw_detections(
        image: np.ndarray,
        detections: List[Tuple[str, Tuple[float, float, float, float], float]],
        class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw detection boxes and labels with enhanced visualization.

    Args:
        image (np.ndarray): Original image
        detections (List[Tuple[str, Tuple[float, float, float, float], float]]):
            List of (class_name, (x, y, w, h), confidence) tuples
        class_colors (Optional[Dict[str, Tuple[int, int, int]]]):
            Custom color mapping for classes

    Returns:
        np.ndarray: Image with drawn detections
    """
    # Make a copy to avoid modifying original image
    image_with_boxes = image.copy()

    # Default colors if not provided
    if class_colors is None:
        class_colors = {
            'Person': (0, 255, 0),  # Green
            'Face': (255, 0, 0),  # Blue
            'Bag': (0, 0, 255),  # Red
        }

    # Font settings
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1
    box_thickness = 2

    # Sort detections by confidence for better visualization
    detections_sorted = sorted(detections, key=lambda x: x[2], reverse=True)

    for class_name, (x, y, w, h), confidence in detections_sorted:
        # Get color for class
        color = class_colors.get(class_name, (255, 255, 255))

        # Calculate box coordinates
        x1 = max(0, int(x - w / 2))
        y1 = max(0, int(y - h / 2))
        x2 = min(image.shape[1], int(x + w / 2))
        y2 = min(image.shape[0], int(y + h / 2))

        # Draw box with opacity based on confidence
        alpha = max(0.3, min(0.9, confidence))  # Map confidence to opacity
        overlay = image_with_boxes.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)
        cv2.addWeighted(overlay, alpha, image_with_boxes, 1 - alpha, 0, image_with_boxes)

        # Prepare label with class and confidence
        label = f"{class_name} {confidence:.2f}"

        # Calculate label background size
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Ensure label stays within image bounds
        label_x = max(0, x1)
        label_y = max(label_h + baseline + 5, y1)

        # Draw label background with semi-transparency
        overlay = image_with_boxes.copy()
        cv2.rectangle(
            overlay,
            (label_x, label_y - label_h - baseline - 5),
            (min(image.shape[1], label_x + label_w + 5), label_y),
            color,
            -1
        )
        cv2.addWeighted(overlay, 0.7, image_with_boxes, 0.3, 0, image_with_boxes)

        # Draw label text
        cv2.putText(
            image_with_boxes,
            label,
            (label_x + 2, label_y - baseline - 3),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

        # Draw confidence bar
        bar_width = int(50 * confidence)
        bar_height = 3
        cv2.rectangle(
            image_with_boxes,
            (label_x, label_y + 2),
            (label_x + bar_width, label_y + bar_height + 2),
            color,
            -1
        )

    return image_with_boxes


# Example usage
def process_image(image_path: str, triton_url: str, model_name: str) -> Tuple[np.ndarray, float]:
    """
    Process an image through the detection pipeline.

    Args:
        image_path (str): Path to input image
        triton_url (str): Triton server URL
        model_name (str): Model name on Triton server

    Returns:
        Tuple[np.ndarray, float]: Annotated image and inference time
    """
    # Initialize Triton client
    client = TritonClient(triton_url, model_name)

    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Store original dimensions
    original_dims = image.shape[:2]

    # Preprocess image
    preprocessed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed = cv2.resize(preprocessed, (960, 544))
    preprocessed = preprocessed.astype(np.float32) / 255.0
    preprocessed = preprocessed.transpose(2, 0, 1)
    preprocessed = np.expand_dims(preprocessed, axis=0)

    # Run inference
    output_bbox, output_cov, inference_time = client.run_inference(preprocessed)

    # Post-process detections
    detections = postprocess(
        output_cov,
        output_bbox,
        (image.shape[1], image.shape[0])
    )

    # Draw detections
    result_image = draw_detections(image, detections)

    return result_image, inference_time


if __name__ == "__main__":
    # Example configuration
    triton_url = "192.168.0.22:8000"
    model_name = "peoplenet"
    image_path = "ultralytics/assets/83.jpg"

    try:
        # Process image
        result_image, inference_time = process_image(image_path, triton_url, model_name)

        # Save result
        cv2.imwrite("output_detection.jpg", result_image)
        print(f"Inference time: {inference_time:.3f} seconds")
        print("Detection results saved to output_detection.jpg")

    except Exception as e:
        print(f"Error processing image: {str(e)}")