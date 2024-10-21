import cv2
import numpy as np
from typing import List, Tuple
from sklearn.cluster import DBSCAN
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype


def read_image(image_path: str) -> np.ndarray:
    """
    Read an image using OpenCV.

    Args:
        image_path (str): Path to the image file

    Returns:
        np.ndarray: Image array in BGR format
    """
    return cv2.imread(image_path)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image for PeopleNet model.

    Args:
        image (np.ndarray): Input image array in BGR format

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 3, 544, 960)
    """
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

    return image


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
    input_name = "input_1:0"  # Adjust if your model uses a different input name
    inputs = [httpclient.InferInput(input_name, preprocessed_image.shape, datatype="FP32")]
    inputs[0].set_data_from_numpy(preprocessed_image)

    # Run inference
    outputs = [
        httpclient.InferRequestedOutput("output_cov/Sigmoid:0"),  # Adjust if your model uses different output names
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
        confidence_threshold: float = 0.5,
        eps: float = 0.2,
        min_samples: int = 1
) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
    """
    Postprocess the model output to get final detections.

    Args:
        cov (np.ndarray): Coverage array of shape (1, 3, 34, 60)
        bbox (np.ndarray): Bounding box array of shape (1, 12, 34, 60)
        confidence_threshold (float): Confidence threshold for filtering detections
        eps (float): DBSCAN epsilon parameter
        min_samples (int): DBSCAN min_samples parameter

    Returns:
        List[Tuple[str, Tuple[float, float, float, float], float]]:
            List of (class_name, (x, y, w, h), confidence) tuples
    """
    classes = ['Bag', 'Face', 'Person']
    results = []

    for class_idx, class_name in enumerate(classes):
        # Extract class-specific arrays
        class_cov = cov[0, class_idx]
        class_bbox = bbox[0, class_idx * 4:(class_idx + 1) * 4]

        # Filter by confidence
        mask = class_cov > confidence_threshold
        confident_cov = class_cov[mask]
        confident_bbox = class_bbox[:, mask]

        if confident_cov.size == 0:
            continue

        # Prepare data for clustering
        grid_y, grid_x = np.mgrid[0:34, 0:60]
        grid_points = np.column_stack((grid_x[mask], grid_y[mask]))

        # Cluster detections
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(grid_points)
        labels = clustering.labels_

        for cluster_id in range(labels.max() + 1):
            cluster_mask = labels == cluster_id
            cluster_cov = confident_cov[cluster_mask]
            cluster_bbox = confident_bbox[:, cluster_mask]

            # Compute weighted average of bounding box coordinates
            weights = cluster_cov / cluster_cov.sum()
            avg_bbox = (cluster_bbox * weights).sum(axis=1)

            # Convert to (x, y, w, h) format
            x1, y1, x2, y2 = avg_bbox.tolist()
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            # Add to results
            confidence = cluster_cov.max().item()
            results.append((class_name, (x, y, w, h), confidence))

    return results


def process_image(triton_url: str, model_name: str, image_path: str) -> List[
    Tuple[str, Tuple[float, float, float, float], float]]:
    """
    Process an image through the PeopleNet model using Triton Inference Server.

    Args:
        triton_url (str): URL of the Triton Inference Server
        model_name (str): Name of the model on Triton server
        image_path (str): Path to the input image file

    Returns:
        List[Tuple[str, Tuple[float, float, float, float], float]]:
            List of (class_name, (x, y, w, h), confidence) tuples
    """
    # Create Triton client
    triton_client = httpclient.InferenceServerClient(url=triton_url)

    # Read the image
    image = read_image(image_path)

    # Preprocess
    preprocessed_image = preprocess(image)

    # Run inference
    cov, bbox = run_inference(triton_client, preprocessed_image, model_name)

    # Postprocess
    detections = postprocess(cov, bbox)

    return detections


# Usage example
if __name__ == "__main__":
    triton_url = "192.168.0.22:8000"  # Adjust this to your Triton server's address
    model_name = "peoplenet"  # Adjust this to your model's name on Triton
    image_path = "ultralytics/assets/83.jpg"

    results = process_image(triton_url, model_name, image_path)

    # Print results
    for class_name, (x, y, w, h), confidence in results:
        print(f"Class: {class_name}, Bbox: ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}), Confidence: {confidence:.2f}")

    # Optionally, visualize results on the image
    image = cv2.imread(image_path)
    for class_name, (x, y, w, h), confidence in results:
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                    2)

    cv2.imshow("PeopleNet Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()