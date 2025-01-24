# This pthon script is used to inference with a Hugging Face Object Detection mode to pseudo label the dataset.
# This implementation is geared towards any HF OD model pretrained on COCO.

# Imports 
import cv2
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import json
from skimage.metrics import structural_similarity as ssim
from sentence_transformers import SentenceTransformer, util
import shutil
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import logging

# Set logging level to ERROR to suppress warnings
logging.set_verbosity_error()


# Detection Function
def detect_objects_hf_batch(input_frames_dir, output_json_path, model_name="facebook/detr-resnet-50", threshold=0.9, batch_size=4):
    """
    Run object detection using a Hugging Face pipeline and save results in COCO format.

    Args:
        input_frames_dir (str): Directory containing input image frames.
        output_json_path (str): Path to save COCO-format detection results.
        model_name (str): Name of the Hugging Face object detection model.
        threshold (float): Confidence threshold for detections.
        batch_size (int): Number of images to process in each batch.
    """
    # Load object detection pipeline
    pipe = pipeline("object-detection", model=model_name)

    # Set up for annotations in json 
    annotations = {"images": [], "annotations": [], "categories": []}

    # COCO categories
    categories = [{"id": i, "name": name} for i, name in enumerate([
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
                ])]
    
    # Adding the above list to the categories list
    annotations["categories"].extend(categories)

    # Create a category name to id mapping
    category_name_to_id = {category["name"]: category["id"] for category in categories}


    # Setting up counters
    image_id = 0
    annotation_id = 0

    # Get a list of all images in the directory
    image_files = sorted(os.listdir(input_frames_dir))

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        images = []
        for file_name in batch_files:
            img_path = os.path.join(input_frames_dir, file_name)
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            
        # Run inference
        results = pipe(images)


        # Process each image in the batch
        for idx, result in enumerate(results):
            img_file = batch_files[idx]
            img_path = os.path.join(input_frames_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            
            # Add image metadata
            annotations["images"].append({
                "id": image_id,
                "file_name": img_file,
                "width": image.width,
                "height": image.height
            })


            # Process detections for each image
            for detection in result:
                box = detection["box"]
                label_name = detection["label"]
                score = detection["score"]

                if score < threshold:
                    continue  # Skip low-confidence detections

                # Get bounding box coordinates and dimensions
                x_min = box["xmin"]
                y_min = box["ymin"]
                x_max = box["xmax"]
                y_max = box["ymax"]
                width = x_max - x_min
                height = y_max - y_min

       
                # Map label (class name) to category_id (integer)
                category_id = category_name_to_id.get(label_name, None)
                if category_id is None:
                    continue  # Skip if no category ID found for this label name


                # Add annotation to dictionary
                annotations["annotations"].append({
                    "id": int(annotation_id),
                    "image_id": int(image_id),
                    "category_id": int(category_id),  # Class label ID
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "area": float(width * height),
                    "score": float(score),
                    # just adding this for compeleteness
                    "segmentation": [],
                    "iscrowd": 0 
                })
                annotation_id += 1

            image_id += 1

    # Save results to COCO format
    with open(output_json_path, "w") as f:
        json.dump(annotations, f)
    print(f"Detections saved to {output_json_path}")

# Visualization Function
def visualize_detections(input_frames_dir, output_json_path, output_dir, threshold=0.9):
    """
    Visualize detections on images and save visualized frames to a directory.

    Args:
        input_frames_dir (str): Directory containing input image frames.
        output_json_path (str): Path to COCO-format detection results.
        output_dir (str): Directory to save visualized frames.
        threshold (float): Confidence threshold for visualization.
    """
    
    # Load COCO detections
    with open(output_json_path, "r") as f:
        annotations = json.load(f)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare image dictionary for quick lookup
    categories = {cat["id"]: cat["name"] for cat in annotations["categories"]}

    # Visualize each image
    for annotation in annotations["images"]:
        img_id = annotation["id"]
        img_file = annotation["file_name"]
        img_path = os.path.join(input_frames_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        
        # Find annotations for this image
        img_annotations = [
            anno for anno in annotations["annotations"] if anno["image_id"] == img_id
        ]

        for anno in img_annotations:
            if anno["score"] < threshold:
                continue  # Skip low-confidence detections
            x_min, y_min, width, height = anno["bbox"]
            x_max = x_min + width
            y_max = y_min + height

            # Draw bounding box
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="lime", width=3)

            # Add label and score with class name
            label_id = anno["category_id"]
            label = categories.get(label_id, f"{label_id}")  # Get class label
            score = anno["score"]
            label_text = f"{label}: {score:.2f}"
            draw.text((x_min, y_min - 10), label_text, fill="lime", width = 5)

        # Save visualized image
        output_path = os.path.join(output_dir, f"visualized_{img_file}")
        image.save(output_path)
        #print(f"Saved visualization: {output_path}")



def filter_coco_json(input_json, output_json, allowed_categories, max_area, min_area):
    """
    Filters a COCO JSON file to include only specified classes and remove annotations with area > max_area.
    
    Args:
        input_json (str): Path to the input COCO JSON file.
        output_json (str): Path to save the filtered COCO JSON file.
        allowed_categories (list): List of category names to retain.
        max_area (float): Maximum allowed area for annotations.
        min_area (float): Minimum allowed area for annotations.
    """
    # Load the original COCO JSON
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
    
    # Create a mapping of category names to their IDs
    category_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    allowed_category_ids = {category_name_to_id[name] for name in allowed_categories if name in category_name_to_id}
  
    # Filter annotations based on category and area
    filtered_annotations = [
        ann for ann in coco_data['annotations']
        if ann['category_id'] in allowed_category_ids and (min_area <=  ann['area'] <= max_area)  ]

    # Get the IDs of images that have valid annotations
    valid_image_ids = {ann['image_id'] for ann in filtered_annotations}
    
    # Filter images to include only those with valid annotations
    filtered_images = [
        img for img in coco_data['images'] if img['id'] in valid_image_ids
    ]
    
    # Update the JSON structure
    filtered_coco_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": [
            cat for cat in coco_data['categories'] if cat['id'] in allowed_category_ids
        ]
    }
    
    # Save the filtered COCO JSON
    with open(output_json, 'w') as f:
        json.dump(filtered_coco_data, f, indent=4)
    
    print(f"Filtered COCO JSON saved to {output_json}")


def analyze_coco_annotations_with_sizes_and_percentages(coco_json_path):
    """
    Analyze a COCO annotations JSON file for class counts, object sizes, and percentages.

    Args:
        coco_json_path (str): Path to the COCO JSON file.

    Returns:
        class_counts (dict): Dictionary with counts of objects per class.
        object_sizes (dict): Dictionary with lists of object sizes per class.
        size_categories (dict): Dictionary with counts and percentages of small, medium, and large objects per class.
    """
    # Load COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Initialize dictionaries for results
    class_counts = defaultdict(int)
    object_sizes = defaultdict(list)
    size_categories = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0, "total": 0})

    # Create a mapping of category_id to category_name
    category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Iterate through annotations
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        category_name = category_mapping[category_id]

        # Update class counts
        class_counts[category_name] += 1

        # Calculate object size (COCO metric: area)
        object_size = annotation['area']
        object_sizes[category_name].append(object_size)

        # Categorize object size
        if object_size < 32**2:
            size_categories[category_name]["small"] += 1
        elif 32**2 <= object_size <= 96**2:
            size_categories[category_name]["medium"] += 1
        else:
            size_categories[category_name]["large"] += 1

        # Increment total count
        size_categories[category_name]["total"] += 1

    # Calculate percentages for size categories
    for category, sizes in size_categories.items():
        total = sizes["total"]
        if total > 0:
            sizes["small_pct"] = 100 * sizes["small"] / total
            sizes["medium_pct"] = 100 * sizes["medium"] / total
            sizes["large_pct"] = 100 * sizes["large"] / total

    return class_counts, object_sizes, size_categories


def display_results_with_sizes_and_percentages(class_counts, object_sizes, size_categories):
    """
    Display the results of COCO annotations analysis with size categorization and percentages.

    Args:
        class_counts (dict): Dictionary of class counts.
        object_sizes (dict): Dictionary of object sizes.
        size_categories (dict): Dictionary of size categories per class.
    """
    print("\nClass Counts:")
    total_objects = sum(class_counts.values())
    for class_name, count in class_counts.items():
        percentage = (count / total_objects) * 100
        print(f"  {class_name}: {count} ({percentage:.2f}%)")

    print("\nObject Size Categories:")
    for class_name, size_counts in size_categories.items():
        total = size_counts["total"]
        if total > 0:
            print(f"  {class_name}:")
            print(f"    - Small: {size_counts['small']} ({size_counts['small_pct']:.2f}%)")
            print(f"    - Medium: {size_counts['medium']} ({size_counts['medium_pct']:.2f}%)")
            print(f"    - Large: {size_counts['large']} ({size_counts['large_pct']:.2f}%)")



def generate_per_class_heatmaps(coco_json_path, image_width, image_height, output_dir=None):
    """
    Generate heat maps of bounding box locations for each class from a COCO annotations JSON file.

    Args:
        coco_json_path (str): Path to the COCO annotations JSON file.
        image_width (int): Width of the images in the dataset.
        image_height (int): Height of the images in the dataset.
        output_dir (str, optional): Directory to save the heatmaps. If None, heatmaps are only displayed.
    """
    # Load COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from category ID to name
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Initialize a dictionary to store heatmaps per class
    heatmaps = {cat_id: np.zeros((image_height, image_width)) for cat_id in category_id_to_name.keys()}

    # Accumulate bounding boxes into their respective class heatmaps
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        x, y, width, height = annotation['bbox']

        # Convert bounding box to integer grid coordinates
        x_min = int(x)
        y_min = int(y)
        x_max = int(x + width)
        y_max = int(y + height)

        # Accumulate the heat map values for the specific class
        heatmaps[category_id][y_min:y_max, x_min:x_max] += 1

    # Normalize heatmaps for visualization and display/save each class heatmap
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    for cat_id, heatmap in heatmaps.items():

        # If there is no object of this class, skip

        # Flatten and check
        flat_array = np.array(heatmap, dtype=object).ravel()
        if all(x == 0 for x in flat_array):
        #if heatmap.all() == 0:
            print(f"No objects found for class {category_id_to_name[cat_id]} (ID: {cat_id})")
            continue

        # Normalize the heat map
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        # Plot the heat map
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Density of Objects")
        plt.title(f"Heatmap for Class: {category_id_to_name[cat_id]} (ID: {cat_id})")
        plt.xlabel("Image Width")
        plt.ylabel("Image Height")

        # Save the heat map if output_dir is specified
        if output_dir:
            heatmap_file = os.path.join(output_dir, f"heatmap_class_{cat_id}_{category_id_to_name[cat_id]}.png")
            plt.savefig(heatmap_file, dpi=300)
            print(f"Heatmap saved to {heatmap_file}")

