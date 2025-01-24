# https://www.pexels.com/video/bustling-city-street-with-double-decker-tram-30295427/ - link to video

# Import python files
import duplicate_detection
import extract_frames
import pseudo_label 

import os
import matplotlib
import argparse
import warnings

warnings.simplefilter('ignore') # In any case, try to avoid warnings as much as possible.

# For display purposes    
matplotlib.use('Agg')


###### Default Parameters ######
# Number of runs, ie frames to process - this is mainly for debugging. Enter int for specific number else use 'all'
n_runs = 'all'

# Boolean flag for whether to provide info for debugging or just for visuals/info. 
info = False

# Frame naming convetion, this is quite hard coded currently, but this is just an example.
company   = "UL" 
location  = "Limerick"
room      = "Warehouse-3"
date      = "11-01-24"
camera_no = "1"

image_name = f"{company}_{location}_{room}_{date}_camera_{camera_no}_frame"

# Image dimension to be created
image_size = (960,544)

# Area of this image - for filtering bounding boxes
image_area = float(image_size[0]*image_size[1])

# Input path to video to process
input_video_path = "street_video.mp4"

# Directory to save frames from extraction process
extracted_frames_dir = f"output_frames_{n_runs}/"

# Directory for offline duplication detection - images to keep 
extracted_frames_dup_keep_dir = f"duplicates_kept_{n_runs}"

# Directory for offline duplication detection - images to deletekeep 
extracted_frames_dup_remove_dir = f"duplicates_remove_{n_runs}"

# Directory to keep model predictions
model_inferences_dir = f"visualized_frames_{n_runs}/"

# Json path for predictions
json_predictions_path = f"detections_hf_{n_runs}.json"

# Filtered 
json_predictions_path_filtered = f"detections_hf_{n_runs}_filtered.json"

# Heatmaps for summary stats 
heatmaps_dir = f"heatmaps_summary_{n_runs}"

# Focus measure tolerance, values will depend on the given environment. But a higher value in focus measure means a better quality image.
# Lowering this tolerance will allow more images into the dataset.
focus_tol = 0.005

# Shannon Entropy measure. A higher value means the image is more likely to be white noise. 
# Upping this tolerance will allow more images into the dataset.
entropy_tol = 0.975

# Threshold for duplicate detection method (this time using CLIP & cosine simalirty)
# Similarity Score must be less than this threshold for the image to remain in dataset. 
clip_dup_threshold = 0.9975

# The code is set up to be model agnostic, given its at least a hugging face Object detection model.
model_type = "facebook/detr-resnet-50" 

# Threshold for confidence score to keep a predicted object
model_confidence = 0.9

# Batch size to use
batch_size = 4

# COCO Classes of interest to keep predictions for - when dont really care about giraffes for our use case
classes_of_interest = ["person", "bus", "bicycle", "car", "motorcycle", "truck"] 

# Filtering out large and small bounding boxes. Expressed as decimal of the percentage of image.
# Ie 0.4 means 70% of the image area
min_bb_area_per = 0.0
max_bb_area_per = 70.0

min_bb_area = (min_bb_area_per/100) * image_area 
max_bb_area = (max_bb_area_per/100) * image_area 


def main(n_runs=n_runs,
             info=info,
             image_size=image_size,
             input_video_path=input_video_path,
             extracted_frames_dir=extracted_frames_dir,
             extracted_frames_dup_keep_dir=extracted_frames_dup_keep_dir,
             extracted_frames_dup_remove_dir=extracted_frames_dup_remove_dir,
             model_inferences_dir=model_inferences_dir,
             json_predictions_path=json_predictions_path,
             json_predictions_path_filtered=json_predictions_path_filtered,
             heatmaps_dir=heatmaps_dir, 
             focus_tol=focus_tol, 
             entropy_tol=entropy_tol, 
             clip_dup_threshold=clip_dup_threshold, 
             model_type=model_type, 
             model_confidence=model_confidence, 
             batch_size=batch_size, 
             classes_of_interest=classes_of_interest,
             min_bb_area=min_bb_area,
             max_bb_area=max_bb_area,
             image_name = image_name):
  
    

    # Step 1. - Get frames from video
    extract_frames.image_from_video(input_video_path, extracted_frames_dir, focus_tol, entropy_tol, n_runs,
                                    info, image_size, image_name)


    # Check if we didnt extract any frames
    if os.path.exists(extracted_frames_dir) and len(os.listdir(extracted_frames_dir)) == 0:
        return print("No frames extracted. Try adjusting the focus_tol and entropy_tol parameters.")


    # Step 2. -Perform duplicate detection

    # Removing the duplicates with CLIP - The search space is (number of images) Choose 2 - (N C 2)
    # This could be improved by clustering the embeddings and then comparing within clusters, but I will leave this for now.
    images = duplicate_detection.load_images(extracted_frames_dir)
    embeddings = duplicate_detection.compute_embeddings(images)
    duplicates = duplicate_detection.find_near_duplicates(embeddings, threshold= clip_dup_threshold)


    if duplicates:
        print(f"Found {len(duplicates)} pairs of near-duplicates.")
        duplicate_detection.remove_duplicates(duplicates, keep_dir=extracted_frames_dup_keep_dir , delete_dir=extracted_frames_dup_remove_dir )
    else:
        print("No near-duplicates found.")

    # Lastly remove these from original folder 
    # Directory where duplicates were moved
    delete_dir = extracted_frames_dup_remove_dir

    # Original folder containing all images
    original_folder = extracted_frames_dir

    # List all files in the duplicates_removed directory
    for file in os.listdir(delete_dir):
        # Build the file path in the original folder
        original_path = os.path.join(original_folder, file)
        
        # Remove the file from the original folder if it exists
        if os.path.exists(original_path):
            os.remove(original_path)
            #print(f"Removed {original_path} from the original folder.")
    print("Removed duplicate files")    


    # Step 3. - Perform model predictions
    # Setting up to run model on images
    input_frames_dir = extracted_frames_dir
    output_json_path = json_predictions_path
    output_dir = model_inferences_dir 

    # Run detection 
    pseudo_label.detect_objects_hf_batch(input_frames_dir, output_json_path, model_name= model_type,  threshold=model_confidence, batch_size=batch_size)

    # Run visualization
    pseudo_label.visualize_detections(input_frames_dir, output_json_path, output_dir, threshold=model_confidence) 

    # Define JSON paths
    input_json = json_predictions_path 
    output_json = json_predictions_path_filtered


    # Run the filtering function
    pseudo_label.filter_coco_json(input_json, output_json, classes_of_interest, max_bb_area, min_bb_area)

    # Input json file
    coco_json_path = json_predictions_path_filtered

    # Calculate the summary stats and display results
    class_counts, object_sizes, size_categories = pseudo_label.analyze_coco_annotations_with_sizes_and_percentages(coco_json_path)
    pseudo_label.display_results_with_sizes_and_percentages(class_counts, object_sizes, size_categories)

    # Image dims
    image_width, image_height  = image_size

    # Save images to dir
    output_dir = heatmaps_dir

    # Generate the heatmaps
    pseudo_label.generate_per_class_heatmaps(coco_json_path, image_width, image_height, output_dir)


def int_or_str(value):
    try:
        return int(value)  # Try to convert to integer
    except ValueError:
        return value  # If it fails, return as a string


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the main function with specified or default parameters.")

     # Add arguments
    parser.add_argument("--n_runs", type=int_or_str, default=n_runs, help="Number of frames to runs")
    parser.add_argument("--info", type=str, default=info, help="Info string for debugging/information - True or False")
    parser.add_argument("--image_size", type=int, nargs=2, default=image_size, help="Image size as (width, height)")
    parser.add_argument("--input_video_path", type=str, default=input_video_path, help="Path to input video")
    parser.add_argument("--extracted_frames_dir", type=str, default=extracted_frames_dir, help="Directory for extracted frames")
    parser.add_argument("--extracted_frames_dup_keep_dir", type=str, default=extracted_frames_dup_keep_dir, help="Directory for duplicate-kept frames")
    parser.add_argument("--extracted_frames_dup_remove_dir", type=str, default=extracted_frames_dup_remove_dir, help="Directory for duplicate-removed frames")
    parser.add_argument("--model_inferences_dir", type=str, default=model_inferences_dir, help="Directory for model inferences")
    parser.add_argument("--json_predictions_path", type=str, default=json_predictions_path, help="Path for JSON predictions")
    parser.add_argument("--json_predictions_path_filtered", type=str, default=json_predictions_path_filtered, help="Path for filtered JSON predictions")
    parser.add_argument("--heatmaps_dir", type=str, default=heatmaps_dir, help="Directory for heatmaps")
    parser.add_argument("--focus_tol", type=float, default=focus_tol, help="Focus tolerance value")
    parser.add_argument("--entropy_tol", type=float, default=entropy_tol, help="Signal to Noise Ratio tolerance value")
    parser.add_argument("--clip_dup_threshold", type=float, default=clip_dup_threshold, help="Clip embedding cosine similarity threshold for duplication detection")
    parser.add_argument("--model_type", type=str, default=model_type, help="Object detection model from Hugging Face")
    parser.add_argument("--model_confidence", type=float, default=model_confidence, help="Model confidence threshold for predictions")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size for inference")
    parser.add_argument("--classes_of_interest", type=str, nargs="*", default=classes_of_interest, help="List of classes of interest")
    parser.add_argument("--min_bb_area", type=float, default=min_bb_area, help="Minimum bounding box area expressed as percentage of image area in decimal")
    parser.add_argument("--max_bb_area", type=float, default=max_bb_area, help="Maximum bounding box area expressed as percentage of image area in decimal")
    parser.add_argument("--image_name", type=str, default=image_name, help="Naming convention for files")

    # Parse arguments
    args = parser.parse_args()

     # Call main with parsed arguments
    main(n_runs=args.n_runs,
         info=args.info,
         image_size=args.image_size,
         input_video_path=args.input_video_path,
         extracted_frames_dir=args.extracted_frames_dir,
         extracted_frames_dup_keep_dir=args.extracted_frames_dup_keep_dir,
         extracted_frames_dup_remove_dir=args.extracted_frames_dup_remove_dir,
         model_inferences_dir=args.model_inferences_dir,
         json_predictions_path=args.json_predictions_path,
         json_predictions_path_filtered=args.json_predictions_path_filtered,
         heatmaps_dir=args.heatmaps_dir, 
         focus_tol=args.focus_tol, 
         entropy_tol=args.entropy_tol, 
         clip_dup_threshold=args.clip_dup_threshold, 
         model_type=args.model_type, 
         model_confidence=args.model_confidence, 
         batch_size=args.batch_size, 
         classes_of_interest=args.classes_of_interest,
         min_bb_area=args.min_bb_area, 
         max_bb_area=args.max_bb_area,
         image_name=args.image_name)
