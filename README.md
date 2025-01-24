# Extract Pseudo labelled Images from Video

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/cathaoiragnew/Pseudo_Label_Video_Frames.git
    cd Pseudo_Label_Video_Frames
    ```

2. Install dependencies:
    - **For Python**:
        ```bash
        pip install -r requirements.txt
        ```

3. Run create_pseudo_data.py to pseudo label the video frames with the passed arguments given below.
    - **For Python**:
        ```bash
        python create_pseudo_data.py [OPTIONS]
        ```
        
    Example running on all frames of example video provided street_video.mp4 for all frames.
    - **For Python**:
        ```bash
        python create_pseudo_data.py --input_video_path 'street_video.mp4' --n_runs 'all' 
        ```

 ### Arguments Table

| Argument                            | Type     | Description                                                                              | Example                             |
|-------------------------------------|----------|------------------------------------------------------------------------------------------|-------------------------------------|
| `--n_runs`                          | Integer or 'all'  | Number of frames to process.                                                            | `--n_runs 10`                      |
| `--info`                            | Boolean   | Debugging information flag (True or False).                                         | `--info True`                      |
| `--image_size`                      | Integer  | Image size as `(width, height)`.                                                        | `--image_size (1280,720)`            |
| `--input_video_path`                | String   | Path to the input video file.                                                           | `--input_video_path ./video.mp4`   |
| `--extracted_frames_dir`            | String   | Directory for storing extracted frames.                                                 | `--extracted_frames_dir ./frames/` |
| `--extracted_frames_dup_keep_dir`   | String   | Directory for duplicate-kept frames.                                                    | `--extracted_frames_dup_keep_dir ./frames_keep/` |
| `--extracted_frames_dup_remove_dir` | String   | Directory for duplicate-removed frames.                                                 | `--extracted_frames_dup_remove_dir ./frames_remove/` |
| `--model_inferences_dir`            | String   | Directory to store model inferences.                                                    | `--model_inferences_dir ./inferences/` |
| `--json_predictions_path`           | String   | Path for JSON predictions.                                                              | `--json_predictions_path ./predictions.json` |
| `--json_predictions_path_filtered`  | String   | Path for filtered JSON predictions.                                                     | `--json_predictions_path_filtered ./filtered_predictions.json` |
| `--heatmaps_dir`                    | String   | Directory to store heatmaps.                                                            | `--heatmaps_dir ./heatmaps/`       |
| `--focus_tol`                       | Float    | Focus tolerance value.                                                                  | `--focus_tol 0.05`                  |
| `--entropy_tol`                     | Float    | Signal-to-noise ratio tolerance value.                                                  | `--entropy_tol 0.975`               |
| `--clip_dup_threshold`              | Float    | Clip embedding cosine similarity threshold for duplication detection.                   | `--clip_dup_threshold 0.9`         |
| `--model_type`                      | String   | Object detection model from Hugging Face.                                               | `--model_type facebook/detr-resnet-50`          |
| `--model_confidence`                | Float    | Model confidence threshold for predictions.                                             | `--model_confidence 0.85`          |
| `--batch_size`                      | Integer  | Batch size for inference.                                                               | `--batch_size 16`                  |
| `--classes_of_interest`             | List  | List of classes of interest.                                                            | `--classes_of_interest ["person", "bus", "bicycle"]` |
| `--min_bb_area`                     | Float  | Minimum bounding box area as a percentage of the image area (in decimal).               | `--min_bb_area 0.01`               |
| `--max_bb_area`                     | Float  | Maximum bounding box area as a percentage of the image area (in decimal).               | `--max_bb_area 0.75`                |
| `--image_name`                      | String   | Naming convention for output files.                                                     | `--image_name frame_{index}`       |


### Example 

<video width="640" height="360" controls>
  <source src="street_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
