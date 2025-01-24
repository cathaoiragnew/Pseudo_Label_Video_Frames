# This pthon script is used to extract frames from a video file. 
# We use open CV to read the video file and extract frames from it.
# Following are the steps to extract frames from a video file:
# 
# Data Preprocessing Steps:
# - Converts the video into image frames
# - Remove any bad frames (white noise/blur)
# - Image frames should be resized to 960x544
# - Save the images in jpg format with an appropriate file name

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
from skimage.measure import shannon_entropy


def create_dir(directory):
    """
    Create directory for saving frames from video. Performs a quick check to make sure it doesnt already exist.

    Args:
        directory (str): Path to the directory you would like to create
    """
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# Take in an image and return its focus/blur value 
def focus_measure(image, threshold, threshold_entropy, info=False):
    """
    Takes in an image and calcualte its focus/blur measure using laplacian. In addition to this we calculate the
    Normalized Shannon Entropy, as this may help in filtering out white noise images.

    Args:
        image (array): Image to be used for calculations
        threshold (float): Threshold to be used for determining if image is focus/blurry or not, w.r.t laplacian. 0: Completely blurry image (low variance). 1: Highly focused image (maximum variance). 
        threshold_entropy(float): Threshold to be used for determining if image is white noise or not. The higher the value, more likely to be white noise.
        info (boolean): This is for printing out information of the parameters, for initially finding the values.
        
    Returns:
        focus_measure_norm (float): The normalized focus measure calculated
        entropy_norm (float): The normalized shannon entropy measure calculated
    """
    
    # Normalized Shannon Entropy - find white noise images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Shannon entropy
    entropy = shannon_entropy(gray)
    
    # Normalize entropy to range 0-1
    max_entropy = np.log2(256)  # For an 8-bit grayscale image

    entropy_norm = entropy / max_entropy

    # Calculate focus measure
    focus_measure = cv2.Laplacian(image, cv2.CV_64F).var()

    # Normalize variance
    max_variance_focus = (255 ** 2) / 12  # For 8-bit images

    # Normalize value
    focus_measure_norm = focus_measure/max_variance_focus

    # Print statements to help when selecting thresholds. 
    if info ==True:
        print(f'Focus measure value: {focus_measure_norm:05f} | ' +
              f'Equality check to keep frame >= | ' +
              f'Tolerance {threshold:05f} | ' +
              f'{focus_measure>=threshold}')

        print(f'Shannon Entropy: {entropy_norm:05f} | ' +
              f'Equality check to keep frame <= | ' +
              f'Tolerance {threshold_entropy:05f} | ' +
              f'{entropy_norm<=threshold_entropy}')

    return focus_measure_norm , entropy_norm


# Extract images from video
def image_from_video(video_path, out_dir , focus_tol = 0.10,
                     entropy_tol = 0.9, n_runs='all',
                     info=False, frame_size= (960,544),
                     image_name = 'Image'):
    """
    Extract images from a video while applying quality checks and filtering.

    This function processes a video to extract and save frames as images. 
    It applies checks for frame quality (focus, signal-to-noise ratio). 

    Args:
        video_path (str): Path to the input video file.
        out_dir (str): Directory where the extracted images will be saved.
        focus_tol (float): Minimum acceptable focus measure to keep a frame. Defaults to 50.0. (lower it is, the more images will be accepted)
        entropy_tol (float): Maximum acceptable signal-to-noise ratio to keep a frame. Defaults to 1.0. (lower it is, the more images will be rejected)
        n_runs (int or str): Number of frames to process ('all' for entire video). Defaults to 'all'.
        info (bool): If True, prints debugging information and visualizes frames. Defaults to False.
        frame_size (tuple): Desired size (width, height) for saved images. Defaults to (960, 544).
        image_name (str): Naming convention for frames.
    """

    # Create directory to hold the images
    create_dir(out_dir)

    # Create video capture object from Video footage
    cap = cv2.VideoCapture(video_path)

    # Set up counter for frames
    frame_counter = 0

    # Number of saved frames
    saved_counter = 0

    # Set up a previous frame to check consecutive frames for duplicates
    prev_frame = None

    # Read in all the frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Flag to keep frame - we assume its fine to keep to begin
        keep_frame = True

        # Some print outs to help inform parameters thresholds etc
        if info==True:
            print(f"\n---------- Image {frame_counter} ---------- ")

        # Caculate the focus/blur & Signal to Noise of the image. 
        focus_meas, entropy_measure = focus_measure(frame, threshold=focus_tol, threshold_entropy= entropy_tol, info=info)
        
        # We have a blurry/noisey image no point keeping it
        if focus_meas <= focus_tol or entropy_measure >= entropy_tol :
            keep_frame = False
            
            ## .... Could add other metrics here for comparison, cosine similarity of embeddings etc. ....
        
        # If current frame has passed the tests, we save the image.
        if keep_frame == True:
            
            # Resize the frame
            resized_frame = cv2.resize(frame, frame_size)
    
            # Save the frame
            frame_file = os.path.join(out_dir,
            f"{image_name}_{frame_counter:05d}.jpg")
            cv2.imwrite(frame_file, resized_frame)

            # Update saved_counter
            saved_counter +=1 
        
        frame_counter += 1

        # Since all comparisons are done & image is saved, we can update the prev_frame with current frame for next iteration
        prev_frame = frame

        # Quick bit of logic so we can choose a certain number of frames to run - for finding parameters 
        if type(n_runs) == int:
            if frame_counter >= n_runs:
                break
                
    cap.release()
    print(f"\nExtracted {saved_counter} frames out of a total of {frame_counter} and saved in {out_dir}.")
        