# This pthon script is used to detect near-duplicates from the created frames.
# This implementation only uses CLIP embeddings to detect near-duplicates.
# But other methods or models could be used to improve the detection.

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
import warnings

warnings.simplefilter('ignore') # In any case, try to avoid warnings as much as possible.

 # Load the CLIP model and its preprocessing pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device used:", device)
model, preprocess = clip.load("ViT-B/32", device=device)

def load_images(image_dir):
    """
    Load images from a directory and preprocess them for CLIP.
    
    Args:
        image_dir (str): Path to the directory containing images.
    
    Returns:
        List of tuples (image_path, processed_image).
    """
    
    images = []
    for image_file in os.listdir(image_dir):

        img_path = os.path.join(image_dir, image_file)
        image = Image.open(img_path).convert("RGB")
        processed_image = preprocess(image).unsqueeze(0).to(device)
        images.append((img_path, processed_image))
            
    return images

def compute_embeddings(images):
    """
    Compute CLIP embeddings for a list of images.
    
    Args:
        images (list): List of tuples (image_path, processed_image).
    
    Returns:
        dict: A dictionary mapping image paths to their embeddings.
    """
    
    embeddings = {}
    for image_path, processed_image in images:
        with torch.no_grad():
            embedding = model.encode_image(processed_image).squeeze(0)
            embedding = embedding / embedding.norm(p=2)  # Normalize embedding - L2 normalization (Euclidean norm normalization)
            embeddings[image_path] = embedding.cpu().numpy()
    return embeddings

def find_near_duplicates(embeddings, threshold=0.95):
    """
    Identify near-duplicate images based on cosine similarity.
    
    Args:
        embeddings (dict): Dictionary of image embeddings.
        threshold (float): Similarity threshold for duplicates (0.95 is high similarity).
    
    Returns:
        list: Pairs of near-duplicate image paths.
    """
    
    image_paths = list(embeddings.keys())
    embedding_matrix = list(embeddings.values())
    similarity_matrix = cosine_similarity(embedding_matrix)
    
    duplicates = []
    for i, img1 in enumerate(image_paths):
        for j, img2 in enumerate(image_paths):

            # Avoid redundant comparison, i -> j is the same as j -> i
            if i < j and similarity_matrix[i, j] >= threshold:
                duplicates.append((img1, img2, similarity_matrix[i, j]))
    return duplicates


def remove_duplicates(duplicates, keep_dir="duplicates_kept", delete_dir="duplicates_removed"):
    """
    Remove near-duplicate images while keeping one representative image.
    
    Parameters:
        duplicates (list): List of tuples (img1, img2, similarity).
        keep_dir (str): Directory to move kept duplicates.
        delete_dir (str): Directory to move deleted duplicates.
    
    """
    
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(delete_dir, exist_ok=True)
    
    # Track images already processed
    processed = set()
    groups = {}


    # Organize duplicates into groups
    for img1, img2, _ in duplicates:
        group = groups.setdefault(img1, {img1})
        group.add(img2)
        groups[img2] = group

    # Keep only one image per group
    for group in set(map(tuple, groups.values())):
        group = list(group)
        representative = group[0]
        
        # Move representative to `keep_dir`
        shutil.copy(representative, os.path.join(keep_dir, os.path.basename(representative)))
        processed.add(representative)

        # Move duplicates to `delete_dir`
        for duplicate in group[1:]:
            if duplicate not in processed:
                shutil.copy(duplicate, os.path.join(delete_dir, os.path.basename(duplicate)))
                processed.add(duplicate)

   
