# File: split_dataset.py (Fixed Version)
import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(images_dir, ann_file, output_dir, train_ratio=0.8):
    """
    Splits images and annotations into train/val sets.
    Uses original images from old train folder, updates paths in new JSONs.
    """
    
    # Load original annotations
    with open(ann_file) as f:
        data = json.load(f)
    
    # Verify JSON structure
    required_keys = {'categories', 'type', 'images', 'annotations'}
    if not required_keys.issubset(data.keys()):
        raise ValueError(f"JSON must contain these keys: {required_keys}")
    
    # Create image ID to filename mapping
    images_id_map = {img['id']: img['file_name'] for img in data['images']}
    
    # Get all image IDs and shuffle
    all_ids = list(images_id_map.keys())
    random.shuffle(all_ids)
    
    # Split IDs
    split_idx = int(len(all_ids) * train_ratio)
    train_ids = set(all_ids[:split_idx])
    val_ids = set(all_ids[split_idx:])
    
    # Create directories
    (train_dir := Path(output_dir)/'train').mkdir(parents=True, exist_ok=True)
    (val_dir := Path(output_dir)/'val').mkdir(parents=True, exist_ok=True)
    
    # Split images and update paths
    split_map = {'train': [], 'val': []}
    for img_id in tqdm(all_ids, desc='Splitting images'):
        src = Path(images_dir)/images_id_map[img_id]
        
        if img_id in train_ids:
            dest = train_dir/images_id_map[img_id]
            new_path = f"train/{images_id_map[img_id]}"  # Update path for JSON
        else:
            dest = val_dir/images_id_map[img_id]
            new_path = f"val/{images_id_map[img_id]}"
            
        shutil.copy2(src, dest)
        split_map['train' if img_id in train_ids else 'val'].append(img_id)
        
        # Update image path in original data
        for img in data['images']:
            if img['id'] == img_id:
                img['file_name'] = new_path
    
    # Split annotations (FIXED KEY NAME)
    train_ann = [ann for ann in data['annotations'] if ann['image_id'] in train_ids]
    val_ann = [ann for ann in data['annotations'] if ann['image_id'] in val_ids]
    
    # Split types (if they exist)
    train_types = [t for t in data['type'] if t.get('image_id') in train_ids]
    val_types = [t for t in data['type'] if t.get('image_id') in val_ids]
    
    # Build new JSON files
    def build_split_json(ids, anns, types, split_name):
        return {
            "images": [img for img in data['images'] if img['id'] in ids],
            "categories": data['categories'],
            "annotations": anns,
            "type": types
        }
    
    # Save splits
    with open(Path(output_dir)/'train.json', 'w') as f:
        json.dump(build_split_json(train_ids, train_ann, train_types, 'train'), f)
    
    with open(Path(output_dir)/'val.json', 'w') as f:
        json.dump(build_split_json(val_ids, val_ann, val_types, 'val'), f)
    
    print(f"✅ Split complete\n"
          f"Train: {len(train_ids)} images, {len(train_ann)} annotations\n"
          f"Val: {len(val_ids)} images, {len(val_ann)} annotations")

# Usage
split_dataset(
    images_dir='ovdetr/cytology_dataset',  # Original images folder
    ann_file='dataset/train.json',         # Original annotations
    output_dir='ovdetr/cytology_dataset',  # Output root
    train_ratio=0.8
)