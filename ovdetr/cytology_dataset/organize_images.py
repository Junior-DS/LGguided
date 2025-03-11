import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(images_dir, ann_file, output_dir, train_ratio=0.8):
    """
    Splits dataset while preserving original JSON structure exactly
    Maintains all four keys: images, categories, annotations, type
    """
    
    # --- 1. Load Original Data ---
    with open(ann_file) as f:
        original_data = json.load(f)

    # --- 2. Validate Structure ---
    required_keys = {'images', 'categories', 'annotations', 'type'}
    if missing := required_keys - original_data.keys():
        raise ValueError(f"Missing required keys: {missing}")

    # --- 3. Prepare Image ID Mapping ---
    image_id_map = {img['id']: img['file_name'] for img in original_data['images']}
    
    # --- 4. Create Split ---
    all_ids = list(image_id_map.keys())
    random.shuffle(all_ids)
    split_idx = int(len(all_ids) * train_ratio)
    train_ids, val_ids = set(all_ids[:split_idx]), set(all_ids[split_idx:])

    # --- 5. Create Directories ---
    (train_dir := Path(output_dir)/'train').mkdir(parents=True, exist_ok=True)
    (val_dir := Path(output_dir)/'val').mkdir(parents=True, exist_ok=True)

    # --- 6. Copy Images & Update Paths ---
    for img_id in tqdm(all_ids, desc='Processing images'):
        src = Path(images_dir)/image_id_map[img_id]
        is_train = img_id in train_ids
        
        # Copy image
        dest_dir = train_dir if is_train else val_dir
        new_path = f"{'train' if is_train else 'val'}/{image_id_map[img_id]}"
        shutil.copy2(src, dest_dir/image_id_map[img_id])
        
        # Update path in original data copy
        for img in original_data['images']:
            if img['id'] == img_id:
                img['file_name'] = new_path

    # --- 7. Split Annotations ---
    train_ann = [ann for ann in original_data['annotations'] if ann['image_id'] in train_ids]
    val_ann = [ann for ann in original_data['annotations'] if ann['image_id'] in val_ids]

    # --- 8. Build Output Structure ---
    def create_split(ids, ann):
        return {
            "images": [img for img in original_data['images'] if img['id'] in ids],
            "categories": original_data['categories'],  # Preserve original
            "annotations": ann,
            "type": original_data['type']  # Preserve original structure
        }

    # --- 9. Save Splits ---
    with open(Path(output_dir)/'train.json', 'w') as f:
        json.dump(create_split(train_ids, train_ann), f, indent=2)
    
    with open(Path(output_dir)/'val.json', 'w') as f:
        json.dump(create_split(val_ids, val_ann), f, indent=2)

    # --- 10. Report ---
    print(f'''
    ✅ Split Complete
    -----------------
    Original Structure Preserved For:
    - categories ({len(original_data['categories'])} classes)
    - type ({len(original_data['type'])} entries)
    
    Train: {len(train_ids)} images, {len(train_ann)} annotations
    Val:   {len(val_ids)} images, {len(val_ann)} annotations
    Output: {Path(output_dir).resolve()}
    ''')

# Usage
if __name__ == "__main__":
    split_dataset(
        images_dir='dataset/train',
        ann_file='dataset/train.json',
        output_dir='cytology_images',
        train_ratio=0.8
    )