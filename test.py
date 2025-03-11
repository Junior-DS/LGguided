import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt



class CytologyPreprocessor:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize paths
        self.base_path = Path(config['data_root']).absolute()
        self.annotation_files = {
            'train': self.base_path/'annotations'/config['train_ann'],
            'val': self.base_path/'annotations'/config['val_ann']
        }
        self._validate_paths()

    def _validate_paths(self):
        """Check all required paths exist before processing"""
        print("\n🔍 Validating paths:")
        required = [
            self.base_path/'images/train',
            self.base_path/'images/val',
            self.annotation_files['train'],
            self.annotation_files['val']
        ]
        
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            print("❌ Missing paths:")
            for p in missing:
                print(f"- {p}")
            raise FileNotFoundError(f"Missing {len(missing)} critical paths")
        
        print("✅ All paths validated successfully")

    def _load_json(self, split):
        with open(self.annotation_files[split]) as f:
            return json.load(f)
    
    def _save_json(self, data, split):
        with open(self.annotation_files[split], 'w') as f:
            json.dump(data, f, indent=2)

    def add_class_splits(self, base_classes):
        """Add base/novel split info to existing annotations"""
        for split in ['train', 'val']:
            data = self._load_json(split)
            
            # Update categories
            for cat in data['categories']:
                cat['split'] = 'base' if cat['name'] in base_classes else 'novel'
            
            self._save_json(data, split)
            print(f"Updated {split} annotations with class splits")


    def generate_proposals(self):
        """Add EdgeBox proposals to existing annotations"""
        # Load edge detection model
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
        edge_boxes = cv2.ximgproc.createEdgeBoxes()
        edge_boxes.setMaxBoxes(100)

        for split in ['train', 'val']:
            data = self._load_json(split)
            images_dir = self.base_path/'images'/split
            
            for img in tqdm(data['images'], desc=f"Processing {split} images"):
                # Clean file path construction
                filename = Path(img['file_name']).name
                img_path = images_dir/filename  # Correct path
                
                # Debug print (remove after testing)
                print(f"Processing: {img_path}")
                
                im = cv2.imread(str(img_path))
                if im is None:
                    print(f"Missing image: {img_path}")
                    continue
                
                # Convert to float32 in [0,1] range
                im_float = im.astype(np.float32) / 255.0
                
                # Generate edge and orientation maps
                edges = edge_detector.detectEdges(im_float)
                orientation_map = edge_detector.computeOrientation(edges)
                
                # Get boxes with BOTH required parameters
                boxes, scores = edge_boxes.getBoundingBoxes(edges, orientation_map)
                
                # Store proposals
                img['proposals'] = [
                    {
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(s)
                    }
                    for (x,y,w,h), s in zip(boxes, scores)
                ]
            
            self._save_json(data, split)
            print(f"Added proposals to {split} annotations")

    def extract_clip_features(self):
        """Extract CLIP features for existing splits"""
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        
        for split in ['train', 'val']:
            data = self._load_json(split)
            features = {}
            images_dir = self.base_path/'images'/split
            
            for img in tqdm(data['images'], desc=f"Extracting {split} features"):
                img_path = images_dir/img['file_name']
                image = Image.open(img_path).convert("RGB")  # Fixed Image.open
                image_input = preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features[img['id']] = model.encode_image(image_input).cpu().numpy()  # Fixed encode_image
            
            output_path = self.base_path/'features'/f"{split}_features.npy"
            output_path.parent.mkdir(exist_ok=True)
            np.save(output_path, features)
            print(f"Saved {split} features to {output_path}")

class CytologyDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.base_path = Path(config['data_root'])
        
        # Load annotations
        with open(self.base_path/'annotations'/config[f"{split}_ann"]) as f:
            self.data = json.load(f)
        
        # Load features
        self.features = np.load(
            self.base_path/'features'/f"{split}_features.npy",
            allow_pickle=True
        ).item()
        
        # Create mappings
        self.category_map = {c['id']: c for c in self.data['categories']}
        self.image_map = {img['id']: img for img in self.data['images']}

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        annotations = [
            ann for ann in self.data['annotations']
            if ann['image_id'] == img_info['id']  # Fixed key name
        ]
        
        return {
            'image_path': str(self.base_path/'images'/self.split/img_info['file_name']),
            'image_id': img_info['id'],
            'proposals': img_info.get('proposals', []),
            'clip_features': torch.tensor(self.features[img_info['id']]),
            'annotations': annotations,
            'categories': self.category_map
        }

# Configuration (Update these paths)
config = {
    'data_root': 'cytology_images',
    'train_ann': 'train.json',
    'val_ann': 'val.json',
    'base_classes': ["ascus", "asch", "agc", "trichomonas", "flora", "herps"]  # Fixed duplicate entry
}

if __name__ == '__main__':
    # Initialize processor
    processor = CytologyPreprocessor(config)
    
    # 1. Add class splits
    processor.add_class_splits(config['base_classes'])
    
    # 2. Generate proposals
    processor.generate_proposals()
    
    # 3. Extract CLIP features
    processor.extract_clip_features()
    
    print("Preparation complete!")