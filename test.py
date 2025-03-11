import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from ovdetr.cytology_dataset.proposal_quality import ProposalEnhancer

class CytologyPreprocessor:
    def __init__(self, config, enhancer_config):
        self.enhancer_config = enhancer_config
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize ProposalEnhancer FIRST
        self.enhancer = ProposalEnhancer(enhancer_config)  # Add this line
        
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
            for cat in data['categories']:
                cat['split'] = 'base' if cat['name'] in base_classes else 'novel'
            self._save_json(data, split)
            print(f"Updated {split} annotations with class splits")

    def _process_image(self, img_info, split):
        """Process single image with error handling"""
        try:
            images_dir = self.base_path/'images'/split
            filename = Path(img_info['file_name']).name
            img_path = images_dir/filename
            
            # Skip if already processed
            if 'proposals' in img_info:
                return img_info
                
            im = cv2.imread(str(img_path))
            if im is None:
                print(f"⚠️ Missing image: {img_path}")
                return img_info
            
            # Resize large images to speed up processing
            h, w = im.shape[:2]
            if max(h, w) > 2000:
                im = cv2.resize(im, (w//2, h//2))
            
            # Initialize models once per process
            if not hasattr(self, 'edge_detector'):
                self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
                self.edge_boxes = cv2.ximgproc.createEdgeBoxes()
                self.edge_boxes.setMaxBoxes(100)
            
            im_float = im.astype(np.float32) / 255.0
            edges = self.edge_detector.detectEdges(im_float)
            boxes, scores = self.edge_boxes.getBoundingBoxes(edges)
            
            return {
                **img_info,
                'proposals': [
                    {"bbox": [float(x), float(y), float(w), float(h)], "score": float(s.item())}
                    for (x,y,w,h), s in zip(boxes, scores)
                ]
            }
        except Exception as e:
            print(f"❌ Error processing {img_info['file_name']}: {str(e)}")
            return img_info



    # Updated generate_proposals method
    def generate_proposals(self):
        """Quality-focused proposal generation with enhanced stability"""
        # Initialize thread-local OpenCV context
        class EdgeDetectionContext:
            def __init__(self, enhancer_config):
                self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
                self.edge_boxes = cv2.ximgproc.createEdgeBoxes()
                self.edge_boxes.setMaxBoxes(150)
                self.edge_boxes.setAlpha(enhancer_config.get('alpha', 0.7))
                self.edge_boxes.setBeta(enhancer_config.get('beta', 0.8))

        def process_image(img, images_dir):
            filename = Path(img['file_name']).name
            img_path = images_dir / filename
            ctx = EdgeDetectionContext(self.enhancer_config)  # Per-thread initialization

            try:
                # Phase 1: Image Validation
                im = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if im is None or im.size == 0 or im.shape[0] < 10 or im.shape[1] < 10:
                    print(f"🛑 Invalid image source: {filename}")
                    return img, []

                # Phase 2: Adaptive Resizing
                h, w = im.shape[:2]
                scale = 1.0
                if max(h, w) > 2000:
                    scale = 0.5
                    im = cv2.resize(im, (int(w*scale), int(h*scale)), 
                                    interpolation=cv2.INTER_AREA)

                # Phase 3: Edge Detection with Fallback
                try:
                    im_float = im.astype(np.float32) / 255.0
                    edges = ctx.edge_detector.detectEdges(im_float)
                    orientation_map = ctx.edge_detector.computeOrientation(edges)
                    edges = ctx.edge_detector.edgesNms(edges, orientation_map)
                except:
                    # Fallback to Canny edge detection
                    print(f"⚠️ StructuredEdge failed, using Canny: {filename}")
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    orientation_map = None

                # Phase 4: Proposal Generation
                try:
                    if orientation_map is not None:
                        boxes, scores = ctx.edge_boxes.getBoundingBoxes(edges, orientation_map)
                    else:
                        boxes, scores = ctx.edge_boxes.getBoundingBoxes(edges)
                except:
                    print(f"⚠️ EdgeBoxes failed: {filename}")
                    boxes, scores = [], []

                # Phase 5: Quality Filtering
                raw_proposals = [{
                    "bbox": [float(x/scale), float(y/scale), float(w/scale), float(h/scale)],
                    "score": float(s.item())
                } for (x,y,w,h), s in zip(boxes, scores)] if scale != 1.0 else [...] 

                filtered = self.enhancer.filter_proposals(raw_proposals)
                nms_proposals = self.enhancer.apply_nms(filtered)
                
                return img, nms_proposals

            except Exception as e:
                print(f"🚨 Critical failure: {filename} - {str(e)}")
                return img, []
            finally:
                del ctx  # Clean up OpenCV context

        # Processing pipeline
        for split in ['train', 'val']:
            data = self._load_json(split)
            images_dir = self.base_path / 'images' / split

            with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers for stability
                futures = [executor.submit(process_image, img, images_dir)
                        for img in data['images'] if 'proposals' not in img]
                
                for future in as_completed(futures):
                    img, proposals = future.result()
                    if proposals:
                        # Merge with existing proposals
                        existing = img.get('proposals', [])
                        merged = self.enhancer.apply_nms(existing + proposals)
                        img['proposals'] = merged[:150]  # Keep top proposals

            self._save_json(data, split)
            print(f"✅ Quality update completed for {split}")
    
    def _process_single_image(args):
        img_info, split, base_path = args
        try:
            # ... (same processing logic as above)
            return img_info
        except Exception as e:
            print(f"Error in {img_info['file_name']}: {str(e)}")
            return img_info
        
    def extract_clip_features(self):
        """Extract CLIP features for existing splits"""
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        
        for split in ['train', 'val']:
            data = self._load_json(split)
            features = {}
            images_dir = self.base_path/'images'/split
            
            print(f"\n🔧 Extracting CLIP features for {split} split")
            for img in tqdm(data['images'], desc="Processing images"):
                try:
                    img_path = images_dir/Path(img['file_name']).name
                    image = Image.open(img_path).convert("RGB")
                    image_input = preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        features[img['id']] = model.encode_image(image_input).cpu().numpy()
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {str(e)}")
                    features[img['id']] = np.zeros((1, 512))  # Fallback zero vector
            
            output_path = self.base_path/'features'/f"{split}_features.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, features)
            print(f"💾 Saved {len(features)} features to {output_path}")
    
    
    def visualize_proposals(self, split, num_samples=3):
        """Quality control visualization"""
        data = self._load_json(split)
        samples = random.sample(data['images'], num_samples)

        for img in samples:
            img_path = self.base_path/'images'/split/Path(img['file_name']).name
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(image)
            
            # Plot top proposals
            for p in sorted(img['proposals'], 
                        key=lambda x: x['score'], reverse=True)[:10]:
                x, y, w, h = p['bbox']
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=1, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x, y-2, f"{p['score']:.2f}", 
                    color='red', fontsize=8)
            
            plt.title(f"Top Proposals: {Path(img_path).name}")
            plt.show()

# Rest of the code remains the same...

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

enhancer_config = {
    'min_size': 32,    # Minimum proposal size (pixels)
    'max_size': 512,   # Maximum proposal size
    'min_score': 0.15, # Minimum confidence score
    'iou_thresh': 0.4  # NMS overlap threshold
}

enhancer_config.update({
    'alpha': 0.75,  # More emphasis on strong edges
    'beta': 0.85,   # Tighter grouping
    'min_score': 0.2,
    'iou_thresh': 0.5
})


# Configuration (Update these paths)
config = {
    'data_root': 'cytology_images',
    'train_ann': 'train.json',
    'val_ann': 'val.json',
    'base_classes': ["ascus", "asch", "agc", "trichomonas", "flora", "herps"],
    'enhancer': enhancer_config,
    'debug': True  # Enable visualization for first 100 images
}

if __name__ == '__main__':
    # Initialize processor
    processor = CytologyPreprocessor(config, enhancer_config)
    
    # # 1. Add class splits
    # processor.add_class_splits(config['base_classes'])
    
    # # 2. Generate proposals
    processor.generate_proposals()
    
    # # 3. Extract CLIP features
    # processor.extract_clip_features()
    
    
    print("Preparation complete!")
    