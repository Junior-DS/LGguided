# File: add_proposals.py
import json
import os
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from tqdm import tqdm

class ProposalGenerator:
    def __init__(self, top_n=100):
        self.top_n = top_n
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, image_path):
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        instances = outputs["instances"]
        
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        proposals = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            proposals.append({
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(score)
            })
        
        return sorted(proposals, key=lambda x: x["score"], reverse=True)[:self.top_n]

def add_proposals_to_annotations(ann_file, output_file, image_root):
    generator = ProposalGenerator()
    
    with open(ann_file) as f:
        data = json.load(f)
    
    image_map = {img['id']: img for img in data['images']}
    
    for ann in tqdm(data['annotations'], desc='Processing annotations'):
        img = image_map[ann['image_id']]
        img_path = os.path.join(image_root, img['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        # Generate proposals once per image
        if 'proposals' not in img:
            img['proposals'] = generator(img_path)
        
        # Add proposals to annotation
        ann['proposals'] = img['proposals']
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    # For training set
    add_proposals_to_annotations(
        ann_file='cytology_dataset/annotations/cytology_train.json',
        output_file='cytology_dataset/annotations/cytology_proposals_train.json',
        image_root='cytology_dataset/images'
    )
    
    # For validation set
    add_proposals_to_annotations(
        ann_file='cytology_dataset/annotations/cytology_val.json',
        output_file='cytology_dataset/annotations/cytology_proposals_val.json',
        image_root='cytology_dataset/images'
    )