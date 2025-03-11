import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


class ProposalEnhancer:
    def __init__(self, config):
        self.config = config
        self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
        self.edge_boxes = self._configure_edgeboxes()
        
    def _configure_edgeboxes(self):
        """Optimized parameters for medical images"""
        eb = cv2.ximgproc.createEdgeBoxes()
        eb.setMaxBoxes(150)  # Generate more initially
        eb.setAlpha(0.7)     # Edge strength weight (default 0.65)
        eb.setBeta(0.8)      # Edge grouping (default 0.75)
        eb.setMinScore(0.05) # Minimum proposal score
        eb.setGamma(0.9)     # Boundary-to-area ratio
        return eb

    def filter_proposals(self, proposals):
        """Quality filters"""
        min_size = self.config['min_size']
        max_size = self.config['max_size']
        min_score = self.config['min_score']
        
        return [
            p for p in proposals
            if (p['score'] >= min_score and
                p['bbox'][2] >= min_size and
                p['bbox'][3] >= min_size and
                p['bbox'][2] <= max_size and
                p['bbox'][3] <= max_size)
        ]

    def apply_nms(self, proposals, iou_thresh=0.5):
        """Non-Max Suppression"""
        boxes = []
        scores = []
        for p in proposals:
            x, y, w, h = p['bbox']
            boxes.append([x, y, x+w, y+h])  # Convert to x1,y1,x2,y2
            scores.append(p['score'])
        
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, 
            score_threshold=self.config['min_score'],
            nms_threshold=iou_thresh
        )
        
        return [proposals[i] for i in indices]

    def visualize(self, image_path, proposals, save_path=None):
        """Quality validation visualization"""
        img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(img)
        
        # Plot top 50 proposals
        for p in proposals[:50]:
            x, y, w, h = p['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1, 
                edgecolor=(1, 0.2, 0.2, 0.8),
                facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(x, y+15, f"{p['score']:.2f}", 
                    color='yellow', fontsize=8, 
                    bbox=dict(facecolor='black', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()

# Configuration
enhancer_config = {
    'min_size': 32,    # Minimum proposal size (pixels)
    'max_size': 512,   # Maximum proposal size
    'min_score': 0.15, # Minimum confidence score
    'iou_thresh': 0.4  # NMS overlap threshold
}