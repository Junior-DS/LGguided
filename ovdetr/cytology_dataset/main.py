from cytology_preprocessor import CytologyPreprocessor
from proposal_quality import ProposalEnhancer
from proposal_quality import enhancer_config
import json




if __name__ == '__main__':
    config = {
        'data_root': 'cytology_images',
        'train_ann': 'train.json',
        'val_ann': 'val.json',
        'base_classes': [...],
        'enhancer': enhancer_config,
        'debug': True  # Enable visualization for first 100 images
    }
    
    processor = CytologyPreprocessor(config)
    
    # 1. Generate and enhance proposals
    processor.generate_proposals()
    
    # 2. Quality validation
    sample_image = 'cytology_images/images/train/02859.bmp'
    with open('cytology_images/annotations/train.json') as f:
        data = json.load(f)
        sample_proposals = next(
            img['proposals'] for img in data['images'] 
            if img['file_name'] == 'train/02859.bmp'
        )
    
    ProposalEnhancer(enhancer_config).visualize(
        sample_image, 
        sample_proposals,
        'quality_check.jpg'
    )