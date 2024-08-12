import cv2
import random
import os
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def visualize_and_save_predictions(cfg, dataset_name, output_dir, num_samples=5):
    # Register the dataset if not already registered
    
    # Load dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # Load the model
    predictor = DefaultPredictor(cfg)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        d = random.choice(dataset_dicts)
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        # Check if outputs is a list and handle accordingly
        if isinstance(outputs, list):
            instances = None
            for item in outputs:
                if isinstance(item, dict) and "instances" in item:
                    instances = item["instances"].to("cpu")
                    break
            if instances is None:
                print(f"Warning: Expected 'instances' in outputs but did not find any. Skipping this sample.")
                continue
        else:
            if "instances" in outputs:
                instances = outputs["instances"].to("cpu")
            else:
                print(f"Warning: Expected 'instances' in outputs but did not find any. Skipping this sample.")
                continue
        
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        out = v.draw_instance_predictions(instances)
        
        output_path = os.path.join(output_dir, f"prediction_{i}.png")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        print(f"Saved visualization to {output_path}")

# Load configuration
cfg = get_cfg()
cfg.merge_from_file("DGE-CDA/configs/jltv_source.yaml")
cfg.MODEL.WEIGHTS = "DGE-CDA/givendata/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Update based on your dataset

IMGS = "DGE-CDA/givendata/imgs/"
register_coco_instances(cfg.DATASETS.TRAIN[0], {},"DGE-CDA/givendata/source_train.json", IMGS)
register_coco_instances(cfg.DATASETS.TRAIN[1], {},"DGE-CDA/givendata/source_valid.json", IMGS)
register_coco_instances(cfg.DATASETS.TEST[0], {},"DGE-CDA/givendata/target.json", IMGS)

# Define the output directory
output_dir = "drive/MyDrive/output/predictions"

# Call the visualization function
visualize_and_save_predictions(cfg, "target", output_dir, num_samples=5)  # Change "target" to your dataset name