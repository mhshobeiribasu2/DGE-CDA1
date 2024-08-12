import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

def cross_validation(cfg, dataset_name, num_folds=5):
    # Load dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)
    random.shuffle(dataset_dicts)
    
    fold_size = len(dataset_dicts) // num_folds
    results = []
    
    for fold in range(num_folds):
        train_dataset = dataset_dicts[:fold * fold_size] + dataset_dicts[(fold + 1) * fold_size:]
        val_dataset = dataset_dicts[fold * fold_size:(fold + 1) * fold_size]
        
        # Register temporary datasets
        DatasetCatalog.register("temp_train", lambda: train_dataset)
        DatasetCatalog.register("temp_val", lambda: val_dataset)
        MetadataCatalog.get("temp_train").set(thing_classes=MetadataCatalog.get(dataset_name).thing_classes)
        MetadataCatalog.get("temp_val").set(thing_classes=MetadataCatalog.get(dataset_name).thing_classes)
        
        # Train model
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # Evaluate model
        evaluator = COCOEvaluator("temp_val", cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "temp_val")
        results.append(inference_on_dataset(trainer.model, val_loader, evaluator))
        
        # Unregister temporary datasets
        DatasetCatalog.unregister("temp_train")
        DatasetCatalog.unregister("temp_val")
    
    return results

def evaluate_on_test_set(cfg, model_path, test_dataset_name):
    cfg.MODEL.WEIGHTS = model_path
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator(test_dataset_name, cfg, False, output_dir="./output/")
    test_loader = build_detection_test_loader(cfg, test_dataset_name)
    return inference_on_dataset(predictor.model, test_loader, evaluator)

cfg = get_cfg()
cfg.merge_from_file("configs/jltv_source.yaml")
cfg.MODEL.WEIGHTS = "givendata/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Update based on your dataset
cfg.OUTPUT_DIR = "./sep_output/"
register_coco_instances("new_test_set", {}, "sep_coco/annotations_coco.json", "sep_images")

test_results = evaluate_on_test_set(cfg, "givendata/model_final.pth", "new_test_set")
print("Test set evaluation results:", test_results)