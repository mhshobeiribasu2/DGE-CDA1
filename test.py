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

cfg = get_cfg()
cfg.merge_from_file("DGE-CDA/configs/jltv_source.yaml")
cfg.DATASETS.TRAIN = ("source_train", "source_valid")
cfg.DATASETS.TEST = ("target",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 2000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Update based on your dataset
cfg.OUTPUT_DIR = "./output/"
IMGS = "DGE-CDA/givendata/imgs/"
register_coco_instances(cfg.DATASETS.TRAIN[0], {},"DGE-CDA/givendata/source_train.json", IMGS)
register_coco_instances(cfg.DATASETS.TRAIN[1], {},"DGE-CDA/givendata/source_valid.json", IMGS)
register_coco_instances(cfg.DATASETS.TEST[0], {},"DGE-CDA/givendata/target.json", IMGS)

results = cross_validation(cfg, "source_train")
print("Cross-validation results:", results)