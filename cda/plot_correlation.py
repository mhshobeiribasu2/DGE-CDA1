# from detectron2.utils.logger import setup_logger
# setup_logger()
import matplotlib.pyplot as plt
import time
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from domain_gap import compute_domain_gap

# list of conditions
CONDITIONS = ["source_valid","target"];



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='plot correlation between domain gap and detection accuracy')
    parser.add_argument('--config-file', default="", metavar='FILE', help='path to config file');
    parser.add_argument('--projs-dir', default="", type=str, help="directory of projections")
    parser.add_argument('--imgs-dir', default="datasets/DGTA_SeaDronesSee_merged/images", type=str, help='directory of images');
    parser.add_argument('--annos-dir', default="datasets/DGTA_SeaDronesSee_merged/experiments", type=str, help='directory of jsons');
    parser.add_argument('--n-samples', default=1000, type=int, help="number of random samples for computing domain gap")
    
    args = parser.parse_args()
    return args

# register training or testing set of each conditions
# "set" can be either "train" or "val"
def register_datasets(args, set):
    for cond in CONDITIONS:
        register_coco_instances(set + "_" + cond, {}, args.annos_dir + "/" + set + "_" + cond + ".json", args.imgs_dir);

# plot histogram
def plot_bar_chart(x, y, y_label, filename, rot=40):
    plt.figure()
    plt.bar(x, y)
    plt.ylabel(y_label)
    plt.xticks(rotation=rot)  # Rotate x-axis values for display
    plt.savefig(filename)     # Save the figure to a file
    plt.show() 

def main():
    
    ti = time.time();
    args = parse_args();
    print("\nCommand Line Args:", args);
    print("\n");

    # load model
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file);
    model = build_model(cfg);
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.OUTPUT_DIR + "/model_final.pth");
    model.eval(); # inference mode

    # register training & testing sets of source domain
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, args.annos_dir + "/" + cfg.DATASETS.TRAIN[0] + ".json", args.imgs_dir)
    register_coco_instances(cfg.DATASETS.TEST[0], {}, args.annos_dir + "/" + cfg.DATASETS.TEST[0] + ".json", args.imgs_dir)
    register_coco_instances(cfg.DATASETS.TEST[1], {}, args.annos_dir + "/" + cfg.DATASETS.TEST[1] + ".json", args.imgs_dir)

    # register training & testing sets of target domains
    # register_datasets(args, set="train");    
    # register_datasets(args, set="val");
    
    # evaludate detection accuracy on testing sets of source domain
    print("\n===============> Evaluate on %s" % cfg.DATASETS.TEST[0]);
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir="./output/");
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0]);
    res = inference_on_dataset(model, val_loader, evaluator);
    source_ap50 = res['bbox']['AP50'];
    
    # evaluate detection accuracy on testing sets of target domains
    ap50_discrepancy = list();
    for condition in CONDITIONS:
        dataset_name = condition;
        print("\n===============> Evaluate on %s" % condition);
        evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/");
        val_loader = build_detection_test_loader(cfg, dataset_name);
        res = inference_on_dataset(model, val_loader, evaluator);
        ap50_discrepancy.append(abs(res['bbox']['AP50'] - source_ap50));


    # compute domain gap with DSS, SWD, and MMD
    dss_list = list();
    swd_list = list();
    mmd_list = list();
    for condition in CONDITIONS:
        dataset_name = condition;
        print("\n===============> Compute DSS on %s" % condition);
        dss = compute_domain_gap(cfg=cfg, model=model, projections_dir=args.projs_dir, \
            current_data=tuple([cfg.DATASETS.TRAIN[0]]), new_data=dataset_name, metric="DSS", n_samples=args.n_samples);
        dss_list.append(dss.item());

        print("\n===============> Compute SWD on %s" % condition);
        swd = compute_domain_gap(cfg=cfg, model=model, projections_dir=args.projs_dir, \
            current_data=tuple([cfg.DATASETS.TRAIN[0]]), new_data=dataset_name, metric="SWD", n_samples=args.n_samples);
        swd_list.append(swd.item());

        print("\n===============> Compute MMD on %s" % condition);
        mmd = compute_domain_gap(cfg=cfg, model=model, projections_dir=args.projs_dir, \
            current_data=tuple([cfg.DATASETS.TRAIN[0]]), new_data=dataset_name, metric="MMD", n_samples=args.n_samples);
        mmd_list.append(mmd.item());

    # Plot histograms
    x_values = ["source_valid","target"];
    plot_bar_chart(x=x_values, y=ap50_discrepancy, y_label="AP discrepancy", filename="ap_discrepancy.png")
    plot_bar_chart(x=x_values, y=dss_list, y_label="DSS", filename="dss.png")
    plot_bar_chart(x=x_values, y=swd_list, y_label="SWD", filename="swd.png")
    plot_bar_chart(x=x_values, y=mmd_list, y_label="MMD", filename="mmd.png")
    print("Elapsed time = %.2fs" % (time.time() - ti));
    # Show the plot
    plt.show()
    

if __name__ == "__main__":
    main();


    