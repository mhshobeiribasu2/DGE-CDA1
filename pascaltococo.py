import os
import json
import xml.etree.ElementTree as ET

def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = filename
    img_id = filename[:filename.rfind(".")]
    if extract_num_from_imgid:
        img_id = ''.join(filter(str.isdigit, img_id))
        if img_id:
            img_id = int(img_id)
        else:
            img_id = filename  # Use filename as fallback ID
    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))
    image_info = {
        "file_name": img_name,
        "height": height,
        "width": width,
        "id": img_id,
    }
    return image_info

def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name').strip()  # Strip any extra whitespace
    print(f"Encountered label: '{label}'")  # Debugging print
    if label not in label2id:
        print(f"Warning: label '{label}' not in label2id mapping. Skipping this object.")
        return None
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Invalid box: {xmin}, {ymin}, {xmax}, {ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    area = o_width * o_height
    annotation = {
        "area": area,
        "iscrowd": 0,
        "image_id": None,
        "bbox": [xmin, ymin, o_width, o_height],
        "category_id": category_id,
        "id": None,
        "ignore": 0,
        "segmentation": [],
    }
    return annotation

def convert_voc_to_coco(voc_annotations_dir, voc_images_dir, output_json_path, label2id):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }
    bnd_id = 1
    for label, label_id in label2id.items():
        output_json_dict['categories'].append({
            "supercategory": label,
            "id": label_id,
            "name": label,
        })
    for xml_file in os.listdir(voc_annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
        annotation_path = os.path.join(voc_annotations_dir, xml_file)
        print(f"Processing {annotation_path}")
        with open(annotation_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()
        image_info = get_image_info(root)
        image_info['id'] = len(output_json_dict['images']) + 1
        output_json_dict['images'].append(image_info)
        for obj in root.findall('object'):
            annotation = get_coco_annotation_from_obj(obj, label2id)
            if annotation is not None:
                annotation['image_id'] = image_info['id']
                annotation['id'] = bnd_id
                bnd_id += 1
                output_json_dict['annotations'].append(annotation)
    
    with open(output_json_path, 'w') as f:
        json.dump(output_json_dict, f, indent=4)
    print(f"COCO annotation saved to {output_json_path}")

# Define paths and label2id mapping
voc_annotations_dir = "sep_images"
voc_images_dir = "sep_images"
output_json_path = "sep_coco/annotations_coco.json"

# Update label2id with all your labels and their corresponding unique IDs
label2id = {
    "label1": 1,
    "label2": 2,
    "label0": 0,
    # Add any additional labels here
}

# Convert VOC to COCO
convert_voc_to_coco(voc_annotations_dir, voc_images_dir, output_json_path, label2id)