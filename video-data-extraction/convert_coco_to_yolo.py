import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import shutil

# -----------------------------------------------------------
# COCO → YOLO Converter
# -----------------------------------------------------------

def load_coco(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_bbox(size, box):
    """Convert COCO bbox [x,y,w,h] → YOLO normalized,
       handles strings or ints automatically."""

    img_w, img_h = size

    # Convert everything to float first
    x, y, w, h = map(float, box)

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h

    return x_center, y_center, w, h


def main(json_path, images_dir, out_dir):

    coco = load_coco(json_path)

    # --- Build category map (fix duplicate names) ---
    category_name_to_id = {}
    category_id_to_newid = {}
    next_id = 0

    for c in coco["categories"]:
        name = c["name"]
        if name not in category_name_to_id:
            category_name_to_id[name] = next_id
            next_id += 1
        category_id_to_newid[c["id"]] = category_name_to_id[name]

    # --- Create output folders ---
    for split in ["train", "valid", "test"]:
        create_dir(f"{out_dir}/{split}/images")
        create_dir(f"{out_dir}/{split}/labels")

    # --- Map images by ID ---
    images = {img["id"]: img for img in coco["images"]}

    # --- Group annotations by image ---
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    # --- Process each image ---
    print("Converting COCO → YOLO...")
    for img_id, imginfo in tqdm(images.items()):

        filename = imginfo["file_name"]
        img_path = os.path.join(images_dir, filename)

        if not os.path.exists(img_path):
            print(f"WARNING: Image not found: {img_path}")
            continue

        # Load actual image size (COCO sometimes has wrong values)
        with Image.open(img_path) as im:
            w, h = im.size

        # Pick split based on modulo (can customize)
        if img_id % 10 == 0:
            split = "valid"
        elif img_id % 10 == 1:
            split = "test"
        else:
            split = "train"

        out_img_path = f"{out_dir}/{split}/images/{filename}"
        out_label_path = f"{out_dir}/{split}/labels/{filename.replace('.jpg', '.txt').replace('.png', '.txt')}"

        # Copy image
        shutil.copy(img_path, out_img_path)

        # Write label file
        lines = []
        for ann in ann_by_image[img_id]:
            cat_new = category_id_to_newid[ann["category_id"]]
            bbox = ann["bbox"]
            bbox_yolo = convert_bbox((w, h), bbox)
            line = f"{cat_new} " + " ".join(f"{x:.6f}" for x in bbox_yolo)
            lines.append(line)

        with open(out_label_path, "w") as f:
            f.write("\n".join(lines))

    # --- Write data.yaml ---
    yaml_path = f"{out_dir}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {len(category_name_to_id)}\n")
        f.write("names:\n")
        for name, new_id in sorted(category_name_to_id.items(), key=lambda x: x[1]):
            f.write(f"  {new_id}: {name}\n")

    print("\n✔ Conversion Complete!")
    print(f"YOLO dataset saved to: {out_dir}")
    print(f"Classes: {category_name_to_id}")


# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to COCO annotations.json")
    parser.add_argument("--images", required=True, help="Folder with images")
    parser.add_argument("--out", required=True, help="Output folder for YOLO dataset")
    args = parser.parse_args()

    main(args.json, args.images, args.out)

#AFTER RUNNING TO TRAIN YOLO RUN THIS: yolo detect train data="path/to/data.yaml" model=yolov8n.pt epochs=100 imgsz=640 batch=16