import os
import shutil
import random
import kagglehub
import yaml
import glob

# Paths
# current file path
proj_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(proj_dir, "..", "data", "DAWN")
print(output_dir)

# Set cache/output directory
os.environ["KAGGLEHUB_CACHE"] = output_dir

# Download latest version
path = kagglehub.dataset_download("orvile/dawn-detection-in-adverse-weather-nature")
                                  #path="DAWN/Fog/Fog/foggy-001.jpg") # Idk how to only download the fog folder...

print("Path to downloaded folder:", path)

fog_path = os.path.join(path,"DAWN", "Fog", "Fog")

# Create train/val split folders
split_dir = os.path.join(output_dir, "own_train_val_split")
for split in ["train", "val"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(split_dir, split, sub), exist_ok=True)

# Create data.yaml

labels = set()
for file in glob.glob("*.txt", 
                      root_dir=os.path.join(fog_path, "Fog_YOLO_darknet"),
                      recursive=True):
    with open(os.path.join(fog_path, "Fog_YOLO_darknet", file)) as f:
        for line in f:
            cls = int(line.split()[0])
            labels.add(cls)

print("Annotated labels contain classes: ", sorted(labels))

# This dataset should only feature the following five categories (not 8 as foggy_cityscape)
# Bus, Truck, Car, Person, Motorcycler + Bicycle
# However, 6 classes where found 
# - bus (6)
# - truck (8)
# - car (3)
# - person (1)
# - motorcycle (4)
# - bicycle (2)
# no annotations for 5 and 7

# map annotations to match the cityscapes dataset
# [person (0), car (1), train (2), rider (3), truck (4), motorcycle(5), bicycle(6), bus(7)]


mapping = {1:0, 2:6, 3:1, 4:5, 6:7, 8:4}

for file in glob.glob("*.txt", 
                      root_dir=os.path.join(fog_path, "Fog_YOLO_darknet"),
                      recursive=True):
    lines = []
    with open(os.path.join(fog_path, "Fog_YOLO_darknet", file)) as f:
        for line in f:
            parts = line.strip().split()
            original = int(parts[0])
            parts[0] = str(mapping[original])
            lines.append(" ".join(parts))

    with open(os.path.join(fog_path, "Fog_YOLO_darknet", file), "w") as f:
        f.write("\n".join(lines))

data_yaml = {
"train": os.path.join(split_dir, "train", "images"),
"val": os.path.join(split_dir, "val", "images"),
"nc": 8,
"names": ["person", "car", "train", "rider", "truck", "motorcycle", "bicycle", "bus"]
}

yaml_path = os.path.join(split_dir, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

print(f"data.yaml created at: {yaml_path}")

# List all images in fog folder
fog_images = [f for f in os.listdir(fog_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
fog_images.sort()
random.shuffle(fog_images)

# Train/val split
train_ratio = 0.8
train_images = fog_images[:int(len(fog_images) * train_ratio)]
val_images = fog_images[int(len(fog_images) * train_ratio):]

# Function to copy images and labels
def copy_files(image_list, split):
    for img_file in image_list:
        # Copy image
        shutil.copy(os.path.join(fog_path, img_file),
        os.path.join(split_dir, split, "images", img_file))
        # Copy label if exists
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(fog_path, "Fog_YOLO_darknet", label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(split_dir, split, "labels", label_file))

# Copy train and val files
copy_files(train_images, "train")
copy_files(val_images, "val")

print(f"Train/Val split done. Train: {len(train_images)}, Val: {len(val_images)}")
print(f"Data available in: {split_dir}")
