import os
import shutil
import random
import kagglehub
import yaml

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

# List all images in fog folder
all_images = [f for f in os.listdir(fog_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
all_images.sort()
random.shuffle(all_images)

# Train/val split
train_ratio = 0.8
train_images = all_images[:int(len(all_images) * train_ratio)]
val_images = all_images[int(len(all_images) * train_ratio):]

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

# Create data.yaml
# This dataset only features the following five categories (not 8 as foggy_cityscape)
# Bus, Truck, Car, Person, Motorcycler + Bicycle
data_yaml = {
"train": os.path.join(split_dir, "train", "images"),
"val": os.path.join(split_dir, "val", "images"),
"nc": 5,
"names": ["bus", "truck", "car", "person", "(motor)bicycle"]
}

yaml_path = os.path.join(split_dir, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

print(f"data.yaml created at: {yaml_path}")
