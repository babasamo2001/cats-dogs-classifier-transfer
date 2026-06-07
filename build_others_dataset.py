import os
import random
import shutil
from pathlib import Path

# CONFIG

# Folder containing the 6 downloaded classes (no cat or dog images)
SOURCE_ROOT = "natural_images"

# Existing dataset root
DATASET_ROOT = "data"

# Classes to merge into "others"
OTHER_CLASSES = [
    "airplane",
    "car",
    "flower",
    "fruit",
    "motorbike",
    "person"
]

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

RANDOM_SEED = 42

# OUTPUT FOLDERS

train_output = os.path.join(DATASET_ROOT, "train", "others")
val_output = os.path.join(DATASET_ROOT, "val", "others")
test_output = os.path.join(DATASET_ROOT, "test", "others")

os.makedirs(train_output, exist_ok=True)
os.makedirs(val_output, exist_ok=True)
os.makedirs(test_output, exist_ok=True)

# COLLECT ALL IMAGES

all_images = []

valid_extensions = [".jpg", ".jpeg", ".png"]

for class_name in OTHER_CLASSES:

    class_path = os.path.join(SOURCE_ROOT, class_name)

    if not os.path.exists(class_path):

        print(f"WARNING: Missing folder -> {class_path}")

        continue

    for ext in valid_extensions:

        images = list(Path(class_path).rglob(f"*{ext}"))

        all_images.extend(images)

print(f"\nFound {len(all_images)} total images.")

# SHUFFLE

random.seed(RANDOM_SEED)

random.shuffle(all_images)

# SPLIT

total = len(all_images)

train_count = int(total * TRAIN_SPLIT)
val_count = int(total * VAL_SPLIT)

train_images = all_images[:train_count]

val_images = all_images[
    train_count:train_count + val_count
]

test_images = all_images[
    train_count + val_count:
]

print("\nDataset Split:")
print(f"Train: {len(train_images)}")
print(f"Val:   {len(val_images)}")
print(f"Test:  {len(test_images)}")

# COPY FUNCTION

def copy_images(image_list, destination_folder):

    for idx, image_path in enumerate(image_list):

        image_path = str(image_path)

        filename = os.path.basename(image_path)

        new_filename = f"{idx}_{filename}"

        destination = os.path.join(
            destination_folder,
            new_filename
        )

        shutil.copy2(image_path, destination)

    print(f"Copied {len(image_list)} images -> {destination_folder}")

# COPY DATA

copy_images(train_images, train_output)

copy_images(val_images, val_output)

copy_images(test_images, test_output)

print("\nDONE BUILDING 'OTHERS' CLASS!")