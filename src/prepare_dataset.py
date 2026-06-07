import os
import random
import shutil

RAW_DIR = "data/raw"

OUTPUT_DIR = "data"

TRAIN_SPLIT = 0.8

TEST_PER_CLASS = 1000

random.seed(42)

classes = [
    "cat",
    "dog",
    "others"
]

# CREATE OUTPUT FOLDERS

for split in ["train", "val", "test"]:

    for cls in classes:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                cls
            ),
            exist_ok=True
        )

# SPLIT DATA

for cls in classes:

    class_dir = os.path.join(
        RAW_DIR,
        cls
    )

    images = os.listdir(class_dir)

    images = [
        img for img in images
        if img.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    random.shuffle(images)

    # TEST

    test_images = images[:TEST_PER_CLASS]

    remaining_images = images[TEST_PER_CLASS:]

    # TRAIN / VAL

    train_size = int(
        len(remaining_images) * TRAIN_SPLIT
    )

    train_images = remaining_images[:train_size]

    val_images = remaining_images[train_size:]

    # COPY FUNCTION

    def copy_files(file_list, split_name):

        for file_name in file_list:

            src = os.path.join(
                class_dir,
                file_name
            )

            dst = os.path.join(
                OUTPUT_DIR,
                split_name,
                cls,
                file_name
            )

            shutil.copy2(src, dst)

    copy_files(train_images, "train")

    copy_files(val_images, "val")

    copy_files(test_images, "test")

    print(f"\n{cls.upper()} COMPLETE")
    print(f"Train: {len(train_images)}")
    print(f"Val: {len(val_images)}")
    print(f"Test: {len(test_images)}")

print("\nDataset preparation completed.")