import os
import tensorflow as tf

from PIL import Image

# =========================================================
# CONFIG
# =========================================================

DATASET_DIR = "./data"

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
)

# =========================================================
# MAIN CLEANING FUNCTION
# =========================================================

def clean_and_convert_dataset(dataset_dir):

    print(f"\nStarting dataset cleanup in: {dataset_dir}\n")

    healthy_count = 0
    converted_count = 0
    deleted_count = 0

    # WALK THROUGH DATASET

    for root, dirs, files in os.walk(dataset_dir):

        for file in files:

            file_path = os.path.join(root, file)

            file_lower = file.lower()

            # REMOVE SYSTEM FILES

            if file.startswith('.') or file_lower == "thumbs.db":

                print(f"Removing system file: {file_path}")

                try:
                    os.remove(file_path)
                    deleted_count += 1

                except Exception as e:
                    print(f"Failed to remove system file: {e}")

                continue

            # SKIP NON-IMAGE FILES

            if not file_lower.endswith(VALID_EXTENSIONS):

                print(f"Skipping non-image file: {file_path}")

                continue

            # PROCESS IMAGE

            try:

                # PIL VALIDATION

                with Image.open(file_path) as img:

                    original_format = img.format

                    # verify image integrity
                    img.verify()

                # reopen after verification
                with Image.open(file_path) as img:

                    # TENSORFLOW VALIDATION

                    img_tf = tf.io.read_file(file_path)

                    img_tf = tf.image.decode_image(
                        img_tf,
                        channels=3
                    )

                    # CONVERT NON-JPEG IMAGES

                    if original_format not in ["JPEG", "MPO"]:

                        rgb_img = img.convert("RGB")

                        base_path, _ = os.path.splitext(file_path)

                        new_jpg_path = base_path + ".jpg"

                        rgb_img.save(
                            new_jpg_path,
                            "JPEG",
                            quality=95
                        )

                        # remove old image if different
                        if file_path != new_jpg_path:

                            os.remove(file_path)

                        print(f"Converted: {file_path} -> {new_jpg_path}")

                        converted_count += 1

                    else:

                        healthy_count += 1

            # HANDLE CORRUPTED FILES

            except (
                IOError,
                SyntaxError,
                Image.DecompressionBombError,
                tf.errors.InvalidArgumentError,
                Exception
            ) as e:

                print(f"Deleting corrupted image: {file_path}")
                print(f"Reason: {e}")

                try:

                    os.remove(file_path)

                    deleted_count += 1

                except Exception as delete_error:

                    print(f"Failed to delete file: {delete_error}")

    # SUMMARY

    print("\n=================================================")
    print("DATASET CLEANUP COMPLETE")
    print("=================================================")

    print(f"Healthy JPEG images kept : {healthy_count}")
    print(f"Images converted to JPG  : {converted_count}")
    print(f"Corrupted files deleted  : {deleted_count}")

if __name__ == "__main__":

    clean_and_convert_dataset(DATASET_DIR)