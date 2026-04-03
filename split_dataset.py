import os
import shutil
import random
from tqdm import tqdm

# Paths
SOURCE_DIR = "data/garbage_classification"
DEST_DIR = "data/garbage_classification_processed"

# Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_dirs(base_path, classes):
    for split in ["train", "val", "test"]:
        for cls in classes:
            path = os.path.join(base_path, split, cls)
            os.makedirs(path, exist_ok=True)


def split_data():
    classes = os.listdir(SOURCE_DIR)
    create_dirs(DEST_DIR, classes)

    for cls in classes:
        cls_path = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(cls_path)

        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for img in tqdm(train_imgs, desc=f"{cls} - train"):
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(DEST_DIR, "train", cls, img))

        for img in tqdm(val_imgs, desc=f"{cls} - val"):
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(DEST_DIR, "val", cls, img))

        for img in tqdm(test_imgs, desc=f"{cls} - test"):
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(DEST_DIR, "test", cls, img))


if __name__ == "__main__":
    split_data()
    print("\nDataset split completed!")

