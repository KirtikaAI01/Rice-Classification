import os
import shutil
import random

def split_data(source_dir, target_dir, split_ratio=0.8):
    classes = os.listdir(source_dir)
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)

        train_size = int(len(images) * split_ratio)
        train_imgs = images[:train_size]
        val_imgs = images[train_size:]

        for category in ["train", "val"]:
            dest_path = os.path.join(target_dir, category, cls)
            os.makedirs(dest_path, exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(target_dir, "train", cls, img))

        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(target_dir, "val", cls, img))

    print("✅ Data split complete!")

# Call function with full paths
split_data("C:/Users/nayan/Downloads/PPPP/raw_data", "C:/Users/nayan/Downloads/PPPP/data", split_ratio=0.8)

