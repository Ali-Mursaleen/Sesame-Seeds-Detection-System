import cv2
import albumentations as A
from tqdm import tqdm
from pathlib import Path


class SesameSeedAugmentor:
    def __init__(self, base_dir="Data", output_dir="Augmented_Data"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "Black").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "Healthy").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "Rain Damage").mkdir(exist_ok=True, parents=True)

        self.classes = ["Black", "Healthy", "Rain Damage"]

        self.augmentations = A.Compose(
            [
                A.Rotate(limit=180, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7
                ),
                A.RandomCrop(height=400, width=400, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.3,
                    brightness_by_max=True,
                    p=0.9,
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=30, p=0.4
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3
                ),
            ]
        )

    def load_images(self):
        images_by_class = {}

        for class_name in self.classes:
            class_dir = self.base_dir / class_name
            images = []

            image_files = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg"))

            print(f"Loading {len(image_files)} images from {class_name}")

            for img_path in tqdm(image_files, desc=f"Loading {class_name}"):
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                else:
                    print(f"Warning: Could not load {img_path}")

            images_by_class[class_name] = images

        return images_by_class

    def augment_image(self, image, num_augmentations=10):
        augmented_images = [image]

        for i in range(num_augmentations):
            try:
                augmented = self.augmentations(image=image)
                augmented_images.append(augmented["image"])
            except Exception as e:
                print(f"Warning: Augmentation failed: {e}")

                augmented_images.append(image.copy())

        return augmented_images

    def save_augmented_images(self, class_name, original_images, num_augmentations=15):
        save_dir = self.output_dir / class_name

        print(f"\nAugmenting {class_name} class...")

        total_saved = 0
        for idx, original_image in enumerate(
            tqdm(original_images, desc=f"Augmenting {class_name}")
        ):
            augmented_images = self.augment_image(original_image, num_augmentations)

            for aug_idx, aug_image in enumerate(augmented_images):
                save_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

                filename = (
                    f"{class_name.replace(' ', '_')}_{idx:03d}_aug_{aug_idx:03d}.jpg"
                )
                save_path = save_dir / filename

                cv2.imwrite(str(save_path), save_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

                total_saved += 1

        print(f"Saved {total_saved} images for {class_name}")
        return total_saved

    def create_dataset_report(self):
        print("\n" + "=" * 50)
        print("DATASET REPORT")
        print("=" * 50)

        print("\nOriginal Dataset:")
        total_original = 0
        for class_name in self.classes:
            class_dir = self.base_dir / class_name
            image_files = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg"))
            count = len(image_files)
            total_original += count
            print(f"  {class_name}: {count} images")

        print("\nAugmented Dataset:")
        total_augmented = 0
        for class_name in self.classes:
            aug_dir = self.output_dir / class_name
            if aug_dir.exists():
                image_files = list(aug_dir.glob("*.jpg"))
                count = len(image_files)
                total_augmented += count
                print(f"  {class_name}: {count} images")

        print(f"\nOriginal Total: {total_original} images")
        print(f"Augmented Total: {total_augmented} images")
        print(f"Multiplier: {total_augmented / total_original:.1f}x")
        print("=" * 50)

    def run_augmentation(self, augmentations_per_image=15):
        print("Starting Sesame Seed Data Augmentation")
        print("=" * 50)

        images_by_class = self.load_images()

        total_images = 0
        for class_name, images in images_by_class.items():
            if images:
                count = self.save_augmented_images(
                    class_name, images, augmentations_per_image
                )
                total_images += count

        self.create_dataset_report()

        print(f"\nAugmentation complete! Total images: {total_images}")
        print(f"Output saved to: {self.output_dir}")

        return total_images


def check_image_sizes():
    base_dir = Path("Data")
    for class_name in ["Black", "Healthy", "Rain Damage"]:
        class_dir = base_dir / class_name
        image_files = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg"))

        if image_files:
            img_path = image_files[0]
            img = cv2.imread(str(img_path))
            if img is not None:
                print(f"{class_name}: Image size = {img.shape[1]}x{img.shape[0]}")


def main():
    print("Checking image sizes...")
    check_image_sizes()

    augmentor = SesameSeedAugmentor(base_dir="Data", output_dir="Augmented_Data")

    augmentor.run_augmentation(augmentations_per_image=15)


if __name__ == "__main__":
    try:
        import albumentations
        import cv2
        import numpy as np
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        import subprocess

        subprocess.check_call(
            ["pip", "install", "albumentations", "opencv-python", "numpy", "tqdm"]
        )

    main()
