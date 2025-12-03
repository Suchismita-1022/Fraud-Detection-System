import os
from PIL import Image
import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fraud_detection_project/
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")  # fraud_detection_project/data/raw/
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  # fraud_detection_project/data/processed/

# Create processed directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Parameters ---
IMG_SIZE = (224, 224)  # Standard ResNet input size


def preprocess_image(img_path, output_path):
    """
    Preprocess image for ResNet training:
    - Convert to RGB
    - Resize to 224x224
    - Save as high-quality JPEG
    """
    try:
        print(f"Processing: {img_path}")

        # Open and convert to RGB
        img = Image.open(img_path).convert("RGB")
        original_size = img.size

        # Resize using high-quality resampling
        img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        # Ensure output path has .jpg extension
        output_path = os.path.splitext(output_path)[0] + ".jpg"

        # Save as high-quality JPEG
        img_resized.save(output_path, "JPEG", quality=95, optimize=True)

        print(f"  ✓ {original_size} → {IMG_SIZE} → {output_path}")
        return output_path

    except Exception as e:
        print(f"  ✗ Error processing {img_path}: {e}")
        return None


def get_processed_paths():
    """
    Extract processed paths based on your folder structure.
    Returns dictionary with category paths.
    """
    paths = {
        "base_dir": BASE_DIR,
        "raw_dir": RAW_DIR,
        "processed_dir": PROCESSED_DIR,
        "categories": {}
    }

    # Check if raw directory exists
    if not os.path.exists(RAW_DIR):
        print(f"❌ Raw directory not found: {RAW_DIR}")
        return paths

    # Scan for categories in raw folder
    for item in os.listdir(RAW_DIR):
        category_path = os.path.join(RAW_DIR, item)
        if os.path.isdir(category_path):
            processed_category_path = os.path.join(PROCESSED_DIR, item)
            paths["categories"][item] = {
                "raw_path": category_path,
                "processed_path": processed_category_path
            }

    return paths


def process_documents():
    """
    Main preprocessing function that processes all images in raw folders.
    """
    print("🚀 Starting document preprocessing...")
    print(f"📂 Project root: {BASE_DIR}")
    print(f"📁 Raw data: {RAW_DIR}")
    print(f"📁 Processed data: {PROCESSED_DIR}")

    # Get folder structure
    paths = get_processed_paths()

    if not paths["categories"]:
        print("❌ No categories found in raw folder!")
        print(f"Expected structure: {RAW_DIR}/[category_name]/")
        return

    print(f"\n📋 Found categories: {list(paths['categories'].keys())}")

    total_processed = 0
    total_errors = 0

    # Process each category
    for category, category_paths in paths["categories"].items():
        print(f"\n📂 Processing category: {category}")

        raw_folder = category_paths["raw_path"]
        processed_folder = category_paths["processed_path"]

        # Create processed category folder
        os.makedirs(processed_folder, exist_ok=True)

        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        image_files = [f for f in os.listdir(raw_folder)
                       if f.lower().endswith(image_extensions)]

        if not image_files:
            print(f"  ⚠  No images found in {raw_folder}")
            continue

        print(f"  📊 Found {len(image_files)} images")

        category_processed = 0
        category_errors = 0

        # Process each image
        for i, filename in enumerate(image_files, 1):
            raw_img_path = os.path.join(raw_folder, filename)
            processed_img_path = os.path.join(processed_folder, filename)

            print(f"  [{i:3d}/{len(image_files)}] ", end="")

            result = preprocess_image(raw_img_path, processed_img_path)

            if result:
                category_processed += 1
                total_processed += 1
            else:
                category_errors += 1
                total_errors += 1

        print(f"  📈 {category} summary: {category_processed} processed, {category_errors} errors")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"🎉 PREPROCESSING COMPLETED!")
    print(f"✅ Total processed: {total_processed} images")
    print(f"❌ Total errors: {total_errors} images")
    print(f"📁 Output directory: {PROCESSED_DIR}")

    # Verify structure
    verify_processed_structure()


def verify_processed_structure():
    """
    Verify the processed folder structure and show sample files.
    """
    print(f"\n{'=' * 60}")
    print("🔍 VERIFYING PROCESSED STRUCTURE:")

    for category in os.listdir(PROCESSED_DIR):
        category_path = os.path.join(PROCESSED_DIR, category)
        if os.path.isdir(category_path):
            image_files = [f for f in os.listdir(category_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            print(f"\n📂 {category}/")
            print(f"  📊 {len(image_files)} processed images")

            # Show first 3 files as sample
            for i, filename in enumerate(image_files[:3]):
                img_path = os.path.join(category_path, filename)
                try:
                    with Image.open(img_path) as img:
                        print(f"  📄 {filename} → {img.size}, {img.mode}")
                except Exception as e:
                    print(f"  ❌ {filename} → Error: {e}")

            if len(image_files) > 3:
                print(f"  ... and {len(image_files) - 3} more files")


def get_dataset_info():
    """
    Get information about the processed dataset for training.
    """
    dataset_info = {
        "processed_dir": PROCESSED_DIR,
        "categories": {},
        "total_images": 0
    }

    if not os.path.exists(PROCESSED_DIR):
        return dataset_info

    for category in os.listdir(PROCESSED_DIR):
        category_path = os.path.join(PROCESSED_DIR, category)
        if os.path.isdir(category_path):
            image_files = [f for f in os.listdir(category_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            dataset_info["categories"][category] = {
                "path": category_path,
                "count": len(image_files),
                "sample_files": image_files[:3]
            }
            dataset_info["total_images"] += len(image_files)

    return dataset_info


if __name__ == "__main__":
    print(" FRAUD DETECTION - IMAGE PREPROCESSING")
    print("=" * 60)

    # Show current folder structure
    paths = get_processed_paths()
    print(f"  Folder Structure:")
    print(f"   Base: {paths['base_dir']}")
    print(f"   Raw:  {paths['raw_dir']}")
    print(f"   Proc: {paths['processed_dir']}")

    # Start processing
    process_documents()

    # Show dataset info for training
    print(f"\n{'=' * 60}")
    print(" DATASET INFO FOR TRAINING:")
    dataset_info = get_dataset_info()

    for category, info in dataset_info["categories"].items():
        print(f"  {category}: {info['count']} images")

    print(f"  Total: {dataset_info['total_images']} images")
    print(f"\n Ready for ResNet training!")
    print(f"   Use this path in your training script: {PROCESSED_DIR}")