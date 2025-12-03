import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import torch


class YOLODocumentDetector:
    """YOLO-based document detector for fraud detection system."""

    def __init__(self, model_path=None):
        """
        Initialize YOLO detector.

        Args:
            model_path (str): Path to YOLO model. If None, uses YOLOv8n pretrained.
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.results_dir = Path("results/detections")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ YOLO detector initialized with model: {self.model_path}")

    def _load_model(self):
        """Load YOLO model."""
        try:
            if self.model_path and Path(self.model_path).exists():
                print(f"📥 Loading custom YOLO model from: {self.model_path}")
                model = YOLO(self.model_path)
            else:
                print("📥 Loading pretrained YOLOv8n model (will download if not cached)")
                model = YOLO('yolov8n.pt')  # Downloads automatically if not present
                self.model_path = 'yolov8n.pt'

            return model

        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise

    def detect_documents(self, image_path, confidence_threshold=0.5, save_crops=True):
        """
        Detect documents in an image.

        Args:
            image_path (str): Path to input image
            confidence_threshold (float): Minimum confidence for detection
            save_crops (bool): Whether to save cropped documents

        Returns:
            dict: Detection results with bounding boxes and cropped images
        """
        image_path = Path(image_path)
        image_name = image_path.stem

        print(f"🔍 Detecting documents in: {image_path.name}")

        try:
            # Run YOLO detection
            results = self.model(str(image_path), conf=confidence_threshold, verbose=False)

            # Process results
            detections = []
            crops = []

            # Load original image for cropping
            original_image = cv2.imread(str(image_path))
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            for i, result in enumerate(results):
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    for j, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        # Convert to x, y, width, height format
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                        detection_info = {
                            'bbox': [x, y, w, h],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.model.names[class_id] if hasattr(self.model, 'names') else 'object'
                        }
                        detections.append(detection_info)

                        # Crop document if requested
                        if save_crops:
                            crop_info = self._save_crop(
                                original_image_rgb,
                                [x, y, w, h],
                                image_name,
                                len(crops)
                            )
                            crops.append(crop_info)

            # Save annotated image
            annotated_path = None
            if detections:
                annotated_path = self._save_annotated_image(
                    original_image_rgb, detections, image_name
                )

            detection_results = {
                'image_path': str(image_path),
                'detections': detections,
                'crops': crops,
                'annotated_image_path': annotated_path,
                'total_detections': len(detections)
            }

            print(f"✅ Found {len(detections)} document(s) with confidence >= {confidence_threshold}")

            return detection_results

        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return {
                'image_path': str(image_path),
                'detections': [],
                'crops': [],
                'annotated_image_path': None,
                'total_detections': 0,
                'error': str(e)
            }

    def _save_crop(self, image, bbox, image_name, crop_id):
        """Save cropped document."""
        x, y, w, h = bbox

        # Add padding around the crop
        padding = 10
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        x2_pad = min(image.shape[1], x + w + padding)
        y2_pad = min(image.shape[0], y + h + padding)

        # Crop the image
        cropped = image[y_pad:y2_pad, x_pad:x2_pad]

        # Save crop
        crop_filename = f"{image_name}crop{crop_id}.jpg"
        crop_path = self.results_dir / crop_filename

        # Convert RGB to BGR for OpenCV saving
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(crop_path), cropped_bgr)

        crop_info = {
            'crop_id': crop_id,
            'crop_path': str(crop_path),
            'crop_bbox': [x_pad, y_pad, x2_pad - x_pad, y2_pad - y_pad],
            'original_bbox': bbox
        }

        print(f"  💾 Saved crop {crop_id}: {crop_path}")
        return crop_info

    def _save_annotated_image(self, image, detections, image_name):
        """Save image with detection annotations."""
        annotated = image.copy()

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection.get('class_name', 'document')

            x, y, w, h = bbox

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Add label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Background rectangle for text
            cv2.rectangle(annotated,
                          (x, y - label_size[1] - 10),
                          (x + label_size[0], y),
                          (0, 255, 0), -1)

            # Text
            cv2.putText(annotated, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Save annotated image
        annotated_filename = f"{image_name}_yolo_detection.jpg"
        annotated_path = self.results_dir / annotated_filename

        # Convert RGB to BGR for saving
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(annotated_path), annotated_bgr)

        print(f"  🎨 Saved annotated image: {annotated_path}")
        return str(annotated_path)

    def detect_batch(self, images_dir, confidence_threshold=0.5, save_crops=True):
        """
        Detect documents in multiple images.

        Args:
            images_dir (str): Directory containing images
            confidence_threshold (float): Minimum confidence for detection
            save_crops (bool): Whether to save cropped documents

        Returns:
            dict: Batch detection results
        """
        images_dir = Path(images_dir)

        if not images_dir.exists():
            raise FileNotFoundError(f"Directory not found: {images_dir}")

        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = [f for f in images_dir.iterdir()
                       if f.suffix.lower() in image_extensions]

        print(f"📁 Processing {len(image_files)} images from {images_dir}")

        batch_results = {}
        total_detections = 0

        for image_file in image_files:
            print(f"\n📸 Processing: {image_file.name}")

            try:
                result = self.detect_documents(
                    image_file, confidence_threshold, save_crops
                )
                batch_results[image_file.name] = result
                total_detections += result['total_detections']

            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
                batch_results[image_file.name] = {'error': str(e)}

        print(f"\n🎉 Batch detection completed!")
        print(f"📊 Total detections across all images: {total_detections}")

        return {
            'images_directory': str(images_dir),
            'total_images': len(image_files),
            'total_detections': total_detections,
            'results': batch_results
        }

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'model_type': 'YOLOv8',
            'classes': getattr(self.model, 'names', {}),
            'device': str(next(self.model.model.parameters()).device) if hasattr(self.model, 'model') else 'unknown'
        }