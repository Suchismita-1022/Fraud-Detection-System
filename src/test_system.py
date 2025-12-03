import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom imports
from yolo_detector import YOLODocumentDetector
from resnet_classifier import ResNetClassifier
# from pathlib import Path
#
# print(Path("ResNet/resnet18_document_classifier_20250920_134851.pth").exists())
from utils import create_visualization, generate_report, save_results

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
TEST_IMAGES_DIR =  BASE_DIR / "data" / "test_images" / "single_tests" / "document1.jpg"
image_path = BASE_DIR / "data" / "test_images" / "single_tests" / "document1.jpg"

# Create necessary directories
for dir_path in [RESULTS_DIR,
                 RESULTS_DIR / "detections",
                 RESULTS_DIR / "classifications",
                 RESULTS_DIR / "combined_results",
                 RESULTS_DIR / "reports"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Define model paths here (outside loop)
yolo_model_path = MODELS_DIR / "YOLOv8" / "yolov8_runs" / "weights" / "best.pt"
resnet_model_path = MODELS_DIR / "ResNet" / "resnet18_document_classifier_20250920_134851.pth"


class DocumentFraudDetectionSystem:
    """Complete system combining YOLO detection and ResNet classification."""

    def __init__(self, yolo_model_path: str, resnet_model_path: str, device=None):
        """Initialize the detection system."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing system on {self.device}")
        # Initialize YOLO detector
        self.yolo_detector = YOLODocumentDetector(
            str(yolo_model_path))  # Make sure YOLODocumentDetector accepts model path
        # Initialize ResNet classifier
        self.resnet_classifier = ResNetClassifier(str(resnet_model_path), self.device)
        print("✅ System initialized successfully!")

    def process_single_image(self, image_path:str, confidence_threshold=0.5, save_results=True, visualize=True):
        """Process a single image through the complete pipeline."""
        print(f"\n📸 Processing: {image_path}")
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        start_time = time.time()
        # Step 1: YOLO Detection
        print("🔍 Step 1: Document Detection (YOLO)")
        detection_results = self.yolo_detector.detect_documents(
            str(image_path), confidence_threshold
        )
        if not detection_results['detections']:
            print("⚠  No documents detected in the image")
            return {
                'image_path': str(image_path),
                'yolo_results': {'detections': []},
                'classification_results': [],
                'summary': 'No documents detected',
                'processing_time': time.time() - start_time
            }
        print(f"✅ Found {len(detection_results['detections'])} document(s)")
        # Step 2: ResNet Classification
        classification_results = []
        for i, crop_info in enumerate(detection_results['crops']):
            crop_path = crop_info['crop_path']
            print(f"   Classifying crop {i + 1}/{len(detection_results['crops'])}")
            # Classify the cropped document
            classification = self.resnet_classifier.classify_document(crop_path)
            classification_results.append({
                'crop_id': i,
                'crop_path': crop_path,
                **classification
            })
        # Step 3: Combine Results
        combined_results = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time,
            'yolo_results': detection_results,
            'classification_results': classification_results,
            'summary': self._generate_summary(classification_results)
        }
        # Step 4: Save Results and Visualize
        if save_results or visualize:
            self._save_and_visualize_results(image_path, combined_results, save_results, visualize)
        return combined_results

    def process_batch(self, images_dir, confidence_threshold=0.5, save_results=True):
        """Process multiple images in a directory."""
        images_dir = Path(images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Directory not found: {images_dir}")
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = [f for f in images_dir.iterdir()
                       if f.suffix.lower() in image_extensions]
        if not image_files:
            print(f"⚠  No images found in {images_dir}")
            return {}
        print(f"📁 Processing {len(image_files)} images from {images_dir}")
        batch_results = {}
        batch_stats = {
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'authentic_documents': 0,
            'forged_documents': 0,
            'no_documents_found': 0
        }
        for i, image_file in enumerate(image_files, 1):
            print(f"\n{'=' * 50}")
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            try:
                result = self.process_single_image(
                    image_file, confidence_threshold, save_results, visualize=False
                )
                batch_results[image_file.name] = result
                batch_stats['processed'] += 1
                # Update statistics
                if not result['classification_results']:
                    batch_stats['no_documents_found'] += 1
                else:
                    for classification in result['classification_results']:
                        if classification['predicted_class'] == 'authentic':
                            batch_stats['authentic_documents'] += 1
                        else:
                            batch_stats['forged_documents'] += 1
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
                batch_results[image_file.name] = {'error': str(e)}
                batch_stats['failed'] += 1
        # Generate batch report
        batch_summary = {
            'batch_directory': str(images_dir),
            'timestamp': datetime.now().isoformat(),
            'statistics': batch_stats,
            'results': batch_results
        }
        if save_results:
            self._save_batch_report(batch_summary)
        return batch_summary

    def _generate_summary(self, classification_results):
        """Generate a summary of classification results."""
        if not classification_results:
            return "No documents detected"
        authentic_count = sum(1 for c in classification_results
                              if c['predicted_class'] == 'authentic')
        forged_count = len(classification_results) - authentic_count
        total = len(classification_results)
        return {
            'total_documents': total,
            'authentic_documents': authentic_count,
            'forged_documents': forged_count,
            'authenticity_rate': authentic_count / total if total > 0 else 0,
            'fraud_rate': forged_count / total if total > 0 else 0
        }

    def _save_and_visualize_results(self, image_path, results, save_results, visualize):
        """Save results and create visualizations."""
        image_name = Path(image_path).stem
        # Save JSON results
        if save_results:
            results_file = RESULTS_DIR / "combined_results" / f"{image_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {results_file}")
        # Create visualization
        if visualize:
            viz_path = RESULTS_DIR / "combined_results" / f"{image_name}_visualization.jpg"
            self._create_visualization(image_path, results, viz_path)
            print(f"🎨 Visualization saved to: {viz_path}")

    def _create_visualization(self, image_path, results, output_path):
        """Create a visualization combining detection and classification results."""
        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        # Draw bounding boxes and labels
        for i, detection in enumerate(results['yolo_results']['detections']):
            bbox = detection['bbox']
            x, y, w, h = bbox
            # Get corresponding classification
            classification = results['classification_results'][i] if i < len(
                results['classification_results']) else None
            # Color based on classification
            if classification:
                if classification['predicted_class'] == 'authentic':
                    color = 'green'
                    label = f"Authentic ({classification['confidence']:.2f})"
                else:
                    color = 'red'
                    label = f"Forged ({classification['confidence']:.2f})"
            else:
                color = 'blue'
                label = f"Document {i + 1}"
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            # Add label
            ax.text(x, y - 10, label, fontsize=12, color=color,
                    fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
                                                 facecolor='white', alpha=0.8))
        ax.set_title(f"Document Analysis Results\n{Path(image_path).name}",
                     fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_batch_report(self, batch_summary):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_report = RESULTS_DIR / "reports" / f"batch_report_{timestamp}.json"
        with open(json_report, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        html_report = RESULTS_DIR / "reports" / f"batch_report_{timestamp}.html"
        self._create_html_report(batch_summary, html_report)
        print(f"📊 Batch report saved to: {json_report}")
        print(f"🌐 HTML report saved to: {html_report}")

    def _create_html_report(self, batch_summary, output_path):
        stats = batch_summary['statistics']
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Fraud Detection - Batch Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }}
                .results {{ margin-top: 30px; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🕵 Document Fraud Detection System</h1>
                <h2>Batch Processing Report</h2>
                <p><strong>Directory:</strong> {batch_summary['batch_directory']}</p>
                <p><strong>Timestamp:</strong> {batch_summary['timestamp']}</p>
            </div>
            <div class="stats">
                <div class="stat-box">
                    <h3>📊 Total Images</h3>
                    <h2>{stats['total_images']}</h2>
                </div>
                <div class="stat-box">
                    <h3>✅ Processed</h3>
                    <h2 class="success">{stats['processed']}</h2>
                </div>
                <div class="stat-box">
                    <h3>❌ Failed</h3>
                    <h2 class="error">{stats['failed']}</h2>
                </div>
                <div class="stat-box">
                    <h3>📄 Authentic</h3>
                    <h2 class="success">{stats['authentic_documents']}</h2>
                </div>
                <div class="stat-box">
                    <h3>⚠ Forged</h3>
                    <h2 class="error">{stats['forged_documents']}</h2>
                </div>
            </div>
            <div class="results">
                <h3>📋 Detailed Results</h3>
                <ul>
        """
        for filename, result in batch_summary['results'].items():
            if 'error' in result:
                html_content += f'<li class="error"> {filename}: {result["error"]}</li>'
            elif not result.get('classification_results'):
                html_content += f'<li class="warning"> {filename}: No documents detected</li>'
            else:
                summary = result['summary']
                html_content += f'<li class="success"> {filename}: {summary["total_documents"]} documents - {summary["authentic_documents"]} authentic, {summary["forged_documents"]} forged</li>'
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        with open(output_path, 'w') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Document Fraud Detection System - YOLO + ResNet")
    parser.add_argument('--image', type=str, help='Path to single image for testing')
    parser.add_argument('--batch', type=str, help='Path to directory containing images for batch testing')
    parser.add_argument('--yolo-model', type=str, help='Path to YOLO model (default: auto-download YOLOv8n)')
    parser.add_argument('--resnet-model', type=str, help='Path to ResNet model (default: auto-detect latest)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for YOLO detection (default: 0.5)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--no-visualize', action='store_true', help='Do not create visualizations')
    args = parser.parse_args()
    if not args.image and not args.batch:
        print(" Please provide either --image or --batch argument")
        print("\nExamples:")
        print("  python test_system.py --image data/test_images/document1.jpg")
        print("  python test_system.py --batch data/test_images/batch_tests/")
        return
    try:
        # Use default models if no argument given
        yolo_path = args.yolo_model or str(yolo_model_path)
        resnet_path = args.resnet_model or str(resnet_model_path)

        system = DocumentFraudDetectionSystem(
            yolo_model_path=yolo_path,
            resnet_model_path=resnet_path
        )
        if args.image:
            result = system.process_single_image(
                args.image,
                confidence_threshold=args.confidence,
                save_results=not args.no_save,
                visualize=not args.no_visualize
            )
            print(f"\n Processing completed!")
            print(f" Processing time: {result['processing_time']:.2f} seconds")
            if result['classification_results']:
                summary = result['summary']
                print(f"Summary: {summary['total_documents']} documents detected")
                print(f"Authentic: {summary['authentic_documents']}")
                print(f"Forged: {summary['forged_documents']}")
        elif args.batch:
            result = system.process_batch(
                args.batch,
                confidence_threshold=args.confidence,
                save_results=not args.no_save
            )
            stats = result['statistics']
            print(f"\n🎉 Batch processing completed!")
            print(f"📊 Statistics:")
            print(f"   Total images: {stats['total_images']}")
            print(f"   Successfully processed: {stats['processed']}")
            print(f"   Failed: {stats['failed']}")
            print(f"   Documents found: {stats['authentic_documents'] + stats['forged_documents']}")
            print(f"   Authentic: {stats['authentic_documents']}")
            print(f"   Forged: {stats['forged_documents']}")
    except Exception as e:
        print(f" Error: {e}")
        raise


if __name__ == "__main__":
    main()
