import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
from datetime import datetime
import seaborn as sns
from typing import Dict, List, Tuple, Optional


def create_visualization(image_path: str, yolo_results: Dict, classification_results: List[Dict],
                         output_path: str, show_confidence: bool = True):
    """
    Create a comprehensive visualization combining YOLO detection and ResNet classification.

    Args:
        image_path: Path to original image
        yolo_results: YOLO detection results
        classification_results: ResNet classification results
        output_path: Path to save visualization
        show_confidence: Whether to show confidence scores
    """
    try:
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image_rgb)

        # Color scheme
        colors = {
            'authentic': '#2ECC71',  # Green
            'forged': '#E74C3C',  # Red
            'unknown': '#3498DB'  # Blue
        }

        # Draw bounding boxes and classifications
        for i, detection in enumerate(yolo_results.get('detections', [])):
            bbox = detection['bbox']
            x, y, w, h = bbox

            # Get corresponding classification
            classification = None
            if i < len(classification_results):
                classification = classification_results[i]

            # Determine color and label
            if classification and 'predicted_class' in classification:
                predicted_class = classification['predicted_class']
                confidence = classification.get('confidence', 0.0)
                color = colors.get(predicted_class, colors['unknown'])

                if show_confidence:
                    label = f"{predicted_class.title()}\n({confidence:.2%})"
                else:
                    label = predicted_class.title()

                # Add risk level
                risk_level = classification.get('risk_level', 'unknown')
                if risk_level in ['high', 'very_high']:
                    label += "\n⚠ HIGH RISK"
                elif risk_level in ['very_low', 'low']:
                    label += "\n✅ LOW RISK"
            else:
                color = colors['unknown']
                label = "Document"

            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=4,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Add label with background
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white',
                              edgecolor=color, linewidth=2, alpha=0.9)
            ax.text(x, y - 10, label, fontsize=12, color=color,
                    fontweight='bold', bbox=bbox_props, verticalalignment='top')

        # Set title
        total_docs = len(yolo_results.get('detections', []))
        authentic_count = sum(1 for c in classification_results
                              if c.get('predicted_class') == 'authentic')
        forged_count = sum(1 for c in classification_results
                           if c.get('predicted_class') == 'forged')

        title = f"Document Fraud Detection Results\n"
        title += f"Image: {Path(image_path).name} | "
        title += f"Documents Found: {total_docs} | "
        title += f"Authentic: {authentic_count} | Forged: {forged_count}"

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        # Add legend
        legend_elements = []
        if authentic_count > 0:
            legend_elements.append(patches.Patch(color=colors['authentic'], label='Authentic'))
        if forged_count > 0:
            legend_elements.append(patches.Patch(color=colors['forged'], label='Forged'))

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        print(f"🎨 Visualization saved: {output_path}")

    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        raise


def generate_report(results: Dict, output_path: str, format_type: str = 'json'):
    """
    Generate comprehensive report from testing results.

    Args:
        results: Combined results from YOLO + ResNet pipeline
        output_path: Path to save report
        format_type: Report format ('json', 'html', 'txt')
    """
    output_path = Path(output_path)

    if format_type == 'json':
        _generate_json_report(results, output_path)
    elif format_type == 'html':
        _generate_html_report(results, output_path)
    elif format_type == 'txt':
        _generate_text_report(results, output_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def _generate_json_report(results: Dict, output_path: Path):
    """Generate JSON report."""
    report_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'document_fraud_detection',
            'version': '1.0'
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"📄 JSON report saved: {output_path}")


def _generate_html_report(results: Dict, output_path: Path):
    """Generate HTML report."""

    # Extract summary statistics
    if isinstance(results, dict) and 'classification_results' in results:
        # Single image results
        classifications = results['classification_results']
        total_docs = len(classifications)
        authentic = sum(1 for c in classifications if c.get('predicted_class') == 'authentic')
        forged = sum(1 for c in classifications if c.get('predicted_class') == 'forged')
        high_conf = sum(1 for c in classifications if c.get('confidence', 0) >= 0.8)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Fraud Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                         gap: 20px; margin-bottom: 30px; }}
                .stat-box {{ background: white; padding: 25px; border-radius: 10px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }}
                .stat-number {{ font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }}
                .authentic {{ color: #2ECC71; }}
                .forged {{ color: #E74C3C; }}
                .neutral {{ color: #3498DB; }}
                .results-section {{ background: white; padding: 30px; border-radius: 10px; 
                                   box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .document-item {{ border-left: 4px solid #ddd; padding: 15px; margin: 10px 0; 
                                 background: #f8f9fa; border-radius: 0 5px 5px 0; }}
                .document-item.authentic {{ border-left-color: #2ECC71; }}
                .document-item.forged {{ border-left-color: #E74C3C; }}
                .confidence-bar {{ width: 100%; height: 10px; background: #ecf0f1; 
                                  border-radius: 5px; overflow: hidden; margin: 10px 0; }}
                .confidence-fill {{ height: 100%; background: #3498DB; }}
                .high-confidence {{ background: #2ECC71 !important; }}
                .medium-confidence {{ background: #F39C12 !important; }}
                .low-confidence {{ background: #E74C3C !important; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🕵 Document Fraud Detection Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Image: {Path(results.get('image_path', 'Unknown')).name}</p>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number neutral">{total_docs}</div>
                    <div>Total Documents</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number authentic">{authentic}</div>
                    <div>Authentic</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number forged">{forged}</div>
                    <div>Forged</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number neutral">{high_conf}</div>
                    <div>High Confidence</div>
                </div>
            </div>

            <div class="results-section">
                <h2>📋 Detailed Results</h2>"""

        for i, classification in enumerate(classifications):
            predicted_class = classification.get('predicted_class', 'unknown')
            confidence = classification.get('confidence', 0.0)
            risk_level = classification.get('risk_level', 'unknown')

            confidence_class = 'high-confidence' if confidence >= 0.8 else \
                'medium-confidence' if confidence >= 0.6 else 'low-confidence'

            html_content += f"""
                <div class="document-item {predicted_class}">
                    <h3>Document {i + 1}</h3>
                    <p><strong>Classification:</strong> {predicted_class.title()}</p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p><strong>Risk Level:</strong> {risk_level.replace('_', ' ').title()}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill {confidence_class}" 
                             style="width: {confidence * 100}%"></div>
                    </div>
                </div>"""

        html_content += """
            </div>
        </body>
        </html>
        """

    else:
        # Batch results - simplified version
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Document Fraud Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Batch Processing Report</h1>
                <p>Results from multiple document analysis</p>
            </div>
            <pre>{}</pre>
        </body>
        </html>
        """.format(json.dumps(results, indent=2))

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"🌐 HTML report saved: {output_path}")


def _generate_text_report(results: Dict, output_path: Path):
    """Generate text report."""

    report_lines = [
        "=" * 60,
        "📋 DOCUMENT FRAUD DETECTION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]

    if 'image_path' in results:
        report_lines.append(f"Image: {Path(results['image_path']).name}")
        report_lines.append("")

    if 'classification_results' in results:
        classifications = results['classification_results']

        # Summary
        total_docs = len(classifications)
        authentic = sum(1 for c in classifications if c.get('predicted_class') == 'authentic')
        forged = sum(1 for c in classifications if c.get('predicted_class') == 'forged')

        report_lines.extend([
            "📊 SUMMARY",
            "-" * 20,
            f"Total Documents: {total_docs}",
            f"Authentic: {authentic}",
            f"Forged: {forged}",
            f"Authenticity Rate: {authentic / total_docs * 100:.1f}%" if total_docs > 0 else "N/A",
            ""
        ])

        # Detailed results
        report_lines.extend([
            "📋 DETAILED RESULTS",
            "-" * 30
        ])

        for i, classification in enumerate(classifications):
            predicted_class = classification.get('predicted_class', 'unknown')
            confidence = classification.get('confidence', 0.0)
            risk_level = classification.get('risk_level', 'unknown')

            report_lines.extend([
                f"Document {i + 1}:",
                f"  Classification: {predicted_class.upper()}",
                f"  Confidence: {confidence:.1%}",
                f"  Risk Level: {risk_level.replace('_', ' ').title()}",
                ""
            ])

    report_text = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"📝 Text report saved: {output_path}")


def save_results(results: Dict, output_dir: str, filename_prefix: str = "results"):
    """
    Save results in multiple formats.

    Args:
        results: Results dictionary
        output_dir: Directory to save results
        filename_prefix: Prefix for result files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"{filename_prefix}_{timestamp}.json"
    generate_report(results, json_path, 'json')

    # Save HTML
    html_path = output_dir / f"{filename_prefix}_{timestamp}.html"
    generate_report(results, html_path, 'html')

    # Save text
    txt_path = output_dir / f"{filename_prefix}_{timestamp}.txt"
    generate_report(results, txt_path, 'txt')

    return {
        'json': str(json_path),
        'html': str(html_path),
        'text': str(txt_path)
    }


def create_confidence_plot(classification_results: List[Dict], output_path: str):
    """Create confidence distribution plot."""

    confidences = [c.get('confidence', 0) for c in classification_results]
    classes = [c.get('predicted_class', 'unknown') for c in classification_results]

    if not confidences:
        print("⚠ No confidence data to plot")
        return

    plt.figure(figsize=(12, 6))

    # Subplot 1: Confidence histogram
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(confidences), color='red', linestyle='--',
                label=f'Mean: {np.mean(confidences):.2f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Confidence by class
    plt.subplot(1, 2, 2)
    authentic_conf = [c for c, cl in zip(confidences, classes) if cl == 'authentic']
    forged_conf = [c for c, cl in zip(confidences, classes) if cl == 'forged']

    data_to_plot = []
    labels = []

    if authentic_conf:
        data_to_plot.append(authentic_conf)
        labels.append('Authentic')
    if forged_conf:
        data_to_plot.append(forged_conf)
        labels.append('Forged')

    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels)
        plt.ylabel('Confidence Score')
        plt.title('Confidence by Predicted Class')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Confidence plot saved: {output_path}")


def create_summary_dashboard(batch_results: Dict, output_path: str):
    """Create summary dashboard for batch results."""

    plt.figure(figsize=(16, 12))

    # Extract statistics
    if 'statistics' in batch_results:
        stats = batch_results['statistics']

        # Pie chart of document classifications
        plt.subplot(2, 3, 1)
        sizes = [stats.get('authentic_documents', 0), stats.get('forged_documents', 0)]
        labels = ['Authentic', 'Forged']
        colors = ['#2ECC71', '#E74C3C']

        if sum(sizes) > 0:
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Document Classification Distribution')

        # Bar chart of processing status
        plt.subplot(2, 3, 2)
        statuses = ['Processed', 'Failed', 'No Documents']
        counts = [stats.get('processed', 0), stats.get('failed', 0),
                  stats.get('no_documents_found', 0)]
        colors_bar = ['#2ECC71', '#E74C3C', '#F39C12']

        bars = plt.bar(statuses, counts, color=colors_bar)
        plt.title('Processing Status')
        plt.ylabel('Count')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                         f'{count}', ha='center', va='bottom')

    # Additional plots if individual results are available
    if 'results' in batch_results and isinstance(batch_results['results'], dict):
        all_classifications = []
        all_confidences = []

        for file_result in batch_results['results'].values():
            if isinstance(file_result, dict) and 'classification_results' in file_result:
                for classification in file_result['classification_results']:
                    all_classifications.append(classification.get('predicted_class', 'unknown'))
                    all_confidences.append(classification.get('confidence', 0.0))

        if all_confidences:
            # Confidence distribution
            plt.subplot(2, 3, 3)
            plt.hist(all_confidences, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
            plt.axvline(np.mean(all_confidences), color='red', linestyle='--',
                        label=f'Mean: {np.mean(all_confidences):.2f}')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Overall Confidence Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Risk level distribution
            plt.subplot(2, 3, 4)
            risk_levels = []
            for file_result in batch_results['results'].values():
                if isinstance(file_result, dict) and 'classification_results' in file_result:
                    for classification in file_result['classification_results']:
                        risk_levels.append(classification.get('risk_level', 'unknown'))

            if risk_levels:
                from collections import Counter
                risk_counts = Counter(risk_levels)
                risks, counts = zip(*risk_counts.items()) if risk_counts else ([], [])

                colors_risk = ['#27AE60', '#F39C12', '#E74C3C', '#8E44AD', '#2C3E50']
                plt.bar(risks, counts, color=colors_risk[:len(risks)])
                plt.title('Risk Level Distribution')
                plt.ylabel('Count')
                plt.xticks(rotation=45)

    plt.suptitle('Document Fraud Detection - Batch Summary Dashboard',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Summary dashboard saved: {output_path}")


def validate_image_path(image_path: str) -> bool:


    image_path = Path(image_path)

    if not image_path.exists():
        return False

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    return image_path.suffix.lower() in valid_extensions


def get_image_info(image_path: str) -> Dict:


    try:
        with Image.open(image_path) as img:
            return {
                'path': str(image_path),
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'file_size_mb': os.path.getsize(image_path) / (1024 * 1024)
            }
    except Exception as e:
        return {'path': str(image_path), 'error': str(e)}


def setup_directories(base_dir: str = "results"):


    base_path = Path(base_dir)
    directories = [
        base_path,
        base_path / "detections",
        base_path / "classifications",
        base_path / "combined_results",
        base_path / "reports",
        base_path / "visualizations"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Directory structure created in: {base_path}")
    return {str(d.name): str(d) for d in directories}