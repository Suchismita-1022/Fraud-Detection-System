import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import json
import glob


class ResNetClassifier:
    """ResNet-based document authenticity classifier."""

    def __init__(self, model_path=None, device=None):
        """
        Initialize ResNet classifier.

        Args:
            model_path (str): Path to trained ResNet model. If None, auto-detects latest.
            device (str): Device to use ('cuda' or 'cpu'). If None, auto-detects.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or self._find_latest_model()
        self.class_names = ['authentic', 'forged']  # Default class names
        self.class_to_idx = {'authentic': 0, 'forged': 1}

        # Initialize model and transforms
        self.model = self._load_model()
        self.transform = self._get_transform()

        print(f" ResNet classifier initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.class_names}")

    def _find_latest_model(self):
        """Find the latest trained ResNet model."""
        models_dir = Path("ResNet/resnet18_document_classifier_20250920_134851.pth")

        if not models_dir.exists():
            raise FileNotFoundError(f"ResNet models directory not found: {models_dir}")

        # Look for ResNet model files
        model_patterns = [
            "resnet18_document_classifier_*.pth",
            "resnet_document_classifier_*.pth",
            "*.pth"
        ]

        model_files = []
        for pattern in model_patterns:
            model_files.extend(glob.glob(str(models_dir / pattern)))

        if not model_files:
            raise FileNotFoundError(f"No ResNet model files found in {models_dir}")

        # Get the most recent model
        latest_model = max(model_files, key=os.path.getctime)
        print(f"📥 Auto-detected latest model: {latest_model}")

        return latest_model

    def _load_model(self):
        """Load the trained ResNet model."""
        try:
            print(f"📥 Loading ResNet model from: {self.model_path}")

            # Create ResNet18 architecture
            model = models.resnet18(pretrained=False)
            num_classes = len(self.class_names)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()

            print(f"✅ Model loaded successfully")
            return model

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _get_transform(self):
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def classify_document(self, image_path, return_probabilities=True):
        """
        Classify a document image as authentic or forged.

        Args:
            image_path (str): Path to document image
            return_probabilities (bool): Whether to return class probabilities

        Returns:
            dict: Classification results
        """
        try:
            image_path = Path(image_path)
            print(f"🧠 Classifying: {image_path.name}")

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()

            # Get predicted class name
            predicted_class = self.class_names[predicted_class_idx]

            # Prepare results
            result = {
                'image_path': str(image_path),
                'original_size': original_size,
                'predicted_class': predicted_class,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence
            }

            if return_probabilities:
                class_probabilities = {}
                for i, class_name in enumerate(self.class_names):
                    class_probabilities[class_name] = probabilities[0][i].item()
                result['class_probabilities'] = class_probabilities

            # Add risk assessment
            result['risk_level'] = self._assess_risk(predicted_class, confidence)

            print(f"  ✅ Prediction: {predicted_class} (confidence: {confidence:.3f})")

            return result

        except Exception as e:
            print(f"❌ Error classifying {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'error': str(e),
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'risk_level': 'error'
            }

    def classify_batch(self, image_paths, return_probabilities=True):
        """
        Classify multiple document images.

        Args:
            image_paths (list): List of image paths
            return_probabilities (bool): Whether to return class probabilities

        Returns:
            dict: Batch classification results
        """
        print(f"📁 Classifying {len(image_paths)} documents")

        results = {}
        stats = {
            'total': len(image_paths),
            'authentic': 0,
            'forged': 0,
            'errors': 0,
            'high_confidence': 0,
            'low_confidence': 0
        }

        for image_path in image_paths:
            result = self.classify_document(image_path, return_probabilities)

            filename = Path(image_path).name
            results[filename] = result

            # Update statistics
            if 'error' in result:
                stats['errors'] += 1
            else:
                predicted_class = result['predicted_class']
                confidence = result['confidence']

                if predicted_class == 'authentic':
                    stats['authentic'] += 1
                elif predicted_class == 'forged':
                    stats['forged'] += 1

                if confidence >= 0.8:
                    stats['high_confidence'] += 1
                elif confidence < 0.6:
                    stats['low_confidence'] += 1

        return {
            'results': results,
            'statistics': stats,
            'summary': self._generate_batch_summary(stats)
        }

    def _assess_risk(self, predicted_class, confidence):
        """Assess risk level based on prediction and confidence."""
        if predicted_class == 'forged':
            if confidence >= 0.9:
                return 'very_high'
            elif confidence >= 0.7:
                return 'high'
            elif confidence >= 0.6:
                return 'medium'
            else:
                return 'low'
        else:  # authentic
            if confidence >= 0.9:
                return 'very_low'
            elif confidence >= 0.7:
                return 'low'
            elif confidence >= 0.6:
                return 'medium'
            else:
                return 'high'  # Low confidence in authenticity = higher risk

    def _generate_batch_summary(self, stats):
        """Generate summary statistics for batch processing."""
        total = stats['total']
        if total == 0:
            return "No documents processed"

        authentic_rate = stats['authentic'] / total * 100
        forged_rate = stats['forged'] / total * 100
        error_rate = stats['errors'] / total * 100

        return {
            'authenticity_rate': authentic_rate,
            'fraud_rate': forged_rate,
            'error_rate': error_rate,
            'high_confidence_predictions': stats['high_confidence'],
            'low_confidence_predictions': stats['low_confidence']
        }

    def save_classification_results(self, results, output_path):
        """Save classification results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        results_with_metadata = {
            'metadata': {
                'model_path': self.model_path,
                'device': self.device,
                'class_names': self.class_names,
                'timestamp': torch.cuda.Event.now().isoformat() if hasattr(torch.cuda.Event, 'now') else 'unknown'
            },
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)

        print(f"💾 Classification results saved to: {output_path}")

    def get_model_info(self):
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_path': self.model_path,
            'architecture': 'ResNet18',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'device': self.device,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Approximate size in MB
        }

    def explain_prediction(self, result):
        """Generate human-readable explanation of prediction."""
        if 'error' in result:
            return f"❌ Error occurred during classification: {result['error']}"

        predicted_class = result['predicted_class']
        confidence = result['confidence']
        risk_level = result['risk_level']

        # Base explanation
        if predicted_class == 'authentic':
            explanation = f"✅ Document appears to be AUTHENTIC (confidence: {confidence:.1%})"
        else:
            explanation = f"⚠ Document appears to be FORGED (confidence: {confidence:.1%})"

        # Add risk assessment
        risk_descriptions = {
            'very_low': 'Very low risk of fraud',
            'low': 'Low risk of fraud',
            'medium': 'Medium risk - manual review recommended',
            'high': 'High risk - requires careful inspection',
            'very_high': 'Very high risk - likely fraudulent'
        }

        explanation += f"\n🎯 Risk Level: {risk_descriptions.get(risk_level, 'Unknown')}"

        # Add confidence interpretation
        if confidence >= 0.9:
            explanation += "\n🔒 Very high confidence in prediction"
        elif confidence >= 0.7:
            explanation += "\n✅ High confidence in prediction"
        elif confidence >= 0.6:
            explanation += "\n⚖ Moderate confidence - consider additional verification"
        else:
            explanation += "\n⚠ Low confidence - human review strongly recommended"

        return explanation