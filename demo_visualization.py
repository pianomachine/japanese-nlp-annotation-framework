#!/usr/bin/env python3
"""
Interactive Demo for Japanese NLP Annotation Framework
Creates visualizations without external dependencies
"""

import json
import math

def create_ascii_bar_chart(data, title, width=50):
    """Create ASCII bar chart."""
    print(f"\n=== {title} ===")
    max_val = max(data.values())
    
    for label, value in data.items():
        bar_length = int((value / max_val) * width)
        bar = '█' * bar_length
        print(f"{label:>3s} |{bar:<{width}} {value:.3f}")

def create_confusion_matrix_ascii(cm, labels):
    """Create ASCII confusion matrix."""
    print("\n=== Confusion Matrix ===")
    print("       ", end="")
    for label in labels:
        print(f"{label:>6s}", end="")
    print()
    
    for i, true_label in enumerate(labels):
        print(f"{true_label:>6s} ", end="")
        for j, pred_label in enumerate(labels):
            print(f"{cm[i][j]:>6d}", end="")
        print()

def analyze_agreement_patterns(data):
    """Analyze annotation agreement patterns."""
    print("\n=== Agreement Analysis ===")
    
    total_samples = len(data['annotation_samples'])
    perfect_agreement = 0
    partial_agreement = 0
    no_agreement = 0
    
    confidence_by_agreement = {
        'perfect': [],
        'partial': [],
        'none': []
    }
    
    for sample in data['annotation_samples']:
        labels = [anno['primary_label'] for anno in sample['annotations'].values()]
        confidences = [anno['confidence'] for anno in sample['annotations'].values()]
        
        unique_labels = len(set(labels))
        avg_confidence = sum(confidences) / len(confidences)
        
        if unique_labels == 1:
            perfect_agreement += 1
            confidence_by_agreement['perfect'].append(avg_confidence)
        elif unique_labels == 2:
            partial_agreement += 1
            confidence_by_agreement['partial'].append(avg_confidence)
        else:
            no_agreement += 1
            confidence_by_agreement['none'].append(avg_confidence)
    
    print(f"Perfect Agreement: {perfect_agreement}/{total_samples} ({perfect_agreement/total_samples:.1%})")
    print(f"Partial Agreement: {partial_agreement}/{total_samples} ({partial_agreement/total_samples:.1%})")
    print(f"No Agreement: {no_agreement}/{total_samples} ({no_agreement/total_samples:.1%})")
    
    print(f"\nConfidence by Agreement Level:")
    for level, confidences in confidence_by_agreement.items():
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"  {level.capitalize()}: {avg_conf:.3f}")

def simulate_active_learning_demo():
    """Simulate active learning selection."""
    print("\n=== Active Learning Simulation ===")
    
    # Simulate uncertainty scores
    samples = [
        ("素晴らしい商品です！", 0.95, "POS"),
        ("普通の商品です。", 0.45, "NEU"),  # High uncertainty
        ("ちょっと微妙かも...", 0.52, "NEG"),  # High uncertainty
        ("最悪でした。", 0.98, "NEG"),
        ("まあまあでした。", 0.48, "NEU"),  # High uncertainty
    ]
    
    # Sort by uncertainty (low confidence = high uncertainty)
    samples.sort(key=lambda x: x[1])
    
    print("Active Learning would select these samples for annotation:")
    print("(Lower confidence = higher learning value)")
    print()
    
    for i, (text, confidence, true_label) in enumerate(samples[:3]):
        uncertainty = 1 - confidence
        print(f"{i+1}. Uncertainty: {uncertainty:.3f}")
        print(f"   Text: {text}")
        print(f"   True Label: {true_label}")
        print()

def calculate_model_performance():
    """Calculate and display model performance metrics."""
    print("\n=== Model Performance Simulation ===")
    
    # Simulate benchmark results
    models = {
        "BERT-Japanese": {"accuracy": 0.932, "f1": 0.928, "speed": 45.2},
        "RoBERTa-Japanese": {"accuracy": 0.924, "f1": 0.920, "speed": 38.7},
        "DeBERTa-Japanese": {"accuracy": 0.945, "f1": 0.941, "speed": 32.1},
        "ELECTRA-Japanese": {"accuracy": 0.918, "f1": 0.915, "speed": 67.3},
    }
    
    print("Model Comparison:")
    print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'Speed (s/s)':<12}")
    print("-" * 52)
    
    for model, metrics in models.items():
        print(f"{model:<20} {metrics['accuracy']:<10.3f} {metrics['f1']:<10.3f} {metrics['speed']:<12.1f}")
    
    # Find best model for each metric
    best_accuracy = max(models.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(models.items(), key=lambda x: x[1]['f1'])
    best_speed = max(models.items(), key=lambda x: x[1]['speed'])
    
    print(f"\nBest Performance:")
    print(f"  Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.3f})")
    print(f"  F1 Score: {best_f1[0]} ({best_f1[1]['f1']:.3f})")
    print(f"  Speed: {best_speed[0]} ({best_speed[1]['speed']:.1f} samples/sec)")

def main():
    """Run comprehensive demo."""
    print("🚀 Japanese NLP Annotation Framework - Interactive Demo")
    print("=" * 60)
    
    # Load data
    try:
        with open('datasets/sentiment/sample_annotations.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find sample data file.")
        return
    
    # 1. Dataset Overview
    print(f"\n📊 Dataset Overview:")
    print(f"  Total Samples: {len(data['annotation_samples'])}")
    print(f"  Annotators: {len(data['dataset_info']['annotators'])}")
    print(f"  Agreement (Kappa): {data['dataset_info']['inter_annotator_agreement']['cohens_kappa']:.3f}")
    
    # 2. Label Distribution
    label_dist = data['statistics']['label_distribution']
    create_ascii_bar_chart(label_dist, "Label Distribution")
    
    # 3. Agreement Analysis
    analyze_agreement_patterns(data)
    
    # 4. Sample Annotations
    print("\n📝 Sample Annotations:")
    for i, sample in enumerate(data['annotation_samples'][:2]):
        print(f"\nSample {i+1}:")
        print(f"  Text: {sample['text'][:60]}...")
        print(f"  Gold Label: {sample['gold_label']}")
        print(f"  Annotator Labels: {[anno['primary_label'] for anno in sample['annotations'].values()]}")
        confidences = [f"{anno['confidence']:.2f}" for anno in sample['annotations'].values()]
        print(f"  Confidences: {confidences}")
    
    # 5. Active Learning Demo
    simulate_active_learning_demo()
    
    # 6. Model Performance
    calculate_model_performance()
    
    # 7. Quality Insights
    print("\n🔍 Quality Insights:")
    avg_conf = data['statistics']['average_confidence']
    disagreement_rate = 1 - data['dataset_info']['inter_annotator_agreement']['percentage_agreement']
    
    print(f"  Average Confidence: {avg_conf:.3f}")
    print(f"  Disagreement Rate: {disagreement_rate:.1%}")
    
    if avg_conf > 0.85:
        print("  ✅ High confidence annotations - good quality!")
    elif avg_conf > 0.7:
        print("  ⚠️  Moderate confidence - review guidelines")
    else:
        print("  ❌ Low confidence - need better training")
    
    print("\n🎉 Demo Complete! This framework provides:")
    print("  • High-quality Japanese text annotations")
    print("  • Systematic quality evaluation")
    print("  • Active learning for efficiency")
    print("  • Comprehensive model benchmarking")

if __name__ == "__main__":
    main()