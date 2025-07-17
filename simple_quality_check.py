#!/usr/bin/env python3
"""
Simple Quality Check - Demonstrates core annotation quality metrics
"""

import json

def calculate_simple_kappa(anno1, anno2):
    """Calculate Cohen's Kappa between two annotators."""
    # Count agreements and disagreements
    agreements = sum(1 for a, b in zip(anno1, anno2) if a == b)
    total = len(anno1)
    
    # Observed agreement
    po = agreements / total
    
    # Expected agreement (by chance)
    labels = list(set(anno1 + anno2))
    pe = 0
    for label in labels:
        p1 = anno1.count(label) / len(anno1)
        p2 = anno2.count(label) / len(anno2)
        pe += p1 * p2
    
    # Cohen's Kappa
    if pe == 1:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    return kappa

def analyze_annotations():
    """Analyze annotation quality with simple metrics."""
    print("🔍 Annotation Quality Analysis")
    print("=" * 40)
    
    # Load data
    with open('datasets/sentiment/sample_annotations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract annotations from all annotators
    annotator_data = {}
    for sample in data['annotation_samples']:
        for annotator_id, annotation in sample['annotations'].items():
            if annotator_id not in annotator_data:
                annotator_data[annotator_id] = []
            annotator_data[annotator_id].append(annotation['primary_label'])
    
    annotators = list(annotator_data.keys())
    
    # Calculate pairwise agreements
    print("📊 Pairwise Agreement (Cohen's Kappa):")
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            anno1 = annotator_data[annotators[i]]
            anno2 = annotator_data[annotators[j]]
            kappa = calculate_simple_kappa(anno1, anno2)
            print(f"  {annotators[i]} vs {annotators[j]}: κ = {kappa:.3f}")
    
    # Calculate overall statistics
    total_comparisons = 0
    total_agreements = 0
    
    for sample in data['annotation_samples']:
        labels = [anno['primary_label'] for anno in sample['annotations'].values()]
        n_annotators = len(labels)
        
        # Count pairwise agreements
        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                total_comparisons += 1
                if labels[i] == labels[j]:
                    total_agreements += 1
    
    percentage_agreement = total_agreements / total_comparisons
    print(f"\n📈 Overall Statistics:")
    print(f"  Percentage Agreement: {percentage_agreement:.1%}")
    print(f"  Total Comparisons: {total_comparisons}")
    print(f"  Agreements: {total_agreements}")
    
    # Quality assessment
    print(f"\n🎯 Quality Assessment:")
    if percentage_agreement > 0.9:
        print("  ✅ Excellent agreement (>90%)")
    elif percentage_agreement > 0.8:
        print("  ✅ Good agreement (80-90%)")
    elif percentage_agreement > 0.7:
        print("  ⚠️  Moderate agreement (70-80%)")
    else:
        print("  ❌ Poor agreement (<70%)")
    
    # Confidence analysis
    confidences = []
    for sample in data['annotation_samples']:
        for annotation in sample['annotations'].values():
            confidences.append(annotation['confidence'])
    
    avg_confidence = sum(confidences) / len(confidences)
    print(f"\n📊 Confidence Analysis:")
    print(f"  Average Confidence: {avg_confidence:.3f}")
    print(f"  Min Confidence: {min(confidences):.3f}")
    print(f"  Max Confidence: {max(confidences):.3f}")
    
    # Find challenging samples
    challenging_samples = []
    for sample in data['annotation_samples']:
        labels = [anno['primary_label'] for anno in sample['annotations'].values()]
        confidences = [anno['confidence'] for anno in sample['annotations'].values()]
        
        if len(set(labels)) > 1 or min(confidences) < 0.7:
            challenging_samples.append({
                'text': sample['text'][:60] + '...',
                'labels': labels,
                'avg_confidence': sum(confidences) / len(confidences)
            })
    
    if challenging_samples:
        print(f"\n🤔 Challenging Samples ({len(challenging_samples)}):")
        for sample in challenging_samples:
            print(f"  Text: {sample['text']}")
            print(f"  Labels: {sample['labels']}")
            print(f"  Avg Confidence: {sample['avg_confidence']:.3f}")
            print()

if __name__ == "__main__":
    analyze_annotations()