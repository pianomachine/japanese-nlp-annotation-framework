#!/usr/bin/env python3
"""
Show detailed annotation samples with Japanese text analysis
"""

import json

def show_detailed_samples():
    """Show detailed annotation samples with explanations."""
    print("🇯🇵 Japanese NLP Annotation Framework - Sample Analysis")
    print("=" * 70)
    
    with open('datasets/sentiment/sample_annotations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Dataset Overview:")
    print(f"  Total Samples: {len(data['annotation_samples'])}")
    print(f"  Annotators: {len(data['dataset_info']['annotators'])}")
    print(f"  Inter-annotator Agreement: {data['dataset_info']['inter_annotator_agreement']['cohens_kappa']:.3f}")
    
    # Show each sample with detailed analysis
    for i, sample in enumerate(data['annotation_samples']):
        print(f"\n{'='*60}")
        print(f"📝 Sample {i+1}: {sample['text_id']}")
        print(f"{'='*60}")
        
        # Show full text
        print(f"🔤 Original Text:")
        print(f"  {sample['text']}")
        
        # Show domain
        print(f"\n🏷️  Domain: {sample['domain']}")
        
        # Show gold label
        print(f"\n✅ Gold Standard Label: {sample['gold_label']}")
        
        # Show each annotator's judgment
        print(f"\n👥 Annotator Judgments:")
        labels = []
        confidences = []
        
        for annotator_id, annotation in sample['annotations'].items():
            labels.append(annotation['primary_label'])
            confidences.append(annotation['confidence'])
            
            print(f"  {annotator_id}:")
            print(f"    Primary Label: {annotation['primary_label']}")
            print(f"    Confidence: {annotation['confidence']:.3f}")
            if 'fine_grained' in annotation and annotation['fine_grained']:
                print(f"    Fine-grained: {annotation['fine_grained']}")
            if 'reasoning' in annotation:
                print(f"    Reasoning: {annotation['reasoning']}")
        
        # Agreement analysis
        unique_labels = len(set(labels))
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"\n📊 Agreement Analysis:")
        print(f"  Unique Labels: {unique_labels}")
        print(f"  Average Confidence: {avg_confidence:.3f}")
        
        if unique_labels == 1:
            print(f"  ✅ Perfect Agreement - All annotators agree")
        elif unique_labels == 2:
            print(f"  ⚠️  Partial Disagreement - 2 different labels")
        else:
            print(f"  ❌ Major Disagreement - {unique_labels} different labels")
        
        # Show notes if available
        if 'notes' in sample:
            print(f"\n📝 Notes: {sample['notes']}")
        
        # Japanese-specific analysis
        text = sample['text']
        print(f"\n🇯🇵 Japanese Language Analysis:")
        
        # Check for specific patterns
        if '！' in text or '!!' in text:
            print(f"  • Contains exclamation marks (often positive)")
        if '。。。' in text or '...' in text:
            print(f"  • Contains ellipsis (often indicates hesitation/negative)")
        if 'です' in text or 'ます' in text:
            print(f"  • Uses polite form (keigo)")
        if '(笑)' in text or '（笑）' in text:
            print(f"  • Contains (笑) - potential sarcasm marker")
        if 'ちょっと' in text:
            print(f"  • Contains 'ちょっと' - often softens criticism")
        if '普通' in text:
            print(f"  • Contains '普通' - neutral indicator")
        if '最悪' in text or '最高' in text:
            print(f"  • Contains extreme expression")
        
        # Confidence assessment
        if avg_confidence > 0.9:
            print(f"  🎯 High confidence sample - clear sentiment")
        elif avg_confidence > 0.7:
            print(f"  🤔 Moderate confidence - some ambiguity")
        else:
            print(f"  ❓ Low confidence - challenging case")
        
        print()

if __name__ == "__main__":
    show_detailed_samples()