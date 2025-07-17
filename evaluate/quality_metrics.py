#!/usr/bin/env python3
"""
Japanese NLP Annotation Quality Metrics
Author: Ryo Yanagisawa
Last Updated: 2024-12-15

This module provides comprehensive quality metrics for evaluating
annotation consistency and model performance on Japanese text data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')


class AnnotationMetrics:
    """Calculate inter-annotator agreement metrics for Japanese text annotation."""
    
    def __init__(self, labels: List[str] = ['POS', 'NEG', 'NEU']):
        self.labels = labels
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
    def cohen_kappa(self, anno1: List[str], anno2: List[str]) -> float:
        """Calculate Cohen's Kappa between two annotators."""
        return cohen_kappa_score(anno1, anno2)
    
    def fleiss_kappa(self, annotations: np.ndarray) -> float:
        """
        Calculate Fleiss' Kappa for multiple annotators.
        
        Args:
            annotations: Shape (n_items, n_categories) - count matrix
        """
        n, k = annotations.shape
        n_annotators = annotations.sum(axis=1)[0]
        
        p_j = annotations.sum(axis=0) / (n * n_annotators)
        P_e_bar = (p_j ** 2).sum()
        
        P_i = (annotations ** 2).sum(axis=1) - n_annotators
        P_i = P_i / (n_annotators * (n_annotators - 1))
        P_bar = P_i.mean()
        
        if P_e_bar == 1:
            return 1.0
        
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
        return kappa
    
    def krippendorff_alpha(self, data: Dict[str, Dict[str, str]], 
                          metric: str = 'nominal') -> float:
        """
        Calculate Krippendorff's Alpha for reliability measurement.
        
        Args:
            data: {item_id: {annotator_id: label}}
            metric: Type of metric ('nominal', 'ordinal', 'interval')
        """
        # Convert to coincidence matrix
        values = sorted(set(label for item in data.values() 
                           for label in item.values()))
        value_to_idx = {v: i for i, v in enumerate(values)}
        
        coincidence_matrix = np.zeros((len(values), len(values)))
        
        for item_id, annotations in data.items():
            annotators = list(annotations.keys())
            if len(annotators) < 2:
                continue
                
            for i in range(len(annotators)):
                for j in range(i + 1, len(annotators)):
                    v1 = value_to_idx[annotations[annotators[i]]]
                    v2 = value_to_idx[annotations[annotators[j]]]
                    coincidence_matrix[v1, v2] += 1
                    if v1 != v2:
                        coincidence_matrix[v2, v1] += 1
        
        # Calculate observed and expected disagreement
        n_total = coincidence_matrix.sum()
        if n_total == 0:
            return 0.0
            
        po = np.diag(coincidence_matrix).sum() / n_total
        
        marginals = coincidence_matrix.sum(axis=0)
        pe = (marginals ** 2).sum() / (n_total ** 2)
        
        if pe == 1:
            return 1.0
            
        alpha = 1 - (1 - po) / (1 - pe)
        return alpha
    
    def calculate_all_metrics(self, annotation_data: Dict) -> Dict:
        """Calculate all agreement metrics from annotation data."""
        # Extract pairwise annotations
        results = {
            'pairwise_kappa': {},
            'fleiss_kappa': None,
            'krippendorff_alpha': None,
            'percentage_agreement': {},
            'confusion_matrices': {}
        }
        
        # Get annotator pairs
        sample = list(annotation_data.values())[0]
        annotators = list(sample['annotations'].keys())
        
        # Pairwise Cohen's Kappa
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                anno1_labels = []
                anno2_labels = []
                
                for item in annotation_data.values():
                    if annotators[i] in item['annotations'] and \
                       annotators[j] in item['annotations']:
                        anno1_labels.append(
                            item['annotations'][annotators[i]]['primary_label']
                        )
                        anno2_labels.append(
                            item['annotations'][annotators[j]]['primary_label']
                        )
                
                pair_key = f"{annotators[i]}_vs_{annotators[j]}"
                results['pairwise_kappa'][pair_key] = self.cohen_kappa(
                    anno1_labels, anno2_labels
                )
                results['percentage_agreement'][pair_key] = sum(
                    a == b for a, b in zip(anno1_labels, anno2_labels)
                ) / len(anno1_labels)
                
                # Confusion matrix
                cm = confusion_matrix(anno1_labels, anno2_labels, labels=self.labels)
                results['confusion_matrices'][pair_key] = cm
        
        # Krippendorff's Alpha
        kripp_data = {}
        for item_id, item in annotation_data.items():
            kripp_data[item_id] = {
                anno: data['primary_label'] 
                for anno, data in item['annotations'].items()
            }
        results['krippendorff_alpha'] = self.krippendorff_alpha(kripp_data)
        
        # Fleiss' Kappa (if applicable)
        # Create count matrix for Fleiss
        n_items = len(annotation_data)
        n_categories = len(self.labels)
        count_matrix = np.zeros((n_items, n_categories))
        
        for idx, item in enumerate(annotation_data.values()):
            for anno_data in item['annotations'].values():
                label_idx = self.label_to_idx[anno_data['primary_label']]
                count_matrix[idx, label_idx] += 1
        
        if count_matrix.sum(axis=1)[0] == count_matrix.sum(axis=1)[-1]:
            results['fleiss_kappa'] = self.fleiss_kappa(count_matrix)
        
        return results


class QualityAnalyzer:
    """Analyze annotation quality and identify problematic patterns."""
    
    def __init__(self):
        self.metrics_calculator = AnnotationMetrics()
        
    def analyze_disagreements(self, annotation_data: Dict) -> pd.DataFrame:
        """Identify and analyze disagreement patterns."""
        disagreements = []
        
        for item_id, item in annotation_data.items():
            labels = [anno['primary_label'] 
                     for anno in item['annotations'].values()]
            
            if len(set(labels)) > 1:  # Disagreement exists
                disagreements.append({
                    'text_id': item_id,
                    'text': item['text'][:100] + '...' if len(item['text']) > 100 else item['text'],
                    'labels': labels,
                    'unique_labels': len(set(labels)),
                    'majority_label': max(set(labels), key=labels.count),
                    'annotator_confidence': [
                        anno['confidence'] 
                        for anno in item['annotations'].values()
                    ]
                })
        
        df = pd.DataFrame(disagreements)
        if not df.empty:
            df['avg_confidence'] = df['annotator_confidence'].apply(np.mean)
            df['confidence_std'] = df['annotator_confidence'].apply(np.std)
        
        return df
    
    def calculate_annotator_performance(self, annotation_data: Dict, 
                                      gold_standard: Optional[Dict] = None) -> pd.DataFrame:
        """Calculate individual annotator performance metrics."""
        annotator_stats = defaultdict(lambda: {
            'total_annotations': 0,
            'avg_confidence': 0,
            'agreement_with_majority': 0,
            'avg_time_seconds': 0,
            'label_distribution': defaultdict(int)
        })
        
        for item in annotation_data.values():
            # Get majority label
            labels = [anno['primary_label'] 
                     for anno in item['annotations'].values()]
            majority_label = max(set(labels), key=labels.count)
            
            for annotator_id, anno_data in item['annotations'].items():
                stats = annotator_stats[annotator_id]
                stats['total_annotations'] += 1
                stats['avg_confidence'] += anno_data['confidence']
                stats['label_distribution'][anno_data['primary_label']] += 1
                
                if anno_data['primary_label'] == majority_label:
                    stats['agreement_with_majority'] += 1
        
        # Calculate averages
        results = []
        for annotator_id, stats in annotator_stats.items():
            n = stats['total_annotations']
            results.append({
                'annotator_id': annotator_id,
                'total_annotations': n,
                'avg_confidence': stats['avg_confidence'] / n,
                'agreement_rate': stats['agreement_with_majority'] / n,
                'pos_ratio': stats['label_distribution']['POS'] / n,
                'neg_ratio': stats['label_distribution']['NEG'] / n,
                'neu_ratio': stats['label_distribution']['NEU'] / n
            })
        
        return pd.DataFrame(results)
    
    def generate_quality_report(self, annotation_data: Dict, 
                               output_path: str = 'quality_report.json') -> Dict:
        """Generate comprehensive quality report."""
        report = {
            'summary': {},
            'agreement_metrics': {},
            'disagreement_analysis': {},
            'annotator_performance': {},
            'recommendations': []
        }
        
        # Calculate agreement metrics
        metrics = self.metrics_calculator.calculate_all_metrics(annotation_data)
        report['agreement_metrics'] = {
            'krippendorff_alpha': float(metrics['krippendorff_alpha']),
            'avg_pairwise_kappa': float(np.mean(list(metrics['pairwise_kappa'].values()))),
            'avg_percentage_agreement': float(np.mean(list(metrics['percentage_agreement'].values())))
        }
        
        # Analyze disagreements
        disagreements_df = self.analyze_disagreements(annotation_data)
        if not disagreements_df.empty:
            report['disagreement_analysis'] = {
                'total_disagreements': len(disagreements_df),
                'disagreement_rate': len(disagreements_df) / len(annotation_data),
                'avg_confidence_on_disagreements': float(disagreements_df['avg_confidence'].mean()),
                'most_problematic_samples': disagreements_df.nsmallest(
                    5, 'avg_confidence'
                )[['text_id', 'text', 'labels']].to_dict('records')
            }
        
        # Annotator performance
        performance_df = self.calculate_annotator_performance(annotation_data)
        report['annotator_performance'] = performance_df.to_dict('records')
        
        # Generate recommendations
        alpha = report['agreement_metrics']['krippendorff_alpha']
        if alpha < 0.667:
            report['recommendations'].append(
                "CRITICAL: Agreement is below acceptable threshold. "
                "Conduct retraining session focusing on guideline clarification."
            )
        elif alpha < 0.8:
            report['recommendations'].append(
                "WARNING: Agreement is moderate. Review disagreement cases "
                "in team meeting and update guidelines."
            )
        else:
            report['recommendations'].append(
                "SUCCESS: Agreement meets quality standards. "
                "Continue current annotation practices."
            )
        
        # Check for annotator bias
        for anno in report['annotator_performance']:
            if abs(anno['pos_ratio'] - 0.33) > 0.15:
                report['recommendations'].append(
                    f"Check {anno['annotator_id']} for potential label bias "
                    f"(POS ratio: {anno['pos_ratio']:.2f})"
                )
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def visualize_agreement_matrix(self, annotation_data: Dict, 
                                   save_path: str = 'agreement_heatmap.png'):
        """Create visualization of annotator agreement patterns."""
        metrics = self.metrics_calculator.calculate_all_metrics(annotation_data)
        
        # Create agreement matrix
        annotators = list(list(annotation_data.values())[0]['annotations'].keys())
        n = len(annotators)
        agreement_matrix = np.ones((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                pair_key = f"{annotators[i]}_vs_{annotators[j]}"
                if pair_key in metrics['pairwise_kappa']:
                    agreement_matrix[i, j] = metrics['pairwise_kappa'][pair_key]
                    agreement_matrix[j, i] = agreement_matrix[i, j]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            agreement_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            xticklabels=annotators,
            yticklabels=annotators,
            cbar_kws={'label': "Cohen's Kappa"}
        )
        plt.title('Inter-Annotator Agreement Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return agreement_matrix


def main():
    """Example usage of quality metrics."""
    # Load sample annotation data
    with open('datasets/sentiment/sample_annotations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotation_samples = {
        sample['text_id']: sample 
        for sample in data['annotation_samples']
    }
    
    # Initialize analyzer
    analyzer = QualityAnalyzer()
    
    # Generate quality report
    print("Generating quality report...")
    report = analyzer.generate_quality_report(
        annotation_samples,
        'quality_report.json'
    )
    
    print(f"\nQuality Metrics Summary:")
    print(f"Krippendorff's Alpha: {report['agreement_metrics']['krippendorff_alpha']:.3f}")
    print(f"Average Pairwise Kappa: {report['agreement_metrics']['avg_pairwise_kappa']:.3f}")
    print(f"Disagreement Rate: {report['disagreement_analysis']['disagreement_rate']:.2%}")
    
    # Visualize agreement
    print("\nGenerating agreement visualization...")
    analyzer.visualize_agreement_matrix(
        annotation_samples,
        'agreement_heatmap.png'
    )
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    print("\nQuality report saved to 'quality_report.json'")
    print("Agreement heatmap saved to 'agreement_heatmap.png'")


if __name__ == "__main__":
    main()