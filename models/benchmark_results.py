#!/usr/bin/env python3
"""
Model Benchmarking for Japanese Sentiment Analysis
Compares multiple models on annotated datasets

Author: Ryo Yanagisawa
Last Updated: 2024-12-15
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ModelBenchmark:
    """Comprehensive benchmarking suite for Japanese NLP models."""
    
    def __init__(self, models_config: Dict[str, str]):
        """
        Initialize benchmark with model configurations.
        
        Args:
            models_config: {model_name: huggingface_model_id}
        """
        self.models_config = models_config
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_dataset(self, data_path: str) -> Tuple[List[str], List[str]]:
        """Load annotated dataset."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for sample in data['annotation_samples']:
            texts.append(sample['text'])
            labels.append(sample['gold_label'])
        
        return texts, labels
    
    def evaluate_model(self, 
                      model_name: str,
                      model_id: str,
                      texts: List[str],
                      true_labels: List[str]) -> Dict:
        """Evaluate a single model."""
        print(f"\nEvaluating {model_name}...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=3
        ).to(self.device)
        
        # Inference timing
        start_time = time.time()
        predictions = []
        confidence_scores = []
        
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                confidence_scores.extend(torch.max(probs, dim=-1).values.cpu().numpy())
        
        inference_time = time.time() - start_time
        
        # Convert predictions to labels
        label_map = {0: 'POS', 1: 'NEG', 2: 'NEU'}
        pred_labels = [label_map[p] for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(
            true_labels, pred_labels, 
            output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=['POS', 'NEG', 'NEU'])
        
        # Model size (approximate)
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / 1024 / 1024
        
        results = {
            'model_name': model_name,
            'model_id': model_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time_seconds': inference_time,
            'samples_per_second': len(texts) / inference_time,
            'avg_confidence': np.mean(confidence_scores),
            'model_size_mb': model_size_mb,
            'confusion_matrix': cm.tolist(),
            'class_metrics': class_report,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def run_benchmark(self, data_path: str) -> pd.DataFrame:
        """Run benchmark on all models."""
        texts, labels = self.load_dataset(data_path)
        print(f"Loaded {len(texts)} samples for evaluation")
        
        for model_name, model_id in self.models_config.items():
            try:
                self.results[model_name] = self.evaluate_model(
                    model_name, model_id, texts, labels
                )
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in self.results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': result['accuracy'],
                    'F1 Score': result['f1_score'],
                    'Inference Speed (samples/sec)': result['samples_per_second'],
                    'Model Size (MB)': result['model_size_mb'],
                    'Avg Confidence': result['avg_confidence']
                })
        
        return pd.DataFrame(comparison_data)
    
    def visualize_results(self, save_dir: str = 'benchmark_results/'):
        """Create visualizations of benchmark results."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Performance comparison
        df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': res['accuracy'],
                'F1 Score': res['f1_score'],
                'Speed': res['samples_per_second']
            }
            for name, res in self.results.items()
            if 'error' not in res
        ])
        
        # 1. Performance bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy and F1
        metrics_df = df[['Model', 'Accuracy', 'F1 Score']].melt(
            id_vars='Model', var_name='Metric', value_name='Score'
        )
        sns.barplot(data=metrics_df, x='Model', y='Score', hue='Metric', ax=ax1)
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Speed comparison
        sns.barplot(data=df, x='Model', y='Speed', ax=ax2)
        ax2.set_title('Inference Speed (samples/second)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrices
        fig, axes = plt.subplots(
            1, len(self.results), 
            figsize=(5 * len(self.results), 5)
        )
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            if 'error' not in result:
                cm = np.array(result['confusion_matrix'])
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['POS', 'NEG', 'NEU'],
                    yticklabels=['POS', 'NEG', 'NEU'],
                    ax=axes[idx]
                )
                axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance vs Speed scatter
        plt.figure(figsize=(10, 8))
        for _, row in df.iterrows():
            plt.scatter(row['Speed'], row['F1 Score'], s=200)
            plt.annotate(
                row['Model'], 
                (row['Speed'], row['F1 Score']),
                xytext=(5, 5), textcoords='offset points'
            )
        
        plt.xlabel('Inference Speed (samples/second)')
        plt.ylabel('F1 Score')
        plt.title('Performance vs Speed Trade-off')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_speed_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self, output_path: str = 'benchmark_report.json'):
        """Generate comprehensive benchmark report."""
        report = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
                'num_models': len(self.models_config),
                'models_evaluated': list(self.models_config.keys())
            },
            'results': self.results,
            'summary': {
                'best_accuracy': max(
                    (r['accuracy'], n) for n, r in self.results.items() 
                    if 'error' not in r
                ),
                'best_f1': max(
                    (r['f1_score'], n) for n, r in self.results.items() 
                    if 'error' not in r
                ),
                'fastest': max(
                    (r['samples_per_second'], n) for n, r in self.results.items() 
                    if 'error' not in r
                ),
                'smallest': min(
                    (r['model_size_mb'], n) for n, r in self.results.items() 
                    if 'error' not in r
                )
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report


def main():
    """Run comprehensive model benchmark."""
    # Define models to benchmark
    models_config = {
        'BERT-base-Japanese': 'cl-tohoku/bert-base-japanese-v3',
        'RoBERTa-base-Japanese': 'nlp-waseda/roberta-base-japanese',
        'DeBERTa-base-Japanese': 'ku-nlp/deberta-v2-base-japanese',
        'ELECTRA-small-Japanese': 'izumi-lab/electra-small-japanese',
    }
    
    # Initialize benchmark
    benchmark = ModelBenchmark(models_config)
    
    # Run benchmark
    print("Starting model benchmark...")
    comparison_df = benchmark.run_benchmark(
        'datasets/sentiment/sample_annotations.json'
    )
    
    # Display results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    benchmark.visualize_results()
    
    # Generate report
    print("Generating detailed report...")
    report = benchmark.generate_report()
    
    print(f"\nBest Accuracy: {report['summary']['best_accuracy'][1]} "
          f"({report['summary']['best_accuracy'][0]:.3f})")
    print(f"Best F1 Score: {report['summary']['best_f1'][1]} "
          f"({report['summary']['best_f1'][0]:.3f})")
    print(f"Fastest Model: {report['summary']['fastest'][1]} "
          f"({report['summary']['fastest'][0]:.1f} samples/sec)")
    
    print("\nBenchmark complete! Results saved to benchmark_report.json")


if __name__ == "__main__":
    main()