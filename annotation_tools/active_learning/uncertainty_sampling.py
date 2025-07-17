#!/usr/bin/env python3
"""
Active Learning for Japanese Text Annotation
Implements uncertainty sampling strategies to reduce annotation effort

Author: Ryo Yanagisawa
Last Updated: 2024-12-15
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintySampler:
    """
    Active learning sampler using uncertainty-based selection strategies.
    Optimized for Japanese text with multi-strategy support.
    """
    
    def __init__(self, 
                 model_name: str = "cl-tohoku/bert-base-japanese-v3",
                 device: str = None):
        """
        Initialize the uncertainty sampler with a pre-trained model.
        
        Args:
            model_name: HuggingFace model identifier
            device: torch device ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing sampler with {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        ).to(self.device)
        self.model.eval()
        
    def get_predictions(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get model predictions with probabilities."""
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    def least_confidence(self, probs: np.ndarray) -> np.ndarray:
        """
        Least confidence strategy: 1 - max(P(y|x))
        Higher scores indicate more uncertainty.
        """
        return 1 - np.max(probs, axis=1)
    
    def margin_sampling(self, probs: np.ndarray) -> np.ndarray:
        """
        Margin sampling: P(y1|x) - P(y2|x)
        Smaller margins indicate more uncertainty.
        """
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        return 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])
    
    def entropy_sampling(self, probs: np.ndarray) -> np.ndarray:
        """
        Entropy-based sampling: -Σ P(y|x) log P(y|x)
        Higher entropy indicates more uncertainty.
        """
        return entropy(probs, axis=1)
    
    def bayesian_disagreement(self, texts: List[str], 
                            n_forward_passes: int = 10) -> np.ndarray:
        """
        Approximate Bayesian uncertainty through dropout sampling.
        Measures disagreement across multiple forward passes.
        """
        self.model.train()  # Enable dropout
        
        all_predictions = []
        for _ in range(n_forward_passes):
            probs = self.get_predictions(texts)
            predictions = np.argmax(probs, axis=1)
            all_predictions.append(predictions)
        
        self.model.eval()  # Disable dropout
        
        # Calculate disagreement
        all_predictions = np.array(all_predictions)
        disagreement_scores = []
        
        for i in range(len(texts)):
            preds = all_predictions[:, i]
            unique, counts = np.unique(preds, return_counts=True)
            disagreement = 1 - (np.max(counts) / n_forward_passes)
            disagreement_scores.append(disagreement)
        
        return np.array(disagreement_scores)
    
    def diversity_sampling(self, texts: List[str], 
                          selected_indices: List[int],
                          embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Diversity-aware sampling to avoid redundant selections.
        Combines uncertainty with diversity metrics.
        """
        if embeddings is None:
            # Get embeddings from model
            embeddings = self.get_embeddings(texts)
        
        if not selected_indices:
            # First selection - return zeros (no diversity penalty)
            return np.zeros(len(texts))
        
        # Calculate similarity to already selected samples
        selected_embeddings = embeddings[selected_indices]
        similarities = cosine_similarity(embeddings, selected_embeddings)
        max_similarities = np.max(similarities, axis=1)
        
        # Diversity score (lower similarity = higher diversity)
        diversity_scores = 1 - max_similarities
        return diversity_scores
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from the model's last hidden layer."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use CLS token embedding from last layer
                embeddings = outputs.hidden_states[-1][:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def select_samples(self,
                      texts: List[str],
                      n_samples: int,
                      strategy: str = 'combined',
                      already_selected: List[int] = None,
                      diversity_weight: float = 0.3) -> Tuple[List[int], Dict]:
        """
        Select samples for annotation using specified strategy.
        
        Args:
            texts: Pool of unlabeled texts
            n_samples: Number of samples to select
            strategy: Selection strategy ('least_confident', 'margin', 
                     'entropy', 'bayesian', 'combined')
            already_selected: Indices of previously selected samples
            diversity_weight: Weight for diversity component (0-1)
            
        Returns:
            selected_indices: Indices of selected samples
            selection_info: Dictionary with selection metrics
        """
        logger.info(f"Selecting {n_samples} samples using {strategy} strategy")
        
        if already_selected is None:
            already_selected = []
        
        # Get predictions
        probs = self.get_predictions(texts)
        
        # Calculate uncertainty scores
        uncertainty_scores = {
            'least_confident': self.least_confidence(probs),
            'margin': self.margin_sampling(probs),
            'entropy': self.entropy_sampling(probs)
        }
        
        if strategy == 'bayesian':
            uncertainty_scores['bayesian'] = self.bayesian_disagreement(texts)
            scores = uncertainty_scores['bayesian']
        elif strategy == 'combined':
            # Combine multiple strategies
            scores = (
                0.4 * uncertainty_scores['entropy'] +
                0.3 * uncertainty_scores['least_confident'] +
                0.3 * uncertainty_scores['margin']
            )
        else:
            scores = uncertainty_scores.get(strategy, uncertainty_scores['entropy'])
        
        # Add diversity component if weight > 0
        if diversity_weight > 0 and len(already_selected) > 0:
            embeddings = self.get_embeddings(texts)
            diversity_scores = self.diversity_sampling(texts, already_selected, embeddings)
            scores = (1 - diversity_weight) * scores + diversity_weight * diversity_scores
        
        # Select top samples
        available_indices = [i for i in range(len(texts)) if i not in already_selected]
        available_scores = scores[available_indices]
        
        selected_idx_in_available = np.argsort(available_scores)[-n_samples:][::-1]
        selected_indices = [available_indices[i] for i in selected_idx_in_available]
        
        # Prepare selection info
        selection_info = {
            'strategy': strategy,
            'avg_uncertainty': float(np.mean(scores[selected_indices])),
            'uncertainty_distribution': {
                'min': float(np.min(scores[selected_indices])),
                'max': float(np.max(scores[selected_indices])),
                'std': float(np.std(scores[selected_indices]))
            },
            'predicted_labels': np.argmax(probs[selected_indices], axis=1).tolist(),
            'confidence_scores': np.max(probs[selected_indices], axis=1).tolist()
        }
        
        logger.info(f"Selected {len(selected_indices)} samples with "
                   f"avg uncertainty: {selection_info['avg_uncertainty']:.3f}")
        
        return selected_indices, selection_info


class ActiveLearningPipeline:
    """
    End-to-end active learning pipeline for Japanese text annotation.
    """
    
    def __init__(self, initial_samples: int = 100):
        self.sampler = UncertaintySampler()
        self.initial_samples = initial_samples
        self.annotation_history = []
        
    def initialize(self, texts: List[str]) -> List[int]:
        """Select initial samples using random + diversity sampling."""
        n = len(texts)
        
        # Random selection for initial diversity
        random_indices = np.random.choice(n, self.initial_samples // 2, replace=False)
        
        # Diversity-based selection for remaining
        embeddings = self.sampler.get_embeddings(texts)
        
        # K-means++ style selection
        selected = list(random_indices)
        remaining = self.initial_samples - len(selected)
        
        for _ in range(remaining):
            if len(selected) == 0:
                selected.append(np.random.randint(n))
            else:
                selected_embeddings = embeddings[selected]
                distances = np.min(
                    np.linalg.norm(embeddings[:, None] - selected_embeddings[None, :], axis=2),
                    axis=1
                )
                probabilities = distances / distances.sum()
                next_idx = np.random.choice(n, p=probabilities)
                selected.append(next_idx)
        
        return selected
    
    def get_next_batch(self, 
                      unlabeled_texts: List[str],
                      labeled_indices: List[int],
                      batch_size: int = 50,
                      strategy: str = 'combined') -> Dict:
        """Get next batch of samples for annotation."""
        
        # Select samples
        selected_indices, selection_info = self.sampler.select_samples(
            unlabeled_texts,
            n_samples=batch_size,
            strategy=strategy,
            already_selected=labeled_indices,
            diversity_weight=0.3
        )
        
        # Track history
        self.annotation_history.append({
            'round': len(self.annotation_history) + 1,
            'n_labeled': len(labeled_indices),
            'batch_size': batch_size,
            'strategy': strategy,
            'selection_info': selection_info
        })
        
        return {
            'indices': selected_indices,
            'texts': [unlabeled_texts[i] for i in selected_indices],
            'info': selection_info,
            'history': self.annotation_history
        }
    
    def estimate_remaining_effort(self, 
                                 current_accuracy: float,
                                 target_accuracy: float = 0.95) -> int:
        """Estimate remaining annotation effort based on learning curve."""
        if len(self.annotation_history) < 2:
            return -1  # Not enough data
        
        # Simple power law estimation
        history = pd.DataFrame(self.annotation_history)
        n_samples = history['n_labeled'].values
        
        # Assume logarithmic improvement
        improvement_rate = (current_accuracy - 0.5) / np.log(n_samples[-1] + 1)
        estimated_samples = np.exp((target_accuracy - 0.5) / improvement_rate) - 1
        
        return max(0, int(estimated_samples - n_samples[-1]))


def main():
    """Example usage of active learning pipeline."""
    # Simulate unlabeled data pool
    sample_texts = [
        "素晴らしい商品でした。期待以上の品質です。",
        "普通の商品です。特に問題はありません。",
        "最悪でした。二度と買いません。",
        "ちょっと期待外れかな。",
        "とても満足しています！",
        # ... more samples
    ] * 20  # Simulate larger dataset
    
    # Initialize pipeline
    pipeline = ActiveLearningPipeline(initial_samples=10)
    
    # Get initial batch
    initial_indices = pipeline.initialize(sample_texts)
    print(f"Initial selection: {len(initial_indices)} samples")
    
    # Simulate annotation rounds
    labeled_indices = list(initial_indices)
    
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Get next batch
        batch = pipeline.get_next_batch(
            sample_texts,
            labeled_indices,
            batch_size=10,
            strategy='combined'
        )
        
        print(f"Selected {len(batch['indices'])} samples")
        print(f"Average uncertainty: {batch['info']['avg_uncertainty']:.3f}")
        
        # Simulate annotation (in practice, this would be human annotation)
        labeled_indices.extend(batch['indices'])
        
        # Estimate remaining effort
        simulated_accuracy = 0.7 + 0.1 * round_num  # Simulate improving accuracy
        remaining = pipeline.estimate_remaining_effort(simulated_accuracy, 0.95)
        print(f"Estimated remaining samples needed: {remaining}")
    
    print(f"\nTotal annotated: {len(labeled_indices)} samples")


if __name__ == "__main__":
    main()