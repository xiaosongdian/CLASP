#!/usr/bin/env python3
"""
Scoring module
- F(S): Interaction decision weighted F1-score
- L(S): Content generation semantic alignment (cosine similarity)
- Q(S): Comprehensive alignment score
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer
import torch

from src.config import ACTION_WEIGHTS, ALPHA, NORMALIZE_L_TO_UNIT, SENTENCE_TRANSFORMER_MODEL


class SemanticScorer:
    """Semantic similarity scorer based on sentence-transformers"""

    def __init__(
        self,
        model_path: str = SENTENCE_TRANSFORMER_MODEL,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Args:
            device:
              - None: Use GPU if CUDA available, otherwise CPU (consistent with historical behavior)
              - "cpu" / "cuda" / "cuda:0" etc: Explicit specification; for multi-process, recommend each process use cpu to avoid VRAM exhaustion
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        self.model = self.model.to(self._device)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode text list into normalized vectors"""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two text segments"""
        if not text_a or not text_b:
            return 0.0
        vecs = self._encode([text_a, text_b])
        return float(np.dot(vecs[0], vecs[1]))

    def batch_cosine_similarity(
        self, texts_a: List[str], texts_b: List[str]
    ) -> List[float]:
        """Batch compute cosine similarity"""
        assert len(texts_a) == len(texts_b)
        if not texts_a:
            return []
        vecs_a = self._encode(texts_a)
        vecs_b = self._encode(texts_b)
        sims = np.sum(vecs_a * vecs_b, axis=1)
        return sims.tolist()


def compute_decision_f1(
    predicted_types: List[str],
    actual_types: List[str],
    weights: Dict[str, float] = ACTION_WEIGHTS,
) -> float:
    """
    Compute interaction decision weighted F1-score: F(S)
    Compute F1 separately for each action type, then weighted sum by information entropy weights.
    """
    all_labels = sorted(weights.keys())
    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    pred_idx = [label_to_idx.get(p, -1) for p in predicted_types]
    true_idx = [label_to_idx.get(a, -1) for a in actual_types]

    # Keep only known labels
    valid = [(p, t) for p, t in zip(pred_idx, true_idx) if p >= 0 and t >= 0]
    if not valid:
        return 0.0

    pred_valid, true_valid = zip(*valid)

    per_class_f1 = f1_score(
        true_valid, pred_valid,
        labels=list(range(len(all_labels))),
        average=None,
        zero_division=0.0,
    )

    weighted_f1 = 0.0
    total_weight = 0.0
    for i, label in enumerate(all_labels):
        w = weights.get(label, 0.0)
        weighted_f1 += w * per_class_f1[i]
        total_weight += w

    return weighted_f1 / total_weight if total_weight > 0 else 0.0


def compute_content_similarity(
    predicted_texts: List[str],
    actual_texts: List[str],
    scorer: SemanticScorer,
    predictions: Optional[List[Dict]] = None,
) -> float:
    """
    Compute content generation semantic alignment: L(S)
    Only compute cosine similarity for text of post/reply actions, take average.

    Args:
        predicted_texts: List of predicted texts
        actual_texts: List of actual texts
        scorer: Semantic similarity calculator
        predictions: Optional, list of prediction results. If provided and contains semantic_similarity field,
                    use pre-computed similarity directly to avoid redundant computation.

    Returns:
        Average semantic similarity
    """
    import math

    valid_pred = []
    valid_actual = []
    precomputed_sims = []

    for idx, (p, a) in enumerate(zip(predicted_texts, actual_texts)):
        if p and a:
            # Check if similarity has already been computed
            if predictions and idx < len(predictions):
                pred_dict = predictions[idx]
                if "semantic_similarity" in pred_dict:
                    sim = pred_dict["semantic_similarity"]
                    # Only use valid similarity values (non-NaN)
                    if not math.isnan(sim):
                        precomputed_sims.append(sim)
                        continue

            # Not yet computed, need to compute
            valid_pred.append(p)
            valid_actual.append(a)

    # If all similarities are pre-computed, return average directly
    if not valid_pred and precomputed_sims:
        return float(np.mean(precomputed_sims))

    # If no valid data at all
    if not valid_pred and not precomputed_sims:
        return 0.0

    # Compute similarities not yet pre-computed
    new_sims = scorer.batch_cosine_similarity(valid_pred, valid_actual) if valid_pred else []

    # Merge pre-computed and newly computed similarities
    all_sims = precomputed_sims + new_sims
    return float(np.mean(all_sims))


def compute_alignment_score(
    f_score: float,
    l_score: float,
    alpha: float = ALPHA,
) -> float:
    """
    Comprehensive alignment score Q(S) = α·F(S) + (1-α)·L'(S).
    L' = (L+1)/2 ∈ [0,1] when config.NORMALIZE_L_TO_UNIT is True; otherwise L' = L (raw cosine, approx [-1,1]).
    """
    if NORMALIZE_L_TO_UNIT:
        l_score = (l_score + 1.0) / 2.0
    return alpha * f_score + (1.0 - alpha) * l_score


def evaluate_predictions(
    predictions: List[Dict],
    actuals: List[Dict],
    scorer: SemanticScorer,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    """
    Given predictions and actual actions, compute F(S), L(S), Q(S).

    predictions: [{"action_type": str, "content": str|None, "semantic_similarity": float|None}, ...]
    actuals: Original action list [{"action_type": str, "action_text": str|None, ...}, ...]

    Returns (f_score, l_score, q_score)

    Note: If predictions contain semantic_similarity field (computed by build_behavior_discrepancies),
         it will be used directly to avoid redundant computation.
    """
    pred_types = [p["action_type"] for p in predictions]
    actual_types = [a["action_type"] for a in actuals]
    f_score = compute_decision_f1(pred_types, actual_types)

    # Only compute content similarity for post/reply
    pred_texts = []
    actual_texts = []
    content_predictions = []  # Only contains post/reply predictions
    for pred, actual in zip(predictions, actuals):
        if actual["action_type"] in ("post", "reply"):
            pred_texts.append(pred.get("content") or "")
            actual_texts.append(actual.get("action_text") or "")
            content_predictions.append(pred)

    # Pass predictions to use pre-computed similarity
    l_score = compute_content_similarity(pred_texts, actual_texts, scorer, content_predictions)
    q_score = compute_alignment_score(f_score, l_score, alpha)

    return f_score, l_score, q_score
