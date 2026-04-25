#!/usr/bin/env python3
"""
评分模块
- F(S)：交互决策加权 F1-score
- L(S)：内容生成语义对齐度（余弦相似度）
- Q(S)：综合对齐得分
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer
import torch

from src.config import ACTION_WEIGHTS, ALPHA, NORMALIZE_L_TO_UNIT, SENTENCE_TRANSFORMER_MODEL


class SemanticScorer:
    """基于 sentence-transformers 的语义相似度评分器"""

    def __init__(
        self,
        model_path: str = SENTENCE_TRANSFORMER_MODEL,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        device:
          - None: 有 CUDA 时用 GPU，否则 CPU（与历史行为一致）
          - "cpu" / "cuda" / "cuda:0" 等：显式指定，多进程时建议各进程用 cpu 避免多份占满显存
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
        """将文本列表编码为归一化向量"""
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
        """计算两段文本的余弦相似度"""
        if not text_a or not text_b:
            return 0.0
        vecs = self._encode([text_a, text_b])
        return float(np.dot(vecs[0], vecs[1]))

    def batch_cosine_similarity(
        self, texts_a: List[str], texts_b: List[str]
    ) -> List[float]:
        """批量计算余弦相似度"""
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
    计算交互决策加权 F1-score: F(S)
    对每种动作类型分别计算 F1，再按信息熵权重加权求和。
    """
    all_labels = sorted(weights.keys())
    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    pred_idx = [label_to_idx.get(p, -1) for p in predicted_types]
    true_idx = [label_to_idx.get(a, -1) for a in actual_types]

    # 只保留已知标签
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
) -> float:
    """
    计算内容生成语义对齐度: L(S)
    仅对 post/reply 动作的文本做余弦相似度，取平均。
    """
    valid_pred = []
    valid_actual = []
    for p, a in zip(predicted_texts, actual_texts):
        if p and a:
            valid_pred.append(p)
            valid_actual.append(a)

    if not valid_pred:
        return 0.0

    sims = scorer.batch_cosine_similarity(valid_pred, valid_actual)
    return float(np.mean(sims))


def compute_alignment_score(
    f_score: float,
    l_score: float,
    alpha: float = ALPHA,
) -> float:
    """
    综合对齐得分:
    - 默认: Q(S) = α·F(S) + (1-α)·L(S)
    - 当 NORMALIZE_L_TO_UNIT=True: 先做 L_norm=(L+1)/2，再计算 Q
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
    给定预测结果和真实动作，计算 F(S), L(S), Q(S)。

    predictions: [{"action_type": str, "content": str|None}, ...]
    actuals: 原始动作列表 [{"action_type": str, "action_text": str|None, ...}, ...]

    返回 (f_score, l_score, q_score)
    """
    pred_types = [p["action_type"] for p in predictions]
    actual_types = [a["action_type"] for a in actuals]
    f_score = compute_decision_f1(pred_types, actual_types)

    # 只对 post/reply 计算内容相似度
    pred_texts = []
    actual_texts = []
    for pred, actual in zip(predictions, actuals):
        if actual["action_type"] in ("post", "reply"):
            pred_texts.append(pred.get("content") or "")
            actual_texts.append(actual.get("action_text") or "")

    l_score = compute_content_similarity(pred_texts, actual_texts, scorer)
    q_score = compute_alignment_score(f_score, l_score, alpha)

    return f_score, l_score, q_score
