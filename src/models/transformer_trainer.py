"""
Transformer fine-tuning (DistilBERT, Toxic-BERT, etc.) with partial or head-only
freezing, label smoothing, gap-aware early stopping, and val threshold tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from src.evaluation.threshold_tuning import predict_with_threshold, search_best_threshold
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _bert_encoder_layers(model) -> list[nn.Module]:
    """Return transformer blocks for BERT / Toxic-BERT."""
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        return list(model.bert.encoder.layer)
    if hasattr(model, "distilbert"):
        return list(model.distilbert.transformer.layer)
    raise AttributeError("Unsupported architecture for layer freeze")


def _distilbert_layers(model) -> list[nn.Module]:
    """Return transformer blocks for DistilBERT (6 layers)."""
    return list(model.distilbert.transformer.layer)


def unfreeze_full_encoder(model) -> int:
    """Train all encoder blocks plus classification head (Final Squeeze / full BERT)."""
    for param in model.parameters():
        param.requires_grad = True

    layers = _bert_encoder_layers(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Full unfreeze — all {len(layers)} encoder blocks + head — "
        f"trainable {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)"
    )
    return len(layers)


def freeze_all_inference(model) -> int:
    """Freeze every parameter — pretrained inference only (Golden Baseline)."""
    for param in model.parameters():
        param.requires_grad = False
    layers = _bert_encoder_layers(model)
    logger.info(
        f"Inference-only — all {len(layers)} encoder blocks + head frozen (zero fine-tuning)"
    )
    return len(layers)


def freeze_head_only(model) -> None:
    """Freeze entire backbone; train classification head only (Expert / Toxic-BERT)."""
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(k in name for k in ("classifier", "pre_classifier", "pooler")):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Head-only freeze — trainable {trainable:,}/{total:,} "
        f"({100 * trainable / total:.2f}%)"
    )


def freeze_encoder_partial(model, freeze_first_n: int = 4) -> int:
    """Freeze first N encoder blocks; train remaining blocks + classification head."""
    for param in model.parameters():
        param.requires_grad = False

    layers = _bert_encoder_layers(model)
    for layer in layers[freeze_first_n:]:
        for param in layer.parameters():
            param.requires_grad = True

    for name, param in model.named_parameters():
        if any(k in name for k in ("classifier", "pre_classifier", "pooler")):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    n_train = len(layers) - freeze_first_n
    logger.info(
        f"Partial freeze: {freeze_first_n}/{len(layers)} blocks frozen — "
        f"training last {n_train} + head — "
        f"trainable {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)"
    )
    return len(layers)


def freeze_distilbert_partial(model, freeze_first_n: int = 4) -> None:
    """Backward-compatible alias for DistilBERT partial freeze."""
    freeze_encoder_partial(model, freeze_first_n=freeze_first_n)


def build_head_only_optimizer(
    model,
    *,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def build_full_optimizer(
    model,
    *,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def build_partial_optimizer(
    model,
    *,
    learning_rate: float = 1e-5,
    encoder_learning_rate: float | None = None,
    head_learning_rate: float | None = None,
    weight_decay: float = 0.01,
    freeze_first_n: int = 4,
) -> torch.optim.Optimizer:
    """
    Parameter groups: frozen layers excluded; top encoder blocks + head (optional split LRs).
    """
    enc_lr = encoder_learning_rate if encoder_learning_rate is not None else learning_rate
    head_lr = head_learning_rate if head_learning_rate is not None else learning_rate

    layers = _bert_encoder_layers(model)
    top_layer_ids = {
        id(p) for layer in layers[freeze_first_n:] for p in layer.parameters()
    }
    head_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and ("classifier" in n or "pre_classifier" in n)
    ]
    head_ids = {id(p) for p in head_params}
    top_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) in top_layer_ids and id(p) not in head_ids
    ]

    groups = []
    if top_params:
        groups.append(
            {"params": top_params, "lr": enc_lr, "weight_decay": weight_decay}
        )
    if head_params:
        groups.append(
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
        )

    if not groups:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    return torch.optim.AdamW(groups)


def _average_state_dicts(state_dicts: list[dict]) -> dict:
    """Element-wise mean of compatible state dicts (Stochastic Weight Averaging)."""
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    avg = {k: v.clone().float() for k, v in state_dicts[0].items()}
    for sd in state_dicts[1:]:
        for k in avg:
            avg[k] += sd[k].float()
    n = float(len(state_dicts))
    return {k: (v / n).to(state_dicts[0][k].dtype) for k, v in avg.items()}


def build_distilbert_optimizer(
    model,
    *,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    freeze_first_n: int = 4,
) -> torch.optim.Optimizer:
    return build_partial_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_first_n=freeze_first_n,
    )


def logits_to_toxic_prob(logits: np.ndarray | torch.Tensor) -> np.ndarray:
    """Map model logits to P(toxic): sigmoid on 'toxic' for 6-label BERT, else softmax."""
    t = torch.as_tensor(logits, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.shape[-1] >= 6:
        return torch.sigmoid(t)[:, 0].numpy()
    return torch.softmax(t, dim=-1)[:, 1].numpy()


def compute_hf_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    probs = logits_to_toxic_prob(logits)
    preds = np.argmax(logits, axis=1)
    return {
        "f1_toxic": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def _symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two logit vectors (R-Drop regularization)."""
    log_p = nn.functional.log_softmax(logits_a, dim=-1)
    log_q = nn.functional.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    kl_pq = nn.functional.kl_div(log_p, q, reduction="batchmean", log_target=False)
    kl_qp = nn.functional.kl_div(log_q, p, reduction="batchmean", log_target=False)
    return (kl_pq + kl_qp) / 2.0


class LabelSmoothingTrainer(Trainer):
    """Cross-entropy with label smoothing for the classification head."""

    def __init__(self, *args, label_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits,
            labels,
            label_smoothing=self.label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss


class RDropTrainer(LabelSmoothingTrainer):
    """R-Drop: dual forward passes + symmetric KL to limit overfitting (Performance Squeeze)."""

    def __init__(self, *args, rdrop_alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs1 = model(**inputs)
        outputs2 = model(**inputs)
        ce = (
            nn.functional.cross_entropy(
                outputs1.logits, labels, label_smoothing=self.label_smoothing
            )
            + nn.functional.cross_entropy(
                outputs2.logits, labels, label_smoothing=self.label_smoothing
            )
        ) / 2.0
        kl = _symmetric_kl(outputs1.logits, outputs2.logits)
        loss = ce + self.rdrop_alpha * kl
        return (loss, outputs1) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            cfg = self.args
            freeze_mode = getattr(self, "_freeze_mode", "partial")
            if freeze_mode == "head_only":
                self.optimizer = build_head_only_optimizer(
                    self.model,
                    learning_rate=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )
            elif freeze_mode == "full_unfreeze":
                self.optimizer = build_full_optimizer(
                    self.model,
                    learning_rate=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )
            else:
                freeze_n = getattr(self, "_freeze_first_n", 4)
                enc_lr = getattr(self, "_encoder_lr", None)
                head_lr = getattr(self, "_head_lr", None)
                self.optimizer = build_partial_optimizer(
                    self.model,
                    learning_rate=cfg.learning_rate,
                    encoder_learning_rate=enc_lr,
                    head_learning_rate=head_lr,
                    weight_decay=cfg.weight_decay,
                    freeze_first_n=freeze_n,
                )
        return self.optimizer


class SWACallback(TrainerCallback):
    """
    Average model weights over the last N completed epochs (Stochastic Weight Averaging).

    Applied in ``on_train_end`` after ``load_best_model_at_end`` so inference uses SWA weights.
    """

    def __init__(self, last_n_epochs: int = 5):
        self.last_n_epochs = last_n_epochs
        self._snapshots: list[dict[str, torch.Tensor]] = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        self._snapshots.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None or not self._snapshots:
            return control
        n_use = min(self.last_n_epochs, len(self._snapshots))
        averaged = _average_state_dicts(self._snapshots[-n_use:])
        model.load_state_dict(averaged, strict=True)
        logger.info(
            f"SWA applied — averaged last {n_use}/{len(self._snapshots)} epoch checkpoints"
        )
        return control


class GapEarlyStoppingCallback(TrainerCallback):
    """
    Stop when validation F1 plateaus OR |train_f1 - val_f1| exceeds max_gap.

    Train F1 is computed on ``train_eval_dataset`` each evaluation step.
    Attach ``trainer`` after constructing the Trainer (see train_distilbert_stable).
    """

    def __init__(
        self,
        train_eval_dataset,
        *,
        patience: int = 3,
        max_gap: float = 0.045,
        metric: str = "f1_toxic",
        gap_check_min_epoch: int = 2,
        gap_stop_enabled: bool = False,
    ):
        self.train_eval_dataset = train_eval_dataset
        self.patience = patience
        self.max_gap = max_gap
        self.metric = metric
        self.gap_check_min_epoch = gap_check_min_epoch
        self.gap_stop_enabled = gap_stop_enabled
        self.best_metric = -1.0
        self.bad_epochs = 0
        self.trainer: Trainer | None = None
        self._checking = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.trainer is None or self._checking:
            return control

        val_f1 = metrics.get(f"eval_{self.metric}", metrics.get(self.metric, 0.0))

        self._checking = True
        try:
            train_metrics = self.trainer.evaluate(
                eval_dataset=self.train_eval_dataset,
                metric_key_prefix="train",
            )
        finally:
            self._checking = False

        train_f1 = train_metrics.get(f"train_{self.metric}", 0.0)
        gap = abs(train_f1 - val_f1)
        logger.info(
            f"Gap monitor — train_f1={train_f1:.4f} val_f1={val_f1:.4f} gap={gap:.4f}"
        )

        epoch = int(state.epoch or 0)
        if (
            self.gap_stop_enabled
            and epoch >= self.gap_check_min_epoch
            and gap > self.max_gap
        ):
            logger.warning(
                f"Gap defense — train-val gap {gap:.4f} > {self.max_gap}; "
                "stopping and reverting to best checkpoint"
            )
            control.should_training_stop = True
            return control

        if val_f1 > self.best_metric:
            self.best_metric = val_f1
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                logger.info(
                    f"Early stop: no {self.metric} improvement for {self.patience} epochs"
                )
                control.should_training_stop = True

        return control


def _predict_probs_from_dataset(
    trainer: Trainer,
    tokenized_dataset,
) -> np.ndarray:
    ds = tokenized_dataset
    if "label" in ds.column_names:
        ds = ds.remove_columns(["label"])
    out = trainer.predict(ds)
    return logits_to_toxic_prob(out.predictions)


def predict_with_tta(
    trainer: Trainer,
    tokenizer,
    texts: list[str],
    labels_placeholder: list[int] | None,
    *,
    max_length: int,
    aug_cfg: dict,
) -> np.ndarray:
    """
    Test-time augmentation: average P(toxic) from original and back-translated texts.
    """
    from datasets import Dataset

    from src.features.augmentation import back_translate_texts

    labels = labels_placeholder if labels_placeholder is not None else [0] * len(texts)

    def _tokenize(ds):
        return tokenizer(ds["text"], truncation=True, max_length=max_length)

    def _prep(raw_texts: list[str]):
        ds = Dataset.from_dict({"text": raw_texts, "label": labels[: len(raw_texts)]})
        tok = ds.map(_tokenize, batched=True)
        drop_cols = [
            c for c in tok.column_names if c not in ("input_ids", "attention_mask", "label")
        ]
        if drop_cols:
            tok = tok.remove_columns(drop_cols)
        tok.set_format("torch")
        return tok

    original_probs = _predict_probs_from_dataset(trainer, _prep(texts))

    if not aug_cfg.get("enabled", False):
        return original_probs

    logger.info(f"TTA — back-translating {len(texts)} test comments")
    bt_texts = back_translate_texts(
        texts,
        source_lang=aug_cfg.get("source_lang", "en"),
        pivot_lang=aug_cfg.get("pivot_lang", "de"),
        max_words=int(aug_cfg.get("max_words", 60)),
        rate_limit_every=int(aug_cfg.get("rate_limit_every", 50)),
        rate_limit_sleep_sec=float(aug_cfg.get("rate_limit_sleep_sec", 1.0)),
    )
    bt_probs = _predict_probs_from_dataset(trainer, _prep(bt_texts))
    averaged = (original_probs + bt_probs) / 2.0
    logger.info("TTA — averaged original and back-translated probabilities")
    return averaged


def _transformer_cfg(cfg: dict) -> dict:
    """Resolve transformer section (expert) or legacy distilbert key."""
    if "transformer" in cfg:
        return cfg["transformer"]
    return cfg["distilbert"]


def _apply_model_freeze(model, bert_cfg: dict) -> tuple[str, int]:
    freeze_mode = bert_cfg.get("freeze_mode", "partial")
    if freeze_mode == "head_only":
        freeze_head_only(model)
        return "head_only", 0

    if freeze_mode in ("full", "full_unfreeze", "all_layers"):
        n_layers = unfreeze_full_encoder(model)
        return "full_unfreeze", 0

    if freeze_mode in ("inference_only", "frozen", "golden_baseline"):
        n_layers = freeze_all_inference(model)
        return "inference_only", n_layers

    layers = _bert_encoder_layers(model)
    if freeze_mode in ("last_n_layers", "train_last_n"):
        train_last_n = int(bert_cfg.get("train_last_n_layers", 4))
        freeze_first_n = max(0, len(layers) - train_last_n)
    else:
        freeze_first_n = int(bert_cfg.get("freeze_first_n_layers", 4))

    freeze_encoder_partial(model, freeze_first_n=freeze_first_n)
    return f"partial_last_{len(layers) - freeze_first_n}", freeze_first_n


def _bert_metrics_from_probs(
    *,
    model_label: str,
    model_id: str,
    freeze_mode: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_probs: np.ndarray,
    test_probs: np.ndarray,
    threshold: float,
    val_f1_at_threshold: float | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    train_preds = predict_with_threshold(train_probs, threshold)
    test_preds = predict_with_threshold(test_probs, threshold)
    f1_train = float(f1_score(y_train, train_preds, average="weighted", zero_division=0))
    f1_test = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))
    f1_toxic_test = float(f1_score(y_test, test_preds, pos_label=1, zero_division=0))
    f1_toxic_train = float(f1_score(y_train, train_preds, pos_label=1, zero_division=0))
    gap_weighted = abs(f1_train - f1_test)
    gap_toxic = abs(f1_toxic_train - f1_toxic_test)
    metrics = {
        "model": model_label,
        "model_id": model_id,
        "freeze_mode": freeze_mode,
        "threshold": round(threshold, 4),
        "val_f1_at_threshold": round(val_f1_at_threshold, 4) if val_f1_at_threshold else None,
        "f1_weighted": round(f1_test, 4),
        "f1_toxic": round(f1_toxic_test, 4),
        "f1_toxic_train": round(f1_toxic_train, 4),
        "train_test_gap_toxic": round(gap_toxic, 4),
        "train_test_gap_toxic_pp": round(gap_toxic * 100, 2),
        "f1_train": round(f1_train, 4),
        "train_test_gap": round(gap_weighted, 4),
        "train_test_gap_pp": round(gap_weighted * 100, 2),
        "gap_ok": gap_weighted < 0.05,
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "fp": int(((y_test == 0) & (test_preds == 1)).sum()),
        "fn": int(((y_test == 1) & (test_preds == 0)).sum()),
    }
    if extra:
        metrics.update(extra)
    return metrics


def evaluate_pretrained_bert_baseline(
    hf_train,
    hf_val,
    hf_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    *,
    seed: int = 42,
    model_label: str = "Golden-Baseline-Toxic-BERT",
) -> dict[str, Any]:
    """
    Step 1 — pretrained Toxic-BERT with all layers frozen (no fine-tuning on project data).
    """
    bert_cfg = cfg.get("baseline", _transformer_cfg(cfg))
    model_id = bert_cfg["model_id"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_len = int(bert_cfg.get("max_length", 128))

    def _tokenize(ds):
        return tokenizer(ds["text"], truncation=True, max_length=max_len)

    def _prep(ds):
        tok = ds.map(_tokenize, batched=True)
        drop_cols = [
            c for c in tok.column_names if c not in ("input_ids", "attention_mask", "label")
        ]
        if drop_cols:
            tok = tok.remove_columns(drop_cols)
        tok.set_format("torch")
        return tok

    tok_train = _prep(hf_train)
    tok_val = _prep(hf_val)
    tok_test = _prep(hf_test)

    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    _apply_model_freeze(model, {**bert_cfg, "freeze_mode": "inference_only"})
    model.eval()

    args = TrainingArguments(
        output_dir="/tmp/golden_baseline_eval",
        per_device_eval_batch_size=int(bert_cfg.get("batch_size", 8)),
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_hf_metrics,
    )

    logger.info(f"Golden Baseline — {model_id} (inference only, no training)")
    tok_val_pred = tok_val.remove_columns(["label"]) if "label" in tok_val.column_names else tok_val
    tok_train_pred = (
        tok_train.remove_columns(["label"]) if "label" in tok_train.column_names else tok_train
    )
    tok_test_pred = (
        tok_test.remove_columns(["label"]) if "label" in tok_test.column_names else tok_test
    )

    val_probs = logits_to_toxic_prob(trainer.predict(tok_val_pred).predictions)
    train_probs = logits_to_toxic_prob(trainer.predict(tok_train_pred).predictions)
    test_probs = logits_to_toxic_prob(trainer.predict(tok_test_pred).predictions)

    y_val_arr = np.asarray(y_val).astype(int)
    y_train_arr = np.asarray(y_train).astype(int)
    y_test_arr = np.asarray(y_test).astype(int)

    th_cfg = bert_cfg.get("threshold_tuning", {})
    threshold = 0.5
    val_f1_at_threshold = None
    if th_cfg.get("enabled", True):
        threshold, val_f1_at_threshold = search_best_threshold(
            y_val_arr,
            val_probs,
            metric=th_cfg.get("metric", "f1_weighted"),
            min_threshold=float(th_cfg.get("min_threshold", 0.05)),
            max_threshold=float(th_cfg.get("max_threshold", 0.95)),
            step=float(th_cfg.get("step", 0.01)),
        )

    metrics = _bert_metrics_from_probs(
        model_label=model_label,
        model_id=model_id,
        freeze_mode="inference_only",
        y_train=y_train_arr,
        y_test=y_test_arr,
        train_probs=train_probs,
        test_probs=test_probs,
        threshold=threshold,
        val_f1_at_threshold=val_f1_at_threshold,
        extra={
            "trained": False,
            "num_labels": int(model.config.num_labels),
            "prob_mode": "sigmoid_toxic" if model.config.num_labels >= 6 else "softmax",
            "esencial_compliant_gap": True,
        },
    )
    metrics["gap_ok"] = metrics["train_test_gap"] < 0.01

    return {
        "metrics": metrics,
        "trainer": trainer,
        "tokenizer": tokenizer,
        "test_probs": test_probs,
        "val_probs": val_probs,
        "train_probs": train_probs,
        "threshold": threshold,
    }


def train_transformer_stable(
    hf_train,
    hf_val,
    hf_test,
    y_test: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    output_dir: Path,
    *,
    seed: int = 42,
    model_label: str = "Transformer-stable",
) -> dict[str, Any]:
    """
    Fine-tune a sequence classifier with stability-focused regularization.

    Returns metrics, trainer, tokenizer, test probabilities, and optimal threshold.
    """
    bert_cfg = _transformer_cfg(cfg)
    model_id = bert_cfg["model_id"]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_len = int(bert_cfg.get("max_length", 128))

    def _tokenize(ds):
        return tokenizer(ds["text"], truncation=True, max_length=max_len)

    def _prep(ds):
        tok = ds.map(_tokenize, batched=True)
        drop_cols = [
            c for c in tok.column_names if c not in ("input_ids", "attention_mask", "label")
        ]
        if drop_cols:
            tok = tok.remove_columns(drop_cols)
        tok.set_format("torch")
        return tok

    tok_train = _prep(hf_train)
    tok_val = _prep(hf_val)
    tok_test = _prep(hf_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "single_label_classification"
    model.config.num_labels = 2
    dropout_p = float(bert_cfg.get("head_dropout", 0.5))
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = dropout_p
    if hasattr(model.config, "seq_classif_dropout"):
        model.config.seq_classif_dropout = dropout_p
    if hasattr(model, "dropout"):
        model.dropout = nn.Dropout(dropout_p)
    model.to(device)

    freeze_mode, freeze_first_n = _apply_model_freeze(model, bert_cfg)

    es_cfg = bert_cfg.get("early_stopping", {})
    es_metric = es_cfg.get("metric", bert_cfg.get("metric_for_best", "f1_toxic"))
    gap_cb = GapEarlyStoppingCallback(
        tok_train,
        patience=int(es_cfg.get("patience", 3)),
        max_gap=float(es_cfg.get("max_train_val_gap", 0.045)),
        metric=es_metric,
        gap_check_min_epoch=int(es_cfg.get("gap_check_min_epoch", 2)),
        gap_stop_enabled=bool(es_cfg.get("gap_stop_enabled", False)),
    )

    callbacks: list[TrainerCallback] = [gap_cb]
    swa_cfg = bert_cfg.get("swa", {})
    swa_cb: SWACallback | None = None
    if swa_cfg.get("enabled", False):
        swa_cb = SWACallback(last_n_epochs=int(swa_cfg.get("last_n_epochs", 5)))
        callbacks.append(swa_cb)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(bert_cfg.get("learning_rate", 1e-5)),
        num_train_epochs=int(bert_cfg.get("max_epochs", 8)),
        per_device_train_batch_size=int(bert_cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(bert_cfg.get("batch_size", 8)),
        weight_decay=float(bert_cfg.get("weight_decay", 0.01)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=bert_cfg.get("metric_for_best", "f1_toxic"),
        greater_is_better=True,
        warmup_ratio=float(bert_cfg.get("warmup_ratio", 0.1)),
        logging_steps=20,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
    )

    rdrop_cfg = bert_cfg.get("rdrop", {})
    label_smooth = float(bert_cfg.get("label_smoothing", 0.1))
    if rdrop_cfg.get("enabled", False):
        trainer = RDropTrainer(
            model=model,
            args=args,
            train_dataset=tok_train,
            eval_dataset=tok_val,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_hf_metrics,
            callbacks=callbacks,
            label_smoothing=label_smooth,
            rdrop_alpha=float(rdrop_cfg.get("alpha", 0.5)),
        )
        rdrop_note = f", R-Drop α={rdrop_cfg.get('alpha', 0.5)}"
    else:
        trainer = LabelSmoothingTrainer(
            model=model,
            args=args,
            train_dataset=tok_train,
            eval_dataset=tok_val,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_hf_metrics,
            callbacks=callbacks,
            label_smoothing=label_smooth,
        )
        rdrop_note = ""

    gap_cb.trainer = trainer
    trainer._freeze_mode = freeze_mode  # noqa: SLF001
    trainer._freeze_first_n = freeze_first_n  # noqa: SLF001
    if bert_cfg.get("encoder_learning_rate") is not None:
        trainer._encoder_lr = float(bert_cfg["encoder_learning_rate"])  # noqa: SLF001
    if bert_cfg.get("head_learning_rate") is not None:
        trainer._head_lr = float(bert_cfg["head_learning_rate"])  # noqa: SLF001

    enc_lr = bert_cfg.get("encoder_learning_rate", bert_cfg.get("learning_rate"))
    head_lr = bert_cfg.get("head_learning_rate", bert_cfg.get("learning_rate"))
    logger.info(
        f"Training {model_id} ({freeze_mode} freeze, enc_lr={enc_lr}, head_lr={head_lr}"
        f"{rdrop_note}"
        f"{', SWA last ' + str(swa_cfg.get('last_n_epochs', 5)) + ' epochs' if swa_cb else ''})..."
    )
    trainer.train()

    val_out = trainer.predict(tok_val)
    val_probs = logits_to_toxic_prob(val_out.predictions)
    y_val_arr = np.asarray(y_val).astype(int)

    th_cfg = bert_cfg.get("threshold_tuning", {})
    threshold = 0.5
    val_f1_at_threshold = None
    if th_cfg.get("enabled", False):
        metric = th_cfg.get("metric", "f1_toxic")
        threshold, val_f1_at_threshold = search_best_threshold(
            y_val_arr,
            val_probs,
            metric=metric,
            min_threshold=float(th_cfg.get("min_threshold", 0.05)),
            max_threshold=float(th_cfg.get("max_threshold", 0.95)),
            step=float(th_cfg.get("step", 0.01)),
        )
        th_step = float(th_cfg.get("step", 0.01))
        logger.info(
            f"Val threshold tuning — best_t={threshold:.3f} "
            f"val_{metric}={val_f1_at_threshold:.4f} (step={th_step})"
        )

    tta_cfg = bert_cfg.get("test_time_augmentation", {})
    test_texts = list(hf_test["text"])
    if tta_cfg.get("enabled", False):
        probs = predict_with_tta(
            trainer,
            tokenizer,
            test_texts,
            list(hf_test["label"]),
            max_length=max_len,
            aug_cfg=tta_cfg,
        )
        preds_default = (probs >= 0.5).astype(int)
    else:
        output = trainer.predict(tok_test)
        probs = logits_to_toxic_prob(output.predictions)
        preds_default = np.argmax(output.predictions, axis=1)
    preds = predict_with_threshold(probs, threshold)

    train_out = trainer.predict(tok_train)
    train_probs = logits_to_toxic_prob(train_out.predictions)
    train_preds = predict_with_threshold(train_probs, threshold)
    train_labels = np.asarray(hf_train["label"]).astype(int)
    y_test_arr = np.asarray(y_test).astype(int)

    f1_train = float(f1_score(train_labels, train_preds, average="weighted", zero_division=0))
    f1_test = float(f1_score(y_test_arr, preds, average="weighted", zero_division=0))
    f1_toxic_test = float(f1_score(y_test_arr, preds, pos_label=1, zero_division=0))
    f1_toxic_train = float(
        f1_score(train_labels, train_preds, pos_label=1, zero_division=0)
    )
    gap_weighted = abs(f1_train - f1_test)
    gap_toxic = abs(f1_toxic_train - f1_toxic_test)

    metrics = {
        "model": model_label,
        "model_id": model_id,
        "freeze_mode": freeze_mode,
        "rdrop_enabled": bool(rdrop_cfg.get("enabled", False)),
        "tta_enabled": bool(tta_cfg.get("enabled", False)),
        "swa_enabled": bool(swa_cfg.get("enabled", False)),
        "swa_epochs_averaged": (
            min(int(swa_cfg.get("last_n_epochs", 5)), len(swa_cb._snapshots))
            if swa_cb and swa_cb._snapshots
            else 0
        ),
        "threshold": round(threshold, 4),
        "threshold_step": float(th_cfg.get("step", 0.01)) if th_cfg.get("enabled") else None,
        "val_f1_at_threshold": round(val_f1_at_threshold, 4) if val_f1_at_threshold else None,
        "f1_weighted": round(f1_test, 4),
        "f1_toxic": round(f1_toxic_test, 4),
        "f1_toxic_train": round(f1_toxic_train, 4),
        "train_test_gap_toxic": round(gap_toxic, 4),
        "train_test_gap_toxic_pp": round(gap_toxic * 100, 2),
        "gap_toxic_ok": gap_toxic < 0.05,
        "f1_train": round(f1_train, 4),
        "train_test_gap": round(gap_weighted, 4),
        "train_test_gap_pp": round(gap_weighted * 100, 2),
        "f1_weighted_default_thresh": round(
            float(f1_score(y_test_arr, preds_default, average="weighted", zero_division=0)), 4
        ),
        "roc_auc": round(float(roc_auc_score(y_test_arr, probs)), 4),
        "fp": int(((y_test_arr == 0) & (preds == 1)).sum()),
        "fn": int(((y_test_arr == 1) & (preds == 0)).sum()),
    }

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return {
        "metrics": metrics,
        "trainer": trainer,
        "tokenizer": tokenizer,
        "test_probs": probs,
        "test_preds": preds,
        "threshold": threshold,
        "val_probs": val_probs,
    }


def infer_transformer_probs(
    model_dir: Path,
    texts,
    *,
    max_length: int = 128,
    batch_size: int = 16,
) -> np.ndarray:
    """Load a saved classifier and return P(toxic) for each text."""
    model_dir = Path(model_dir)
    text_list = list(texts)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(batch_probs.tolist())

    return np.array(probs)


def train_distilbert_stable(
    hf_train,
    hf_val,
    hf_test,
    y_test: np.ndarray,
    cfg: dict,
    output_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Backward-compatible wrapper for stable DistilBERT pipeline."""
    y_val = np.asarray(hf_val["label"]).astype(int)
    return train_transformer_stable(
        hf_train,
        hf_val,
        hf_test,
        y_test,
        y_val,
        cfg,
        output_dir,
        seed=seed,
        model_label="DistilBERT-stable",
    )
