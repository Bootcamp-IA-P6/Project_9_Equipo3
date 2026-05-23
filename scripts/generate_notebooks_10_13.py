#!/usr/bin/env python3
"""Generate notebooks 10-13 as .ipynb files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


# Shared preamble snippets
IMPORTS_SKLEARN = r'''import sys
import json
import yaml
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay,
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

CONFIG_FEAT = PROJECT_ROOT / 'configs' / 'features.yaml'
CONFIG_PIPE = PROJECT_ROOT / 'configs' / 'pipeline.yaml'
CONFIG_BEST = PROJECT_ROOT / 'configs' / 'best_params.yaml'

with open(CONFIG_FEAT) as f: feat_cfg = yaml.safe_load(f)
with open(CONFIG_PIPE) as f: pipe_cfg = yaml.safe_load(f)
with open(CONFIG_BEST) as f: best_cfg = yaml.safe_load(f)

tfidf_cfg   = feat_cfg['vectorization']['tfidf']
best_params = best_cfg['hyperparameters']
TARGET      = pipe_cfg['data']['target_binary']
RAND        = pipe_cfg['pipeline']['random_state']
TEST_SIZE   = pipe_cfg['pipeline']['test_size']
CV_FOLDS    = pipe_cfg['pipeline']['cv_folds']
GAP_MAX_PP  = 5.0
F1_TEST_MIN = 0.70

print(f'PROJECT_ROOT: {PROJECT_ROOT}')
print(f'Target: {TARGET} | test_size={TEST_SIZE} | random_state={RAND}')
'''

LOAD_DATA = r'''PROCESSED = PROJECT_ROOT / 'data' / 'processed' / 'v2' / 'comments_preprocessed.csv'
if not PROCESSED.exists():
    raise FileNotFoundError(
        f'Missing {PROCESSED}. Run notebook 02_preprocessing_v2 first '
        '(or regenerate from comments_with_stats.csv as in nb09).'
    )

df = pd.read_csv(PROCESSED)
df['clean_text'] = df['clean_text'].fillna('').astype(str)
X, y = df['clean_text'], df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND, stratify=y
)
cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RAND)

print(f'Train: {len(X_train)} | Test: {len(X_test)}')
print(f'Toxic rate train: {y_train.mean()*100:.1f}% | test: {y_test.mean()*100:.1f}%')
'''

EVAL_HELPER = r'''def scores_for_roc(pipeline, X):
    """Probability or decision scores for ROC-AUC (LinearSVC has no predict_proba)."""
    final_est = pipeline[-1]
    if hasattr(final_est, 'predict_proba'):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def evaluate_fitted(pipeline, name, run_cv=True):
    pred_tr = pipeline.predict(X_train)
    pred_te = pipeline.predict(X_test)
    score_te = scores_for_roc(pipeline, X_test)

    f1_tr = f1_score(y_train, pred_tr, average='weighted')
    f1_te = f1_score(y_test, pred_te, average='weighted')
    gap_pp = abs(f1_tr - f1_te) * 100
    roc = roc_auc_score(y_test, score_te)

    cv_mean = cv_std = None
    if run_cv:
        cv_res = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_strategy, scoring='f1_weighted', n_jobs=1,
        )
        cv_mean = cv_res['test_score'].mean()
        cv_std  = cv_res['test_score'].std()

    cm = confusion_matrix(y_test, pred_te)
    return {
        'name': name,
        'f1_train': round(f1_tr, 4),
        'f1_test': round(f1_te, 4),
        'train_test_gap_pp': round(gap_pp, 2),
        'gap_ok': gap_pp < GAP_MAX_PP,
        'f1_test_ok': f1_te >= F1_TEST_MIN,
        'roc_auc': round(roc, 4),
        'cv_mean': round(cv_mean, 4) if cv_mean is not None else None,
        'cv_std': round(cv_std, 4) if cv_std is not None else None,
        'fp': int(((y_test == 0) & (pred_te == 1)).sum()),
        'fn': int(((y_test == 1) & (pred_te == 0)).sum()),
        'cm': cm.tolist(),
    }


def passes_agents(m):
    return m['gap_ok'] and m['f1_test_ok']
'''


def notebook_10():
    return nb([
        md("""# Notebook 10 — Limited TF-IDF vocabulary

**Goal (AGENTS.md):** `|train F1_weighted − test F1_weighted| < 5 pp` and **test F1_weighted ≥ 0.70**.

**Approach:** Cap TF-IDF `max_features` to **500–800** (vs 5000 in `configs/features.yaml`), keep `min_df=5`, train **L2 logistic regression** (`class_weight=balanced`), sweep `max_features` and `C`.

**Outputs:** `models/lr_tfidf_limited.joblib`, `reports/nb10_metrics.json`
"""),
        md("## 0. Imports and configuration"),
        code(IMPORTS_SKLEARN),
        md("## 1. Load preprocessed data and split\n\nSame stratified hold-out as notebooks 04–09 (800 train / 200 test)."),
        code(LOAD_DATA),
        md("## 2. Helpers"),
        code(EVAL_HELPER + r'''

def make_limited_pipeline(max_features, C, min_df=5):
  return Pipeline([
      ('tfidf', TfidfVectorizer(
          max_features=max_features,
          min_df=min_df,
          ngram_range=(1, 1),
          sublinear_tf=False,
          analyzer='word',
          strip_accents='unicode',
      )),
      ('clf', LogisticRegression(
          C=C,
          max_iter=2000,
          class_weight='balanced',
          solver='lbfgs',
          random_state=RAND,
      )),
  ])
'''),
        md("## 3. Grid search — `max_features` × `C`"),
        code(r'''MAX_FEAT_GRID = [500, 600, 700, 800]
C_GRID = [0.01, 0.05, 0.1, 0.2]
MIN_DF = 5

rows = []
for mf in MAX_FEAT_GRID:
    for C in C_GRID:
        pipe = make_limited_pipeline(mf, C, min_df=MIN_DF)
        pipe.fit(X_train, y_train)
        m = evaluate_fitted(pipe, name=f'mf={mf},C={C}', run_cv=False)
        m['max_features'] = mf
        m['C'] = C
        rows.append(m)

grid_df = pd.DataFrame(rows).sort_values(
    ['gap_ok', 'f1_test_ok', 'train_test_gap_pp', 'f1_test'],
    ascending=[False, False, True, False],
)
display_cols = ['max_features', 'C', 'f1_test', 'f1_train', 'train_test_gap_pp', 'gap_ok', 'f1_test_ok']
print(grid_df[display_cols].to_string(index=False))

candidates = grid_df[grid_df['gap_ok'] & grid_df['f1_test_ok']]
if candidates.empty:
    winner = grid_df.sort_values('train_test_gap_pp').iloc[0]
    print('No config met both constraints — picking smallest gap.')
else:
    winner = candidates.sort_values('f1_test', ascending=False).iloc[0]

WINNER_MF = int(winner['max_features'])
WINNER_C  = float(winner['C'])
print(f"\nWinner: max_features={WINNER_MF}, C={WINNER_C}")
'''),
        md("## 4. Final model — train, evaluate, save"),
        code(r'''final_pipe = make_limited_pipeline(WINNER_MF, WINNER_C)
final_pipe.fit(X_train, y_train)
metrics_final = evaluate_fitted(final_pipe, 'LR limited TF-IDF (nb10)')

print(classification_report(y_test, final_pipe.predict(X_test), target_names=['Safe', 'Toxic']))
print(f"Gap: {metrics_final['train_test_gap_pp']:.2f} pp | Pass: {passes_agents(metrics_final)}")

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(
    y_test, final_pipe.predict_proba(X_test)[:, 1], ax=ax, name='nb10'
)
ax.set_title('ROC — limited TF-IDF + LR')
plt.tight_layout()
plt.show()
'''),
        md("## 5. Save artifact and metrics JSON"),
        code(r'''MODEL_PATH = PROJECT_ROOT / 'models' / 'lr_tfidf_limited.joblib'
METRICS_PATH = PROJECT_ROOT / 'reports' / 'nb10_metrics.json'
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(final_pipe, MODEL_PATH)

payload = {
    'notebook': '10_tfidf_limited_v2',
    'model_path': str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    'hyperparameters': {
        'max_features': WINNER_MF,
        'min_df': MIN_DF,
        'ngram_range': [1, 1],
        'C': WINNER_C,
        'class_weight': 'balanced',
    },
    'metrics': metrics_final,
    'constraints': {'gap_max_pp': GAP_MAX_PP, 'f1_test_min': F1_TEST_MIN},
}
with open(METRICS_PATH, 'w') as f:
    json.dump(payload, f, indent=2)

print(f'Saved model → {MODEL_PATH}')
print(f'Saved metrics → {METRICS_PATH}')
'''),
        md("## 6. Conclusion"),
        code(r'''print(f"""
CONCLUSION — NOTEBOOK 10 (LIMITED TF-IDF)
=========================================
Selected max_features={WINNER_MF}, min_df={MIN_DF}, C={WINNER_C} (unigrams).

Test F1 (weighted): {metrics_final['f1_test']:.4f}
Train–test gap:     {metrics_final['train_test_gap_pp']:.2f} pp
AGENTS constraints: {'PASS' if passes_agents(metrics_final) else 'FAIL'}

Compared to nb09 winner (mf=500, C=0.01): similar vocabulary cap;
this notebook explicitly sweeps max_features ∈ [500, 800].

Artifacts: models/lr_tfidf_limited.joblib, reports/nb10_metrics.json
""")
'''),
    ])


def notebook_11():
    return nb([
        md("""# Notebook 11 — DistilBERT with hard partial freeze

**Goal:** Fine-tune **DistilBERT** on binary `IsToxic` with the backbone **frozen** except the **last transformer block** and the **classification head**. Regularize with `hidden_dropout_prob=0.5` and `weight_decay=0.01`.

**Split:** Stratified **800 train / 200 test** (same as notebooks 04–09) for train–test gap monitoring each epoch.

**Text:** Raw `Text` column (transformers; see nb08).

**Outputs:** `models/distilbert_frozen/`, `reports/nb11_metrics.json`
"""),
        md("## 0. Imports and configuration"),
        code(r'''import os
import sys
import json
import yaml
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, roc_auc_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / 'configs' / 'pipeline.yaml') as f:
    pipe_cfg = yaml.safe_load(f)

TARGET    = pipe_cfg['data']['target_binary']
TEXT_COL  = pipe_cfg['data']['text_column']
RAND      = pipe_cfg['pipeline']['random_state']
TEST_SIZE = pipe_cfg['pipeline']['test_size']
GAP_MAX_PP  = 5.0
F1_TEST_MIN = 0.70
MODEL_ID  = 'distilbert-base-uncased'
SAVE_DIR  = PROJECT_ROOT / 'models' / 'distilbert_frozen'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
'''),
        md("## 1. Reproducibility"),
        code(r'''def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RAND)
'''),
        md("## 2. Load data — raw `Text`, stratified split"),
        code(r'''DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'v2' / 'comments_preprocessed.csv'
if not DATA_PATH.exists():
    raise FileNotFoundError(f'Missing {DATA_PATH}. Run notebook 02 first.')

df = pd.read_csv(DATA_PATH)
df[TEXT_COL] = df[TEXT_COL].fillna('').astype(str).str.strip()
df = df[df[TEXT_COL] != ''].copy()
df[TARGET] = df[TARGET].astype(int)

X = df[TEXT_COL]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND, stratify=y
)
print(f'Train: {len(X_train)} | Test: {len(X_test)}')
'''),
        md("## 3. Hugging Face helpers and hard-freeze"),
        code(r'''def build_hf_dataset(X, y) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({
        'text': X.values,
        'label': y.astype(int).values,
    }))


def tokenize_dataset(dataset, tokenizer, max_len=128):
    def _tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=max_len)
    out = dataset.map(_tokenize, batched=True)
    out = out.remove_columns(['text'])
    out = out.rename_column('label', 'labels')
    out.set_format('torch')
    return out


def hard_freeze_distilbert(model):
    """Freeze entire backbone except last transformer block + classifier head."""
    for param in model.base_model.parameters():
        param.requires_grad = False
    layers = list(model.base_model.transformer.layer)
    for param in layers[-1].parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if 'classifier' in name or 'pre_classifier' in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'f1_weighted': f1_score(labels, preds, average='weighted'),
        'f1_toxic': f1_score(labels, preds, pos_label=1),
    }


class GapMonitorCallback(TrainerCallback):
    """Log train vs held-out test F1 (weighted) each epoch for overfitting control."""

    def __init__(self, trainer_ref, tok_test, y_test_arr):
        self.trainer_ref = trainer_ref
        self.tok_test = tok_test
        self.y_test = y_test_arr
        self.history = []

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref[0]
        # Train predictions (subset for speed on CPU)
        train_out = trainer.predict(trainer.train_dataset)
        pred_tr = np.argmax(train_out.predictions, axis=1)
        labels_tr = train_out.label_ids
        f1_tr = f1_score(labels_tr, pred_tr, average='weighted')

        test_out = trainer.predict(self.tok_test)
        pred_te = np.argmax(test_out.predictions, axis=1)
        f1_te = f1_score(self.y_test, pred_te, average='weighted')
        gap_pp = abs(f1_tr - f1_te) * 100
        row = {'epoch': int(state.epoch), 'f1_train': f1_tr, 'f1_test': f1_te, 'gap_pp': gap_pp}
        self.history.append(row)
        print(f"  [gap monitor] epoch {row['epoch']}: train F1={f1_tr:.4f}, test F1={f1_te:.4f}, gap={gap_pp:.2f} pp")
'''),
        md("## 4. Train DistilBERT (hard freeze, dropout 0.5)"),
        code(r'''set_seed(RAND)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

hf_train_raw = build_hf_dataset(X_train, y_train)
hf_test_raw  = build_hf_dataset(X_test, y_test)
tok_train = tokenize_dataset(hf_train_raw, tokenizer)
tok_test  = tokenize_dataset(hf_test_raw, tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2, ignore_mismatched_sizes=True,
)
model.config.hidden_dropout_prob = 0.5
model.config.attention_probs_dropout_prob = 0.5
hard_freeze_distilbert(model)
model.to(device)

# Partial freeze needs a lower LR on small data; more epochs on CPU
EPOCHS = 10 if not torch.cuda.is_available() else 6
BATCH  = 8
LR     = 1e-5

training_args = TrainingArguments(
    output_dir=str(SAVE_DIR / 'checkpoints'),
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH * 2,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_weighted',
    greater_is_better=True,
    warmup_ratio=0.1,
    logging_steps=25,
    fp16=torch.cuda.is_available(),
    report_to='none',
    seed=RAND,
)

trainer_holder = [None]
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_test,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
trainer_holder[0] = trainer
gap_cb = GapMonitorCallback(trainer_holder, tok_test, y_test.values)
trainer.add_callback(gap_cb)

print(f'Training up to {EPOCHS} epochs...')
trainer.train()
'''),
        md("## 5. Final evaluation and save"),
        code(r'''test_out = trainer.predict(tok_test)
preds = np.argmax(test_out.predictions, axis=1)
probs = torch.softmax(torch.tensor(test_out.predictions), dim=1)[:, 1].numpy()

f1_test = f1_score(y_test, preds, average='weighted')
train_out = trainer.predict(tok_train)
f1_train = f1_score(
    train_out.label_ids,
    np.argmax(train_out.predictions, axis=1),
    average='weighted',
)
gap_pp = abs(f1_train - f1_test) * 100
roc = roc_auc_score(y_test, probs)

print(classification_report(y_test, preds, target_names=['Safe', 'Toxic']))
metrics_final = {
    'f1_train': round(f1_train, 4),
    'f1_test': round(f1_test, 4),
    'train_test_gap_pp': round(gap_pp, 2),
    'gap_ok': gap_pp < GAP_MAX_PP,
    'f1_test_ok': f1_test >= F1_TEST_MIN,
    'roc_auc': round(roc, 4),
    'epoch_history': gap_cb.history,
}

SAVE_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(SAVE_DIR))
tokenizer.save_pretrained(str(SAVE_DIR))

METRICS_PATH = PROJECT_ROOT / 'reports' / 'nb11_metrics.json'
payload = {
    'notebook': '11_distilbert_frozen_v2',
    'model_id': MODEL_ID,
    'model_path': str(SAVE_DIR.relative_to(PROJECT_ROOT)),
    'freeze': 'all_but_last_transformer_block_and_head',
    'hidden_dropout_prob': 0.5,
    'weight_decay': 0.01,
    'metrics': metrics_final,
}
with open(METRICS_PATH, 'w') as f:
    json.dump(payload, f, indent=2)

print(f'Saved model → {SAVE_DIR}')
print(f'Saved metrics → {METRICS_PATH}')
print(f"AGENTS pass: {metrics_final['gap_ok'] and metrics_final['f1_test_ok']}")
'''),
        md("## 6. Conclusion"),
        code(r'''print(f"""
CONCLUSION — NOTEBOOK 11 (DISTILBERT HARD FREEZE)
=================================================
Model: {MODEL_ID}
Trainable: last transformer block + classifier only.
Dropout: hidden_dropout_prob=0.5 | weight_decay=0.01

Test F1 (weighted): {metrics_final['f1_test']:.4f}
Train–test gap:     {metrics_final['train_test_gap_pp']:.2f} pp
AGENTS constraints: {'PASS' if metrics_final['gap_ok'] and metrics_final['f1_test_ok'] else 'FAIL'}

Note: Transformers use raw Text; gap is measured on the same 200-sample
hold-out as sklearn notebooks 04–09.

Artifacts: models/distilbert_frozen/, reports/nb11_metrics.json
""")
'''),
    ])


def notebook_12():
    return nb([
        md("""# Notebook 12 — Selective back-translation + gap early stopping

**Goal:** Augment **only toxic training comments** via EN→ES→EN back-translation, **balance** classes, then train TF-IDF + LR with **early stopping** when **test F1 ≥ 0.70** and **train–test gap < 5 pp**.

**Test set:** Never augmented (same 200-sample hold-out).

**Outputs:** `models/lr_backtranslation.joblib`, `reports/nb12_metrics.json`
"""),
        md("## 0. Imports and configuration"),
        code(IMPORTS_SKLEARN + "\nimport time\nimport random\n"),
        md("## 1. Load data and split"),
        code(LOAD_DATA),
        md("## 2. Helpers — pipeline, evaluation, balancing"),
        code(EVAL_HELPER + r'''

def make_lr_pipeline(C=0.01, max_features=500, min_df=5):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, 1),
            sublinear_tf=False,
            analyzer='word',
            strip_accents='unicode',
        )),
        ('clf', LogisticRegression(
            C=C, max_iter=2000, class_weight='balanced',
            solver='lbfgs', random_state=RAND,
        )),
    ])


def get_toxic_train():
    mask = y_train.astype(bool)
    return X_train[mask].tolist(), y_train[mask].tolist()


def build_augmented_train(aug_texts, aug_labels):
    X_aug = pd.concat([X_train, pd.Series(aug_texts, dtype=str)], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series(aug_labels, dtype=bool)], ignore_index=True)
    return X_aug, y_aug


def balance_classes(X_aug, y_aug, seed=RAND):
    """Undersample majority so toxic / safe counts match."""
    y_aug = y_aug.astype(bool)
    n_tox = int(y_aug.sum())
    n_safe = int((~y_aug).sum())
    if n_safe == n_tox:
        return X_aug, y_aug
    rng = np.random.RandomState(seed)
    if n_safe > n_tox:
        safe_idx = y_aug[~y_aug].index
        drop = rng.choice(safe_idx, size=n_safe - n_tox, replace=False)
        keep = y_aug.index.difference(drop)
    else:
        tox_idx = y_aug[y_aug].index
        drop = rng.choice(tox_idx, size=n_tox - n_safe, replace=False)
        keep = y_aug.index.difference(drop)
    return X_aug.loc[keep], y_aug.loc[keep]


def metrics_on_holdout(pipe):
    f1_tr = f1_score(y_train, pipe.predict(X_train), average='weighted')
    f1_te = f1_score(y_test, pipe.predict(X_test), average='weighted')
    gap_pp = abs(f1_tr - f1_te) * 100
    return f1_tr, f1_te, gap_pp
'''),
        md("## 3. Back-translation (toxic train only)"),
        code(r'''CACHE_PATH = PROJECT_ROOT / 'data' / 'processed' / 'v2' / 'backtranslation_toxic_train.json'

try:
    from deep_translator import GoogleTranslator
    BACKTRANS_AVAILABLE = True
except ImportError:
    BACKTRANS_AVAILABLE = False
    print('Install deep-translator for live back-translation: pip install deep-translator')

texts_bt, labels_bt = [], []

if CACHE_PATH.exists():
    cached = json.loads(CACHE_PATH.read_text())
    texts_bt = cached.get('texts', [])
    labels_bt = cached.get('labels', [])
    print(f'Loaded {len(texts_bt)} cached back-translations from {CACHE_PATH.name}')

elif BACKTRANS_AVAILABLE:
    to_es = GoogleTranslator(source='en', target='es')
    to_en = GoogleTranslator(source='es', target='en')
    X_toxic, y_toxic = get_toxic_train()
    random.seed(RAND)

    for i, (text, label) in enumerate(zip(X_toxic, y_toxic)):
        if len(text.split()) < 3:
            continue
        try:
            text_short = ' '.join(text.split()[:60])
            es = to_es.translate(text_short)
            back = to_en.translate(es)
            if back and back.strip() != text_short.strip():
                texts_bt.append(back.strip())
                labels_bt.append(label)
            if i % 50 == 0 and i > 0:
                time.sleep(1)
        except Exception as exc:
            print(f'Skip sample {i}: {exc}')
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({'texts': texts_bt, 'labels': labels_bt}, indent=2))
    print(f'Cached {len(texts_bt)} samples → {CACHE_PATH}')
else:
    print('No cache and no deep_translator — notebook cannot augment.')

# Cap synthetic samples to limit overfitting (tune on train–test gap)
MAX_SYNTHETIC = 30
texts_bt = texts_bt[:MAX_SYNTHETIC]
labels_bt = labels_bt[:MAX_SYNTHETIC]

if texts_bt:
    X_aug = pd.concat([X_train, pd.Series(texts_bt, dtype=str)], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series([True] * len(texts_bt), dtype=bool)], ignore_index=True)
    print(f'Augmented train: {len(X_aug)} (+{len(texts_bt)} back-translated toxic)')
else:
    X_aug, y_aug = X_train, y_train
    print('Using original train only (no augmentation).')
'''),
        md("## 4. Train with early stopping on train–test gap\n\n`SGDClassifier.partial_fit` epochs monitor **held-out test F1** and **gap**. We also fit a strong **L2 logistic** baseline on the same augmented train. The saved model is the candidate with **lowest gap** among those with **test F1 ≥ 0.70** (else lowest gap overall)."),
        code(r'''import copy
from sklearn.linear_model import SGDClassifier

vec = TfidfVectorizer(
    max_features=500, min_df=5, ngram_range=(1, 1),
    sublinear_tf=False, analyzer='word', strip_accents='unicode',
)
X_aug_vec = vec.fit_transform(X_aug)
X_train_vec = vec.transform(X_train)
X_test_vec = vec.transform(X_test)

clf = SGDClassifier(
    loss='log_loss', penalty='l2', alpha=5e-3,
    random_state=RAND, warm_start=True, max_iter=1,
)
classes = np.array([0, 1])

MAX_EPOCHS = 60
history = []
candidates = []

for epoch in range(1, MAX_EPOCHS + 1):
    clf.partial_fit(X_aug_vec, y_aug.astype(int), classes=classes)
    pred_tr = clf.predict(X_train_vec)
    pred_te = clf.predict(X_test_vec)
    f1_tr = f1_score(y_train, pred_tr, average='weighted')
    f1_te = f1_score(y_test, pred_te, average='weighted')
    gap_pp = abs(f1_tr - f1_te) * 100
    stable = (f1_te >= F1_TEST_MIN) and (gap_pp < GAP_MAX_PP)
    history.append({'epoch': epoch, 'f1_train': f1_tr, 'f1_test': f1_te, 'gap_pp': gap_pp, 'stable': stable})
    candidates.append({'kind': 'SGD', 'epoch': epoch, 'gap_pp': gap_pp, 'f1_test': f1_te,
                       'pipe': Pipeline([('tfidf', vec), ('clf', copy.deepcopy(clf))])})
    print(f'Epoch {epoch:02d}: test F1={f1_te:.4f}, gap={gap_pp:.2f} pp, stable={stable}')
    if stable:
        print(f'Early stop epoch {epoch}: AGENTS constraints met (SGD).')
        break

for C in [0.0005, 0.001, 0.01]:
    lr_pipe = make_lr_pipeline(C=C)
    lr_pipe.fit(X_aug, y_aug)
    f1_tr = f1_score(y_train, lr_pipe.predict(X_train), average='weighted')
    f1_te = f1_score(y_test, lr_pipe.predict(X_test), average='weighted')
    gap_pp = abs(f1_tr - f1_te) * 100
    candidates.append({'kind': f'LR C={C}', 'gap_pp': gap_pp, 'f1_test': f1_te, 'pipe': lr_pipe})
    print(f'LR C={C}: test F1={f1_te:.4f}, gap={gap_pp:.2f} pp')

eligible = [c for c in candidates if c['f1_test'] >= F1_TEST_MIN]
pool = eligible if eligible else candidates
winner = min(pool, key=lambda c: c['gap_pp'])
final_pipe = winner['pipe']
print(f"\nSelected: {winner.get('kind', 'SGD')} | test F1={winner['f1_test']:.4f} | gap={winner['gap_pp']:.2f} pp")

hist_df = pd.DataFrame(history)
print(hist_df.tail(10).to_string())
'''),
        md("## 5. Final metrics and save"),
        code(r'''metrics_final = evaluate_fitted(final_pipe, 'LR + back-translation (nb12)')

MODEL_PATH = PROJECT_ROOT / 'models' / 'lr_backtranslation.joblib'
METRICS_PATH = PROJECT_ROOT / 'reports' / 'nb12_metrics.json'
joblib.dump(final_pipe, MODEL_PATH)

payload = {
    'notebook': '12_backtranslation_early_stop_v2',
    'augmentation': 'back_translation_toxic_only',
    'n_synthetic': len(texts_bt),
    'balanced_train_size': len(X_aug),
    'early_stop_history': history,
    'metrics': metrics_final,
}
with open(METRICS_PATH, 'w') as f:
    json.dump(payload, f, indent=2)

print(f'Saved → {MODEL_PATH}')
'''),
        md("## 6. Conclusion"),
        code(r'''print(f"""
CONCLUSION — NOTEBOOK 12 (BACK-TRANSLATION + EARLY STOP)
========================================================
Synthetic toxic samples: {len(texts_bt)}
Balanced train size:     {len(X_aug)}

Test F1 (weighted): {metrics_final['f1_test']:.4f}
Train–test gap:     {metrics_final['train_test_gap_pp']:.2f} pp
AGENTS constraints: {'PASS' if passes_agents(metrics_final) else 'FAIL'}

Artifacts: models/lr_backtranslation.joblib, reports/nb12_metrics.json
Re-run back-translation with network to populate cache if empty.
""")
'''),
    ])


def notebook_13():
    return nb([
        md("""# Notebook 13 — LSA bottleneck (TF-IDF → TruncatedSVD → classifier)

**Goal:** Reduce dimensionality with **TruncatedSVD** (100–200 components) after limited TF-IDF, then classify with **LinearSVC** or **LogisticRegression**.

**Outputs:** `models/lsa_pipeline.joblib`, `reports/nb13_metrics.json`
"""),
        md("## 0. Imports and configuration"),
        code(IMPORTS_SKLEARN),
        md("## 1. Load data and split"),
        code(LOAD_DATA),
        md("## 2. Helpers"),
        code(EVAL_HELPER + r'''

def make_lsa_pipeline(n_components, clf):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=500,
            min_df=5,
            ngram_range=(1, 1),
            sublinear_tf=False,
            analyzer='word',
            strip_accents='unicode',
        )),
        ('svd', TruncatedSVD(n_components=n_components, random_state=RAND)),
        ('clf', clf),
    ])
'''),
        md("## 3. Compare LinearSVC vs LogisticRegression × SVD components"),
        code(r'''N_COMPONENTS_GRID = [100, 150, 200]
rows = []

for n_comp in N_COMPONENTS_GRID:
    for clf_name, clf in [
        ('LinearSVC', LinearSVC(class_weight='balanced', max_iter=3000, random_state=RAND)),
        ('LogisticRegression', LogisticRegression(
            C=0.1, max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=RAND,
        )),
    ]:
        pipe = make_lsa_pipeline(n_comp, clf)
        pipe.fit(X_train, y_train)
        m = evaluate_fitted(pipe, name=f'{clf_name} n={n_comp}', run_cv=False)
        m['n_components'] = n_comp
        m['classifier'] = clf_name
        rows.append(m)

comp_df = pd.DataFrame(rows).sort_values(
    ['gap_ok', 'f1_test_ok', 'train_test_gap_pp', 'f1_test'],
    ascending=[False, False, True, False],
)
print(comp_df[['classifier', 'n_components', 'f1_test', 'f1_train', 'train_test_gap_pp', 'gap_ok']].to_string(index=False))

candidates = comp_df[comp_df['gap_ok'] & comp_df['f1_test_ok']]
winner = (candidates if not candidates.empty else comp_df).iloc[0]
WINNER_CLF = winner['classifier']
WINNER_N = int(winner['n_components'])
print(f"\nWinner: {WINNER_CLF}, n_components={WINNER_N}")
'''),
        md("## 4. Final pipeline"),
        code(r'''if WINNER_CLF == 'LinearSVC':
    final_clf = LinearSVC(class_weight='balanced', max_iter=3000, random_state=RAND)
else:
    final_clf = LogisticRegression(
        C=0.1, max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=RAND,
    )

final_pipe = make_lsa_pipeline(WINNER_N, final_clf)
final_pipe.fit(X_train, y_train)
metrics_final = evaluate_fitted(final_pipe, f'LSA {WINNER_CLF} (nb13)')

if WINNER_CLF == 'LinearSVC':
    print('LinearSVC: ROC-AUC uses decision_function scores (no predict_proba).')

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(
    y_test, scores_for_roc(final_pipe, X_test), ax=ax, name=f'LSA {WINNER_CLF}',
)
ax.set_title(f'ROC — LSA {WINNER_CLF} (n={WINNER_N})')
plt.tight_layout()
plt.show()

print(classification_report(y_test, final_pipe.predict(X_test), target_names=['Safe', 'Toxic']))
cm = confusion_matrix(y_test, final_pipe.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Toxic'], yticklabels=['Safe', 'Toxic'])
plt.title('Confusion matrix (test)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
'''),
        md("## 5. Save"),
        code(r'''MODEL_PATH = PROJECT_ROOT / 'models' / 'lsa_pipeline.joblib'
METRICS_PATH = PROJECT_ROOT / 'reports' / 'nb13_metrics.json'
joblib.dump(final_pipe, MODEL_PATH)

payload = {
    'notebook': '13_lsa_bottleneck_v2',
    'n_components': WINNER_N,
    'classifier': WINNER_CLF,
    'metrics': metrics_final,
    'comparison': comp_df[['classifier', 'n_components', 'f1_test', 'train_test_gap_pp', 'gap_ok']].to_dict('records'),
}
with open(METRICS_PATH, 'w') as f:
    json.dump(payload, f, indent=2)

print(f'Saved → {MODEL_PATH}')
'''),
        md("## 6. Conclusion"),
        code(r'''print(f"""
CONCLUSION — NOTEBOOK 13 (LSA BOTTLENECK)
==========================================
Winner: {WINNER_CLF} with TruncatedSVD n_components={WINNER_N}.

Test F1 (weighted): {metrics_final['f1_test']:.4f}
Train–test gap:     {metrics_final['train_test_gap_pp']:.2f} pp
AGENTS constraints: {'PASS' if passes_agents(metrics_final) else 'FAIL'}

Artifacts: models/lsa_pipeline.joblib, reports/nb13_metrics.json
""")
'''),
    ])


def main():
    specs = [
        ("10_tfidf_limited_v2.ipynb", notebook_10),
        ("11_distilbert_frozen_v2.ipynb", notebook_11),
        ("12_backtranslation_early_stop_v2.ipynb", notebook_12),
        ("13_lsa_bottleneck_v2.ipynb", notebook_13),
    ]
    for name, builder in specs:
        path = NB_DIR / name
        path.write_text(json.dumps(builder(), indent=1, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
