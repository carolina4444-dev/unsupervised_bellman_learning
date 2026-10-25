#!/usr/bin/env python3
"""Evaluate a trained BGSUL encoder on ASSIN 2 without training the encoder.

Default protocol:
- Load the saved BGSUL encoder, vocabulary and config.
- Use ASSIN 2 validation only to select a cosine threshold for binary entailment.
- Report semantic-similarity Pearson/Spearman on test using raw cosine similarities.
- Report test entailment metrics at the validation-selected threshold.
- Do not update model weights or fit a supervised sentence-pair model.

Expected model directory (produced by bgsul_bellman_critic.py):
  config.json, vocabulary.json, encoder.weights.h5

Dependencies: tensorflow, numpy, datasets. scipy/sklearn are not required.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.layers import TextVectorization
except ImportError:
    tf = None
    TextVectorization = None



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', default='bgsul_output', help='BGSUL output directory')
    p.add_argument('--dataset-name', default='nilc-nlp/assin2', help='Hugging Face dataset id')
    p.add_argument('--validation-split', default='validation')
    p.add_argument('--test-split', default='test')
    p.add_argument('--output-dir', default='assin2_eval')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto')
    p.add_argument('--max-examples', type=int, default=0, help='Optional debug cap per split; 0 means all rows')
    return p.parse_args()


def configure_device(mode):
    if tf is None:
        raise RuntimeError('TensorFlow is required. Activate the BGSUL .venv and install the same TensorFlow version used for training.')
    gpus = tf.config.list_physical_devices('GPU')
    if mode == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        print('Device: CPU (explicitly requested)')
    elif mode == 'gpu':
        if not gpus:
            raise RuntimeError(' --device gpu was requested, but TensorFlow detected no GPU.')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('Detected GPU(s):', gpus)
    else:
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print('Detected GPU(s):', gpus)
        else:
            print('No GPU detected; using CPU.')


def load_encoder(model_dir):
    model_dir = Path(model_dir)
    required = ['config.json', 'vocabulary.json', 'encoder.weights.h5']
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing files in {model_dir}: {missing}. Run BGSUL training first.')

    with open(model_dir / 'config.json', encoding='utf-8') as f:
        cfg = json.load(f)
    with open(model_dir / 'vocabulary.json', encoding='utf-8') as f:
        vocab = json.load(f)
    if not isinstance(vocab, list) or len(vocab) < 2:
        raise ValueError('vocabulary.json must contain the vocabulary saved by BGSUL.')

    # Import the exact encoder class used by the training script, avoiding a duplicated architecture.
    try:
        from bgsul_bellman_critic import TextEncoder
    except Exception as exc:
        raise RuntimeError(
            'Could not import TextEncoder from bgsul_bellman_critic.py. Keep this evaluator '
            'in the same directory as the training script.'
        ) from exc

    max_len = int(cfg.get('max_len', 64))
    vocab_size = int(cfg.get('vocab_size', 16000))
    dim = int(cfg.get('dim', cfg.get('embedding_dim', 192)))
    heads = 4 if dim % 4 == 0 else 1
    ff_dim = dim * 2

    vectorizer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_len,
        standardize='lower_and_strip_punctuation',
    )
    # TextVectorization reserves indices 0 (padding) and 1 (OOV) automatically.
    vectorizer.set_vocabulary(vocab[2:])

    encoder = TextEncoder(vocab_size, max_len, dim, heads, ff_dim, 0.1)
    encoder(tf.zeros((2, max_len), dtype=tf.int32), training=False)  # build variables before loading weights
    encoder.load_weights(str(model_dir / 'encoder.weights.h5'))
    print(f'Loaded encoder: dim={dim}, max_len={max_len}, configured_vocab_size={vocab_size}, actual_vocab={len(vocab)}')
    return encoder, vectorizer, max_len


def load_split(dataset_name, split, max_examples=0):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('The Hugging Face `datasets` package is required: pip install datasets') from exc
    ds = load_dataset(dataset_name, split=split)
    if max_examples and max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    required = {'premise', 'hypothesis'}
    if not required.issubset(set(ds.column_names)):
        raise ValueError(f'{dataset_name}/{split} missing columns {required}; found {ds.column_names}')
    return ds


def label_to_binary(label):
    # ASSIN 2 labels: 0=NONE, 1=ENTAILMENT. Support strings and HF ClassLabel values.
    if isinstance(label, (int, np.integer)):
        return int(label == 1)
    if isinstance(label, float) and math.isfinite(label):
        return int(round(label) == 1)
    text = str(label).strip().upper()
    if 'ENTAIL' in text and 'NONE' not in text:
        return 1
    if text in {'1', 'TRUE', 'YES'}:
        return 1
    if text in {'0', 'NONE', 'FALSE', 'NO'}:
        return 0
    raise ValueError(f'Cannot map entailment label to binary class: {label!r}')


def encode_texts(encoder, vectorizer, texts, batch_size):
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        batch = tf.constant([str(x) for x in texts[start:start + batch_size]])
        ids = vectorizer(batch)
        z = encoder(ids, training=False).numpy().astype(np.float32)
        # Encoder outputs are normalized already; normalize defensively for cosine evaluation.
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        all_vecs.append(z)
    if not all_vecs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_vecs, axis=0)


def rankdata_average(values):
    """Average ranks for ties; avoids requiring scipy."""
    values = np.asarray(values)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        # Ranks are one-based; average ranks across the tied group.
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    return ranks


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pearson(x, y):
    return correlation(x, y)


def spearman(x, y):
    return correlation(rankdata_average(x), rankdata_average(y))


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'accuracy': float((tp + tn) / max(1, len(y_true))),
        'precision_entailment': float(precision),
        'recall_entailment': float(recall),
        'f1_entailment': float(f1),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def choose_threshold(y_true, scores):
    """Choose threshold on validation only; maximize F1, then accuracy, then precision."""
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    unique = np.unique(scores)
    if len(unique) == 0:
        raise ValueError('Cannot select threshold from an empty validation split.')
    # Midpoints include both extreme all-negative/all-positive operating points.
    candidates = [float(np.nextafter(unique[0], -np.inf)), float(np.nextafter(unique[-1], np.inf))]
    candidates.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())
    best = None
    for threshold in candidates:
        pred = (scores >= threshold).astype(np.int32)
        m = binary_metrics(y_true, pred)
        key = (m['f1_entailment'], m['accuracy'], m['precision_entailment'], -abs(threshold))
        if best is None or key > best[0]:
            best = (key, threshold, m)
    return float(best[1]), best[2]


def evaluate_split(ds, encoder, vectorizer, batch_size):
    premise = [str(x) for x in ds['premise']]
    hypothesis = [str(x) for x in ds['hypothesis']]
    e1 = encode_texts(encoder, vectorizer, premise, batch_size)
    e2 = encode_texts(encoder, vectorizer, hypothesis, batch_size)
    if e1.shape != e2.shape or len(e1) != len(ds):
        raise RuntimeError('Embedding shape mismatch while encoding ASSIN 2 pairs.')
    cosine = np.sum(e1 * e2, axis=1).astype(np.float64)
    scores = np.asarray(ds['relatedness_score'], dtype=np.float64) if 'relatedness_score' in ds.column_names else None
    labels = np.asarray([label_to_binary(x) for x in ds['entailment_judgment']], dtype=np.int32) if 'entailment_judgment' in ds.column_names else None
    return {'premise': premise, 'hypothesis': hypothesis, 'emb_premise': e1, 'emb_hypothesis': e2,
            'cosine': cosine, 'relatedness': scores, 'labels': labels}


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    return value


def main():
    args = parse_args()
    configure_device(args.device)
    encoder, vectorizer, max_len = load_encoder(args.model_dir)

    print(f'Loading ASSIN 2 validation split: {args.validation_split}')
    val_ds = load_split(args.dataset_name, args.validation_split, args.max_examples)
    print(f'Loading ASSIN 2 test split: {args.test_split}')
    test_ds = load_split(args.dataset_name, args.test_split, args.max_examples)
    print(f'ASSIN 2 rows: validation={len(val_ds):,}, test={len(test_ds):,}')

    val = evaluate_split(val_ds, encoder, vectorizer, args.batch_size)
    test = evaluate_split(test_ds, encoder, vectorizer, args.batch_size)

    report = {
        'dataset': args.dataset_name,
        'validation_split': args.validation_split,
        'test_split': args.test_split,
        'model_dir': os.path.abspath(args.model_dir),
        'encoder_training': 'self-supervised only; no ASSIN labels used to update encoder weights',
        'protocol': 'zero-shot cosine embeddings; entailment threshold selected on validation only; test evaluated once',
        'embedding_dim': int(test['emb_premise'].shape[1]),
        'n_validation': len(val_ds),
        'n_test': len(test_ds),
    }

    threshold = None
    if val['labels'] is not None and test['labels'] is not None:
        threshold, val_at_threshold = choose_threshold(val['labels'], val['cosine'])
        test_pred = (test['cosine'] >= threshold).astype(np.int32)
        report['rte'] = {
            'threshold_selected_on_validation_cosine': threshold,
            'validation_metrics_at_selected_threshold': val_at_threshold,
            'test_metrics': binary_metrics(test['labels'], test_pred),
        }
    else:
        test_pred = None
        report['rte'] = {'status': 'skipped: entailment_judgment field not available'}

    if val['relatedness'] is not None and test['relatedness'] is not None:
        # Correlations use uncalibrated cosine, as appropriate for zero-shot embedding evaluation.
        # For interpretable score-scale errors, map cosine [-1,1] linearly to ASSIN's [1,5] range;
        # this is a fixed transform, not fitted on ASSIN labels.
        pred_1_to_5 = np.clip(1.0 + 2.0 * (test['cosine'] + 1.0), 1.0, 5.0)
        report['sts'] = {
            'test_pearson_raw_cosine': pearson(test['relatedness'], test['cosine']),
            'test_spearman_raw_cosine': spearman(test['relatedness'], test['cosine']),
            'test_mae_fixed_cosine_to_1_5': float(np.mean(np.abs(pred_1_to_5 - test['relatedness']))),
            'test_rmse_fixed_cosine_to_1_5': float(np.sqrt(np.mean(np.square(pred_1_to_5 - test['relatedness'])))),
            'score_mapping': 'clip(1 + 2 * (cosine + 1), 1, 5); fixed mapping, no label-fitted calibration',
        }
    else:
        report['sts'] = {'status': 'skipped: relatedness_score field not available'}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'assin2_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(safe_json(report), f, ensure_ascii=False, indent=2)

    pred_path = out_dir / 'assin2_test_predictions.csv'
    with open(pred_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'premise', 'hypothesis', 'cosine_similarity', 'relatedness_gold',
                         'entailment_gold', 'entailment_pred', 'threshold'])
        for i in range(len(test['cosine'])):
            writer.writerow([
                i, test['premise'][i], test['hypothesis'][i], float(test['cosine'][i]),
                '' if test['relatedness'] is None else float(test['relatedness'][i]),
                '' if test['labels'] is None else int(test['labels'][i]),
                '' if test_pred is None else int(test_pred[i]),
                '' if threshold is None else threshold,
            ])

    print('\n' + '=' * 64)
    print('ASSIN 2 ZERO-SHOT EVALUATION (BGSUL)')
    print('=' * 64)
    if 'test_pearson_raw_cosine' in report.get('sts', {}):
        print(f"STS Pearson  : {report['sts']['test_pearson_raw_cosine']:.4f}")
        print(f"STS Spearman : {report['sts']['test_spearman_raw_cosine']:.4f}")
        print(f"STS MAE      : {report['sts']['test_mae_fixed_cosine_to_1_5']:.4f} (fixed cosine mapping)")
        print(f"STS RMSE     : {report['sts']['test_rmse_fixed_cosine_to_1_5']:.4f} (fixed cosine mapping)")
    if 'test_metrics' in report.get('rte', {}):
        m = report['rte']['test_metrics']
        print(f"RTE threshold: {threshold:.6f} (selected on validation)")
        print(f"RTE Accuracy : {m['accuracy']:.4f}")
        print(f"RTE Precision: {m['precision_entailment']:.4f}")
        print(f"RTE Recall   : {m['recall_entailment']:.4f}")
        print(f"RTE F1       : {m['f1_entailment']:.4f}")
    print(f'\nMetrics: {out_dir / "assin2_metrics.json"}')
    print(f'Predictions: {pred_path}')
    print('No model weights were trained or updated during evaluation.')


if __name__ == '__main__':
    main()
