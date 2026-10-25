#!/usr/bin/env python3
"""
Frozen XLM-RoBERTa sentence-embedding baseline on ASSIN 2.

Protocol
--------
* Encoder: xlm-roberta-base, no fine-tuning.
* Sentence embedding: attention-mask-aware mean pooling over last hidden states,
  followed by L2 normalization.
* STS: cosine similarity; report Pearson, Spearman, MAE and RMSE using a fixed
  mapping from cosine [-1, 1] to the ASSIN relatedness range [1, 5].
* RTE: classify entailment from cosine similarity. Select threshold on validation
  only, then freeze it for test. Report macro-F1, positive-class F1, accuracy,
  precision and recall.
* ASSIN labels are used only for evaluation and validation threshold selection;
  they are never used to update the encoder.

Install:
  pip install torch transformers datasets scipy scikit-learn numpy

Run:
  python evaluate_xlmr_assin2.py
  python evaluate_xlmr_assin2.py --model xlm-roberta-base --batch-size 32
  python evaluate_xlmr_assin2.py --output-dir results_xlmr_assin2

For reproducibility, this script evaluates the Hugging Face dataset splits:
nilc-nlp/assin2 (train=6500, validation=500, test=2448).
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def encode_texts(texts, tokenizer, model, device, batch_size=32, max_length=128):
    vectors = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = model(**encoded)
        pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, model.config.hidden_size))


def safe_corr(fn, y_true, y_pred):
    result = fn(y_true, y_pred)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def sts_metrics(gold, cosine):
    # Fixed, label-independent mapping: cosine -1 -> score 1; cosine +1 -> score 5.
    predicted = 1.0 + 2.0 * (np.asarray(cosine) + 1.0)
    gold = np.asarray(gold, dtype=np.float64)
    return {
        "pearson": safe_corr(pearsonr, gold, predicted),
        "spearman": safe_corr(spearmanr, gold, predicted),
        "mae_fixed_cosine_mapping": float(np.mean(np.abs(gold - predicted))),
        "rmse_fixed_cosine_mapping": float(np.sqrt(np.mean((gold - predicted) ** 2))),
    }, predicted


def choose_rte_threshold(cosine, labels):
    # Exhaustive candidate thresholds based on observed validation similarities.
    cosine = np.asarray(cosine, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.unique(np.concatenate([
        cosine,
        np.array([-1.000001, 1.000001]),
        (np.unique(cosine)[:-1] + np.unique(cosine)[1:]) / 2.0,
    ]))
    best = None
    for threshold in candidates:
        pred = (cosine >= threshold).astype(np.int64)
        score = f1_score(labels, pred, average="macro", zero_division=0)
        # Deterministic tie-break: choose threshold closest to 0.5, then lower.
        key = (float(score), -abs(float(threshold) - 0.5), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def rte_metrics(cosine, labels, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    pred = (np.asarray(cosine) >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "entailment_f1": float(f1_score(labels, pred, pos_label=1, zero_division=0)),
        "precision_entailment": float(precision_score(labels, pred, pos_label=1, zero_division=0)),
        "recall_entailment": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
    }, pred


def evaluate_split(split_name, split, tokenizer, model, device, args, output_dir):
    premises = split["premise"]
    hypotheses = split["hypothesis"]
    gold_sts = np.asarray(split["relatedness_score"], dtype=np.float64)
    gold_rte = np.asarray(split["entailment_judgment"], dtype=np.int64)

    # The two sides are encoded independently, as expected for sentence embeddings.
    emb_p = encode_texts(premises, tokenizer, model, device, args.batch_size, args.max_length)
    emb_h = encode_texts(hypotheses, tokenizer, model, device, args.batch_size, args.max_length)
    cosine = np.sum(emb_p * emb_h, axis=1).clip(-1.0, 1.0)

    sts, sts_pred = sts_metrics(gold_sts, cosine)
    results = {"n_pairs": len(split), "sts": sts, "cosine_mean": float(cosine.mean()),
               "cosine_std": float(cosine.std())}

    if split_name == "validation":
        threshold = choose_rte_threshold(cosine, gold_rte)
        rte, rte_pred = rte_metrics(cosine, gold_rte, threshold)
        results["rte"] = rte
        results["_threshold"] = threshold
    else:
        threshold = args.rte_threshold
        rte, rte_pred = rte_metrics(cosine, gold_rte, threshold)
        results["rte"] = rte

    if args.save_predictions:
        path = output_dir / f"{split_name}_predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sentence_pair_id", "premise", "hypothesis", "gold_relatedness",
                "cosine_similarity", "predicted_relatedness_fixed_mapping",
                "gold_entailment", "predicted_entailment", "rte_threshold",
            ])
            ids = split["sentence_pair_id"] if "sentence_pair_id" in split.column_names else range(len(split))
            for i in range(len(split)):
                writer.writerow([
                    ids[i], premises[i], hypotheses[i], float(gold_sts[i]),
                    float(cosine[i]), float(sts_pred[i]), int(gold_rte[i]),
                    int(rte_pred[i]), float(threshold),
                ])

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--dataset", default="nilc-nlp/assin2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_xlmr_assin2")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}", flush=True)
    dataset = load_dataset(args.dataset)
    required = {"train", "validation", "test"}
    if not required.issubset(set(dataset.keys())):
        raise ValueError(f"Expected splits {sorted(required)}; found {list(dataset.keys())}")
    for name in ("train", "validation", "test"):
        print(f"{name}: {len(dataset[name])} pairs", flush=True)

    print(f"Loading frozen encoder: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # No training split is used for model fitting. It is reported for dataset
    # transparency only; threshold selection uses validation only.
    results = {
        "model": args.model,
        "pooling": "attention-mask-aware mean pooling of last hidden state; L2 normalized",
        "encoder_finetuned": False,
        "sts_mapping": "predicted_score = 1 + 2 * (cosine_similarity + 1)",
        "rte_threshold_selection": "validation macro-F1; threshold applied unchanged to test",
        "device": str(device),
        "splits": {},
    }

    val = evaluate_split("validation", dataset["validation"], tokenizer, model, device, args, output_dir)
    threshold = val.pop("_threshold")
    args.rte_threshold = threshold
    results["splits"]["validation"] = val
    results["rte_threshold_selected_on_validation"] = threshold

    test = evaluate_split("test", dataset["test"], tokenizer, model, device, args, output_dir)
    results["splits"]["test"] = test

    out_json = output_dir / "metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print("FROZEN XLM-ROBERTA BASELINE — ASSIN 2")
    print("=" * 68)
    for split_name in ("validation", "test"):
        r = results["splits"][split_name]
        print(f"\n{split_name.upper()} (n={r['n_pairs']})")
        print(f"STS Pearson : {r['sts']['pearson']:.4f}")
        print(f"STS Spearman: {r['sts']['spearman']:.4f}")
        print(f"STS MAE     : {r['sts']['mae_fixed_cosine_mapping']:.4f}")
        print(f"STS RMSE    : {r['sts']['rmse_fixed_cosine_mapping']:.4f}")
        print(f"RTE threshold: {r['rte']['threshold']:.6f}")
        print(f"RTE macro-F1: {r['rte']['macro_f1']:.4f}")
        print(f"RTE F1      : {r['rte']['entailment_f1']:.4f}")
        print(f"RTE accuracy: {r['rte']['accuracy']:.4f}")
        print(f"RTE precision: {r['rte']['precision_entailment']:.4f}")
        print(f"RTE recall   : {r['rte']['recall_entailment']:.4f}")
    print(f"\nMetrics saved to: {out_json.resolve()}")
    if args.save_predictions:
        print(f"Predictions saved to: {output_dir.resolve()}/*_predictions.csv")


if __name__ == "__main__":
    main()
