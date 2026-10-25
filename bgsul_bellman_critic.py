#!/usr/bin/env python3
"""BGSUL: Bellman-Gated Semantic Unit Learner (self-supervised).

Extends semantic_bellman.py with:
- Portuguese Wikipedia loading (Hugging Face datasets, optional dependency)
- document-aware sentence segmentation
- self-supervised two-view contrastive/invariance losses
- online critic + slowly updated target critic
- robust confidence-weighted Bellman consistency
- typed reciprocal semantic and within-document contextual graph edges
- periodic graph refresh and residual-informed boundary segmentation
- diagnostics, JSONL logs, embeddings, units, and model checkpoints

No reinforcement-learning environment or ASSIN labels are used.

Install:
    pip install tensorflow numpy datasets

Examples:
    python bgsul_bellman_critic.py --max-documents 5000 --epochs 10
    python bgsul_bellman_critic.py --file corpus.txt --epochs 20
    python bgsul_bellman_critic.py --demo --epochs 3
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DEFAULT_CORPUS = """
A aprendizagem automática permite identificar padrões em dados.
As redes neuronais aprendem representações de informação complexa.
A aprendizagem profunda utiliza várias camadas de redes neuronais.
Os transformadores são arquiteturas neuronais baseadas em atenção.
A atenção relaciona diferentes partes de uma sequência.
Os modelos de linguagem aprendem regularidades estatísticas a partir de texto.

Portugal tem investido em energias renováveis, incluindo energia eólica e solar.
Estas tecnologias alteraram a composição da produção de eletricidade no país.
A história de Portugal inclui diferentes períodos políticos e culturais.
A astronomia estuda estrelas, planetas e outros objetos do universo.
"""


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    # Keep Portuguese diacritics and punctuation; sentence tokenizer is intentionally dependency-free.
    chunks = re.split(r"(?<=[.!?…])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ0-9])", normalize_text(text))
    out = []
    for s in chunks:
        s = normalize_text(s.strip(' \t\r\n"“”'))
        if len(s.split()) >= 3:
            out.append(s)
    return out


def load_corpus(args):
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read()
        sents = split_sentences(raw)
        # A single local file is treated as one document.
        return sents, [0] * len(sents), [os.path.basename(args.file)] * len(sents)
    if args.demo:
        docs = [x.strip() for x in re.split(r"\n\s*\n", DEFAULT_CORPUS) if x.strip()]
        sentences, doc_ids, titles = [], [], []
        for di, doc in enumerate(docs):
            ss = split_sentences(doc)
            sentences.extend(ss); doc_ids.extend([di] * len(ss)); titles.extend([f"demo_{di}"] * len(ss))
        return sentences, doc_ids, titles
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Portuguese Wikipedia requires `pip install datasets`. Or pass --file or --demo.") from e
    print(f"Loading {args.dataset_name!r}, config={args.dataset_config!r} (streaming={args.streaming})")
    ds = load_dataset(args.dataset_name, args.dataset_config, split="train", streaming=args.streaming)
    sentences, doc_ids, titles = [], [], []
    count = 0
    for row in ds:
        title = normalize_text(str(row.get("title", "")))
        body = str(row.get("text", row.get("content", "")))
        if len(body) < args.min_document_chars:
            continue
        combined = (title + ". " if title else "") + body
        ss = split_sentences(combined)
        if len(ss) < 2:
            continue
        sentences.extend(ss); doc_ids.extend([count] * len(ss)); titles.extend([title] * len(ss))
        count += 1
        if count >= args.max_documents:
            break
    if len(sentences) < 3:
        raise RuntimeError("Corpus yielded fewer than three usable sentences. Check dataset/config/network settings.")
    return sentences, doc_ids, titles


class TransformerBlock(layers.Layer):
    def __init__(self, dim, heads, ff_dim, dropout):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation=tf.nn.gelu), layers.Dropout(dropout), layers.Dense(dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, padding_mask=None, training=False):
        mask = None
        if padding_mask is not None:
            mask = tf.logical_and(padding_mask[:, None, :], padding_mask[:, :, None])
        a = self.attn(query=x, value=x, key=x, attention_mask=mask, training=training)
        x = self.norm1(x + self.drop1(a, training=training))
        y = self.ffn(x, training=training)
        return self.norm2(x + self.drop2(y, training=training))


class TextEncoder(Model):
    def __init__(self, vocab_size, max_len, dim, heads, ff_dim, dropout):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, dim, mask_zero=True)
        self.pos_emb = layers.Embedding(max_len, dim)
        self.block = TransformerBlock(dim, heads, ff_dim, dropout)
        self.projection = layers.Dense(dim)

    def call(self, ids, training=False):
        mask = tf.not_equal(ids, 0)
        length = tf.shape(ids)[1]
        x = self.token_emb(ids) + self.pos_emb(tf.range(length))[None, :, :]
        x = self.block(x, padding_mask=mask, training=training)
        m = tf.cast(mask[:, :, None], tf.float32)
        pooled = tf.reduce_sum(x * m, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1.0)
        return tf.math.l2_normalize(self.projection(pooled), axis=-1)


class Critic(Model):
    def __init__(self, dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(dim, activation=tf.nn.gelu), layers.LayerNormalization(),
            layers.Dense(dim // 2, activation=tf.nn.gelu), layers.Dense(1)
        ])

    def call(self, z, training=False):
        return self.net(z, training=training)


def make_vectorizer(sentences, vocab_size, max_len):
    vec = layers.TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=max_len,
                                   standardize="lower_and_strip_punctuation")
    vec.adapt(tf.data.Dataset.from_tensor_slices(sentences).batch(256))
    return vec


def augment_ids(ids, mask_id, probability):
    # Token masking as a semantics-preserving noisy view; never mask padding.
    r = tf.random.uniform(tf.shape(ids))
    mask = tf.logical_and(r < probability, tf.not_equal(ids, 0))
    return tf.where(mask, tf.cast(mask_id, ids.dtype), ids)


def nt_xent(z1, z2, temperature):
    logits = tf.matmul(z1, z2, transpose_b=True) / temperature
    labels = tf.range(tf.shape(z1)[0])
    a = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    b = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
    return tf.reduce_mean(0.5 * (a + b))


def huber(x, delta=1.0):
    ax = tf.abs(x)
    return tf.where(ax <= delta, 0.5 * tf.square(x), delta * (ax - 0.5 * delta))


def lexical_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def build_graph(embeddings: np.ndarray, sentences: List[str], doc_ids: List[int], k: int,
                semantic_min_sim: float, context_weight: float):
    """Return edge arrays: type 0 semantic reciprocal-neighbour, type 1 contextual adjacency."""
    n = len(sentences)
    sims = np.asarray(embeddings, np.float32) @ np.asarray(embeddings, np.float32).T
    np.fill_diagonal(sims, -np.inf)
    k_eff = min(k, max(1, n - 1))
    top = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
    nbrs = [set(row.tolist()) for row in top]
    edges: Dict[Tuple[int, int, int], float] = {}
    for i in range(n):
        for j in nbrs[i]:
            if i in nbrs[j] and sims[i, j] >= semantic_min_sim:
                a, b = (i, j) if i < j else (j, i)
                # Confidence is based on similarity, but treated as a weight rather than a label.
                w = float(np.clip((sims[i, j] - semantic_min_sim) / max(1e-6, 1.0 - semantic_min_sim), 0.1, 1.0))
                edges[(a, b, 0)] = max(edges.get((a, b, 0), 0.0), w)
    # Context edges connect adjacent sentences only within the same source document.
    by_doc = defaultdict(list)
    for i, d in enumerate(doc_ids):
        by_doc[d].append(i)
    for indices in by_doc.values():
        for i, j in zip(indices[:-1], indices[1:]):
            a, b = (i, j) if i < j else (j, i)
            # Context edges have intentionally low confidence; they do not assert semantic similarity.
            edges[(a, b, 1)] = max(edges.get((a, b, 1), 0.0), context_weight)
    if not edges:
        # Tiny-corpus fallback: contextual edges only.
        for i in range(n - 1):
            edges[(i, i + 1, 1)] = context_weight
    src, dst, typ, weights = [], [], [], []
    for (i, j, t), w in edges.items():
        src.extend([i, j]); dst.extend([j, i]); typ.extend([t, t]); weights.extend([w, w])
    return (np.asarray(src, np.int32), np.asarray(dst, np.int32), np.asarray(typ, np.int32),
            np.asarray(weights, np.float32), sims)


@dataclass
class Config:
    vocab_size: int = 16000
    max_len: int = 64
    dim: int = 192
    heads: int = 4
    ff_dim: int = 384
    dropout: float = 0.1
    gamma: float = 0.95
    temperature: float = 0.15
    mask_probability: float = 0.12
    lambda_bellman: float = 0.25
    lambda_invariance: float = 0.25
    lambda_contrastive: float = 1.0
    lambda_preserve: float = 0.05
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    target_tau: float = 0.02


class BGSUL:
    def __init__(self, vectorizer, cfg: Config, mask_id=1):
        self.vectorizer, self.cfg, self.mask_id = vectorizer, cfg, mask_id
        self.encoder = TextEncoder(cfg.vocab_size, cfg.max_len, cfg.dim, cfg.heads, cfg.ff_dim, cfg.dropout)
        self.critic = Critic(cfg.dim)
        self.target_critic = Critic(cfg.dim)
        dummy = tf.zeros([2, cfg.max_len], tf.int32)
        z = self.encoder(dummy)
        self.critic(z); self.target_critic(z)
        self.target_critic.set_weights(self.critic.get_weights())
        self.reference_encoder = TextEncoder(cfg.vocab_size, cfg.max_len, cfg.dim, cfg.heads, cfg.ff_dim, cfg.dropout)
        self.reference_encoder(dummy)
        self.reference_encoder.set_weights(self.encoder.get_weights())
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    @tf.function(reduce_retracing=True)
    def train_step(self, ids_a, ids_b, ids_dst, edge_type, edge_weight):
        with tf.GradientTape() as tape:
            z_a = self.encoder(ids_a, training=True)
            ids_aug = augment_ids(ids_a, self.mask_id, self.cfg.mask_probability)
            z_aug = self.encoder(ids_aug, training=True)
            z_dst = self.encoder(ids_dst, training=True)
            z_ref = tf.stop_gradient(self.reference_encoder(ids_a, training=False))

            # Edge type 0 = reciprocal semantic neighbour; type 1 = contextual adjacency.
            v_src = self.critic(z_a, training=True)
            v_dst_target = tf.stop_gradient(self.target_critic(tf.stop_gradient(z_dst), training=False))
            cosine = tf.reduce_sum(z_a * tf.stop_gradient(z_dst), axis=-1, keepdims=True)
            sem = tf.cast(tf.equal(edge_type[:, None], 0), tf.float32)
            # Semantic edges have a similarity-derived target; contextual edges have zero immediate reward.
            # Context links propagate critic value weakly without forcing endpoint embeddings together.
            reward = sem * tf.stop_gradient(cosine)
            target = tf.stop_gradient(reward + self.cfg.gamma * v_dst_target)
            residual = v_src - target
            # Robust, confidence-weighted residual loss. Context edges are further downweighted.
            type_weight = tf.where(tf.equal(edge_type[:, None], 0), 1.0, 0.25)
            effective_weight = edge_weight[:, None] * type_weight
            loss_b = tf.reduce_sum(effective_weight * huber(residual, 1.0)) / tf.maximum(tf.reduce_sum(effective_weight), 1e-6)
            loss_inv = tf.reduce_mean(1.0 - tf.reduce_sum(z_a * z_aug, axis=-1))
            loss_c = nt_xent(z_a, z_aug, self.cfg.temperature)
            loss_p = tf.reduce_mean(1.0 - tf.reduce_sum(z_a * z_ref, axis=-1))
            total = (self.cfg.lambda_bellman * loss_b + self.cfg.lambda_invariance * loss_inv +
                     self.cfg.lambda_contrastive * loss_c + self.cfg.lambda_preserve * loss_p)
        variables = self.encoder.trainable_variables + self.critic.trainable_variables
        grads = tape.gradient(total, variables)
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        gradients, vars_ = zip(*pairs)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, vars_))
        # Polyak update of target critic.
        for target_var, online_var in zip(self.target_critic.variables, self.critic.variables):
            target_var.assign((1.0 - self.cfg.target_tau) * target_var + self.cfg.target_tau * online_var)
        abs_residual = tf.abs(residual)
        sem_mask = tf.cast(tf.equal(edge_type[:, None], 0), tf.float32)
        ctx_mask = tf.cast(tf.equal(edge_type[:, None], 1), tf.float32)
        sem_residual = tf.reduce_sum(abs_residual * sem_mask) / tf.maximum(tf.reduce_sum(sem_mask), 1.0)
        ctx_residual = tf.reduce_sum(abs_residual * ctx_mask) / tf.maximum(tf.reduce_sum(ctx_mask), 1.0)
        return total, loss_b, loss_inv, loss_c, loss_p, tf.reduce_mean(abs_residual), sem_residual, ctx_residual

    def encode(self, sentences, batch_size=256):
        result = []
        for start in range(0, len(sentences), batch_size):
            ids = self.vectorizer(tf.constant(sentences[start:start + batch_size]))
            result.append(self.encoder(ids, training=False).numpy())
        return np.concatenate(result, axis=0)

    def values(self, embeddings, batch_size=512):
        return np.concatenate([self.critic(embeddings[i:i + batch_size], training=False).numpy().reshape(-1)
                              for i in range(0, len(embeddings), batch_size)])


def segment_units(sentences, doc_ids, embeddings, values, similarities, gamma, boundary_threshold, min_unit_sentences=1):
    """Split each document using similarity, lexical shift and critic-residual evidence."""
    n = len(sentences)
    boundaries = np.zeros(n, dtype=np.float32)
    residuals = np.zeros(n, dtype=np.float32)
    for i in range(n - 1):
        if doc_ids[i] != doc_ids[i + 1]:
            boundaries[i] = 1.0
            continue
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        lex = lexical_jaccard(sentences[i], sentences[i + 1])
        delta = float(values[i] - (0.0 + gamma * values[i + 1]))
        residuals[i] = abs(delta)
        # High dissimilarity and lexical shift suggest a boundary; residual contributes modestly.
        raw = 0.50 * (1.0 - sim) + 0.30 * (1.0 - lex) + 0.20 * min(abs(delta), 2.0) / 2.0
        boundaries[i] = raw
    # Normalize only within the observed range to make threshold interpretable for a given corpus.
    units = []
    start = 0
    current_doc = doc_ids[0] if n else 0
    for i in range(n - 1):
        doc_change = doc_ids[i] != doc_ids[i + 1]
        split_here = doc_change or (boundaries[i] >= boundary_threshold and (i + 1 - start) >= min_unit_sentences)
        if split_here:
            if i + 1 > start:
                units.append({"doc_id": int(current_doc), "start": int(start), "end": int(i + 1),
                              "text": " ".join(sentences[start:i + 1]),
                              "boundary_score_after": float(boundaries[i])})
            start = i + 1
            current_doc = doc_ids[start] if start < n else current_doc
    if start < n:
        units.append({"doc_id": int(current_doc), "start": int(start), "end": int(n),
                      "text": " ".join(sentences[start:n]), "boundary_score_after": None})
    return units, boundaries, residuals


def main():
    p = argparse.ArgumentParser(description="BGSUL self-supervised Bellman critic for Portuguese sentence embeddings")
    p.add_argument("--file", default=None, help="Local plain-text corpus (one document)")
    p.add_argument("--demo", action="store_true", help="Use tiny built-in demo corpus instead of Wikipedia")
    p.add_argument("--dataset-name", default="wikimedia/wikipedia")
    p.add_argument("--dataset-config", default="20231101.pt")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-documents", type=int, default=5000)
    p.add_argument("--min-document-chars", type=int, default=200)
    p.add_argument("--max-sentences", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--graph-k", type=int, default=8)
    p.add_argument("--semantic-min-sim", type=float, default=0.35)
    p.add_argument("--context-weight", type=float, default=0.15)
    p.add_argument("--graph-warmup-epochs", type=int, default=2)
    p.add_argument("--graph-refresh-every", type=int, default=2)
    p.add_argument("--boundary-threshold", type=float, default=0.62)
    p.add_argument("--mask-probability", type=float, default=0.12)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--output-dir", default="bgsul_output")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    sentences, doc_ids, titles = load_corpus(args)
    if len(sentences) > args.max_sentences:
        sentences = sentences[:args.max_sentences]; doc_ids = doc_ids[:args.max_sentences]; titles = titles[:args.max_sentences]
    # Reindex documents to stable consecutive integer ids after truncation.
    remap = {}; doc_ids = [remap.setdefault(d, len(remap)) for d in doc_ids]
    print(f"Sentences: {len(sentences):,} | documents: {len(set(doc_ids)):,}")
    if len(sentences) < 3:
        raise ValueError("Need at least three usable sentences.")

    cfg = Config(vocab_size=args.vocab_size, max_len=args.max_len, dim=args.dim,
                 heads=4 if args.dim % 4 == 0 else 1, ff_dim=args.dim * 2,
                 gamma=args.gamma, temperature=args.temperature, mask_probability=args.mask_probability,
                 batch_size=args.batch_size, learning_rate=args.learning_rate)
    vectorizer = make_vectorizer(sentences, cfg.vocab_size, cfg.max_len)
    vocab = vectorizer.get_vocabulary()
    # TextVectorization reserves 0 for padding and 1 for OOV; OOV is used as a corruption token.
    mask_id = 1
    model = BGSUL(vectorizer, cfg, mask_id)
    token_ids = vectorizer(tf.constant(sentences)).numpy().astype(np.int32)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.jsonl")
    if os.path.exists(log_path): os.remove(log_path)

    graph = None
    for epoch in range(1, args.epochs + 1):
        if graph is None or (epoch > args.graph_warmup_epochs and (epoch - args.graph_warmup_epochs - 1) % max(1, args.graph_refresh_every) == 0):
            embeddings = model.encode(sentences)
            # During warm-up, construct contextual edges only; enable semantic neighbours later.
            graph_similarity_threshold = args.semantic_min_sim if epoch > args.graph_warmup_epochs else 2.0
            src, dst, edge_type, edge_weight, sim_matrix = build_graph(
                embeddings, sentences, doc_ids, args.graph_k, graph_similarity_threshold, args.context_weight)
            graph = (src, dst, edge_type, edge_weight, sim_matrix)
            sem_n = int(np.sum(edge_type == 0)); ctx_n = int(np.sum(edge_type == 1))
            print(f"Graph refresh @ epoch {epoch}: semantic directed edges={sem_n:,}, contextual directed edges={ctx_n:,}")
        src, dst, edge_type, edge_weight, _ = graph
        order = np.random.permutation(len(src))
        totals = []
        for begin in range(0, len(order), cfg.batch_size):
            ids = order[begin:begin + cfg.batch_size]
            if len(ids) < 2:
                continue
            # Keep one edge per source sentence in a batch to reduce accidental contrastive false negatives.
            _, first_positions = np.unique(src[ids], return_index=True)
            ids = ids[np.sort(first_positions)]
            if len(ids) < 2:
                continue
            s = src[ids]; d = dst[ids]
            # For contrastive learning, use source nodes as anchors; their augmented views form positives.
            result = model.train_step(token_ids[s], token_ids[s], token_ids[d], edge_type[ids], edge_weight[ids])
            totals.append([float(x.numpy()) for x in result])
        arr = np.asarray(totals, dtype=np.float64) if totals else np.zeros((1, 8))
        names = ["loss", "bellman", "invariance", "contrastive", "preservation", "residual_abs", "semantic_residual_abs", "contextual_residual_abs"]
        record = {"epoch": epoch, "sentences": len(sentences), "graph_directed_semantic_edges": int(np.sum(edge_type == 0)),
                  "graph_directed_contextual_edges": int(np.sum(edge_type == 1))}
        record.update({name: float(np.mean(arr[:, i])) for i, name in enumerate(names)})
        with open(log_path, "a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("Epoch %03d | loss %.4f | bellman %.4f | inv %.4f | nce %.4f | preserve %.4f | |res| %.4f (sem %.4f / ctx %.4f)" %
              (epoch, record["loss"], record["bellman"], record["invariance"], record["contrastive"], record["preservation"], record["residual_abs"], record["semantic_residual_abs"], record["contextual_residual_abs"]))

    embeddings = model.encode(sentences)
    values = model.values(embeddings)
    # Final boundary inference uses local evidence and critic value differences. The learned graph remains separate.
    final_graph = build_graph(embeddings, sentences, doc_ids, args.graph_k, args.semantic_min_sim, args.context_weight)
    units, boundary_scores, residuals = segment_units(sentences, doc_ids, embeddings, values, final_graph[4], cfg.gamma, args.boundary_threshold)

    np.savez_compressed(os.path.join(args.output_dir, "embeddings.npz"), embeddings=embeddings,
                        values=values, boundary_scores=boundary_scores, residuals=residuals,
                        doc_ids=np.asarray(doc_ids, np.int32), sentences=np.asarray(sentences, dtype=object))
    with open(os.path.join(args.output_dir, "units.jsonl"), "w", encoding="utf-8") as f:
        for u in units: f.write(json.dumps(u, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "sentences.jsonl"), "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            f.write(json.dumps({"index": i, "doc_id": int(doc_ids[i]), "title": titles[i], "text": s,
                                "critic_value": float(values[i]), "boundary_score_after": float(boundary_scores[i]),
                                "residual_proxy": float(residuals[i])}, ensure_ascii=False) + "\n")
    model.encoder.save_weights(os.path.join(args.output_dir, "encoder.weights.h5"))
    model.critic.save_weights(os.path.join(args.output_dir, "critic.weights.h5"))
    model.target_critic.save_weights(os.path.join(args.output_dir, "target_critic.weights.h5"))
    with open(os.path.join(args.output_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    config_out = vars(args).copy(); config_out.update({"embedding_dim": cfg.dim, "vocabulary_actual_size": len(vocab), "sentence_count": len(sentences), "unit_count": len(units), "objective": "self-supervised; no ASSIN labels"})
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f: json.dump(config_out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved outputs to: {args.output_dir}")
    print(f"Units: {len(units):,} | embeddings: {embeddings.shape} | log: {log_path}")
    print("ASSIN labels were not loaded or used.")

if __name__ == "__main__":
    main()
