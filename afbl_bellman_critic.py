#!/usr/bin/env python3
"""AFBL: Adversarial Frontier Bellman Learning, TensorFlow implementation.

Extends the user's BGSUL implementation without modifying it. Uses the same
TextEncoder and Critic classes, loads BGSUL encoder/critic weights, and trains:
  * a frontier generator G (scores candidate boundaries from adjacent embeddings)
  * a WGAN-GP frontier discriminator D (scores frontier feature vectors)
  * optionally fine-tunes BGSUL encoder + Bellman critic with self-supervised loss

No ASSIN labels are read or used. Frontier pseudo-evidence is derived only from
local latent distribution shift and Bellman value/residual signals.

Keep this file beside bgsul_bellman_critic.py.
Dependencies: tensorflow, numpy (and the dependencies of BGSUL).
"""
import argparse, json, os, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# Import original implementation; do not duplicate its encoder architecture.
try:
    import bgsul_bellman_critic as bgsul
except Exception as e:
    raise RuntimeError("Place afbl_bellman_critic.py beside bgsul_bellman_critic.py") from e


class FrontierGenerator(Model):
    """Boundary proposal network: adjacent pair features -> frontier logit."""
    def __init__(self, dim, hidden=256, dropout=0.1):
        super().__init__(name="frontier_generator")
        self.net = tf.keras.Sequential([
            layers.Dense(hidden, activation=tf.nn.gelu), layers.LayerNormalization(),
            layers.Dropout(dropout), layers.Dense(hidden // 2, activation=tf.nn.gelu),
            layers.Dense(1)
        ])
    def call(self, features, training=False):
        return self.net(features, training=training)


class FrontierDiscriminator(Model):
    """Wasserstein critic over frontier evidence/features (no sigmoid)."""
    def __init__(self, input_dim, hidden=256):
        super().__init__(name="frontier_discriminator")
        self.net = tf.keras.Sequential([
            layers.Dense(hidden), layers.LeakyReLU(0.2),
            layers.Dense(hidden // 2), layers.LeakyReLU(0.2),
            layers.Dense(1)
        ])
    def call(self, features, training=False):
        return self.net(features, training=training)


def robust_z(x, eps=1e-6):
    """
    Robust normalization using median and MAD.
    TensorFlow 2.20 compatible.
    No TensorFlow Probability dependency.
    """
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    x = tf.reshape(x, [-1])

    sorted_x = tf.sort(x)
    n = tf.shape(sorted_x)[0]

    mid = n // 2

    med = tf.cond(
        tf.equal(n % 2, 0),
        lambda: (sorted_x[mid - 1] + sorted_x[mid]) / 2.0,
        lambda: sorted_x[mid]
    )

    abs_dev = tf.abs(x - med)
    sorted_dev = tf.sort(abs_dev)

    mad = tf.cond(
        tf.equal(n % 2, 0),
        lambda: (sorted_dev[mid - 1] + sorted_dev[mid]) / 2.0,
        lambda: sorted_dev[mid]
    )

    return (x - med) / (1.4826 * mad + eps)


def make_pair_features(z):
    """For ordered embeddings [N,D], return [N-1,4D] features."""
    left, right = z[:-1], z[1:]
    return tf.concat([left, right, left - right, left * right], axis=-1)


def local_shift_features(z, window=5):
    """Non-parametric local distribution shift; output length N-1.

    Uses cosine distance between left/right local means and local variance shift.
    This is pseudo-evidence, not a label and is stop-gradient in adversarial updates.
    """
    z = tf.math.l2_normalize(z, axis=-1)
    n = tf.shape(z)[0]
    idx = tf.range(n)
    feats = []
    w = max(1, int(window))
    for t in tf.range(1, n):
        lo = tf.maximum(0, t - w)
        hi = tf.minimum(n, t + w)
        left = z[lo:t]
        right = z[t:hi]
        ml = tf.reduce_mean(left, axis=0)
        mr = tf.reduce_mean(right, axis=0)
        mean_dist = 1.0 - tf.reduce_sum(tf.math.l2_normalize(ml[None, :], -1) *
                                         tf.math.l2_normalize(mr[None, :], -1))
        vl = tf.reduce_mean(tf.square(left - ml))
        vr = tf.reduce_mean(tf.square(right - mr))
        var_shift = tf.abs(vl - vr)
        feats.append(tf.stack([mean_dist, var_shift]))
    return tf.stack(feats, axis=0) if feats else tf.zeros([0, 2], tf.float32)


def evidence_from_embeddings(z, values, window=5):
    """Build detached pseudo-frontier probability from shift + Bellman residual.

    values are V(z_i). Residual proxy is |V_i - gamma*V_{i+1}| because the
    unsupervised corpus has no externally supplied reward. Robust scaling is
    applied over candidate positions. The resulting probability is only a
    training target/evidence distribution; it is not ground-truth annotation.
    """
    shift = local_shift_features(z, window)
    dv = tf.abs(values[:-1] - 0.95 * values[1:])[:, None]
    raw = tf.concat([shift, dv], axis=-1)
    raw = tf.stop_gradient(raw)
    components = [robust_z(raw[:, i]) for i in range(raw.shape[-1])]
    score = 0.55 * components[0] + 0.20 * components[1] + 0.25 * components[2]
    # Soft target, not a hard boundary label.
    return tf.stop_gradient(tf.sigmoid(score)), tf.stop_gradient(raw)


def gradient_penalty(discriminator, real, fake):
    batch = tf.shape(real)[0]
    eps = tf.random.uniform([batch, 1], 0.0, 1.0)
    interp = eps * real + (1.0 - eps) * fake
    with tf.GradientTape() as tape:
        tape.watch(interp)
        score = discriminator(interp, training=True)
    grad = tape.gradient(score, interp)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=-1) + 1e-12)
    return tf.reduce_mean(tf.square(norm - 1.0))


def load_bgsul(pretrained_dir, device="auto"):
    """Load exact BGSUL architecture and pretrained encoder/critic weights."""
    # Configure TensorFlow device directly
    if device.lower() == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif device.lower() in ("gpu", "cuda"):
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("GPU requested, but TensorFlow detected no GPUs.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    elif device.lower() == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        raise ValueError(f"Unsupported device: {device}")
    p = Path(pretrained_dir)
    for name in ["config.json", "vocabulary.json", "encoder.weights.h5", "critic.weights.h5"]:
        if not (p / name).exists():
            raise FileNotFoundError(f"Missing {p/name}; run BGSUL first (critic weights required for AFBL).")
    cfg_json = json.loads((p / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((p / "vocabulary.json").read_text(encoding="utf-8"))
    max_len = int(cfg_json.get("max_len", 64)); vocab_size = int(cfg_json.get("vocab_size", 16000))
    dim = int(cfg_json.get("dim", cfg_json.get("embedding_dim", 192)))
    heads = 4 if dim % 4 == 0 else 1
    vectorizer = layers.TextVectorization(max_tokens=vocab_size, output_mode="int",
        output_sequence_length=max_len, standardize="lower_and_strip_punctuation")
    vectorizer.set_vocabulary(vocab[2:])
    encoder = bgsul.TextEncoder(vocab_size, max_len, dim, heads, dim * 2, 0.1)
    critic = bgsul.Critic(dim)
    dummy = tf.zeros([2, max_len], tf.int32)
    z = encoder(dummy); critic(z)
    encoder.load_weights(str(p / "encoder.weights.h5"))
    critic.load_weights(str(p / "critic.weights.h5"))
    target = bgsul.Critic(dim); target(z); target.set_weights(critic.get_weights())
    print(f"Loaded BGSUL encoder+critic: dim={dim}, max_len={max_len}, vocab={len(vocab)}")
    return encoder, critic, target, vectorizer, cfg_json, vocab


def encode_corpus(encoder, vectorizer, sentences, batch_size=256):
    out=[]
    for i in range(0, len(sentences), batch_size):
        ids=vectorizer(tf.constant([str(s) for s in sentences[i:i+batch_size]]))
        out.append(encoder(ids, training=False))
    return tf.concat(out, axis=0)


def sample_indices(n, batch_size):
    if n <= 0: raise ValueError("No candidate boundaries.")
    k=min(n, batch_size)
    return tf.random.uniform([k], minval=0, maxval=n, dtype=tf.int32)


def train_afbl(args):
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    encoder, bellman, target, vectorizer, cfg, vocab = load_bgsul(args.pretrained_dir, args.device)
    # Reuse BGSUL corpus loader and sentence/document handling.
    corpus_args = argparse.Namespace(file=args.file, demo=args.demo,
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        streaming=args.streaming, max_documents=args.max_documents,
        min_document_chars=args.min_document_chars)
    sentences, doc_ids, titles = bgsul.load_corpus(corpus_args)
    if args.max_sentences > 0 and len(sentences) > args.max_sentences:
        sentences=sentences[:args.max_sentences]; doc_ids=doc_ids[:args.max_sentences]; titles=titles[:args.max_sentences]
    if len(sentences) < 4: raise ValueError("AFBL requires >=4 sentences.")
    dim=int(cfg.get("dim", cfg.get("embedding_dim",192)))
    generator=FrontierGenerator(4*dim, args.hidden)
    discriminator=FrontierDiscriminator(4*dim+3, args.hidden)
    # Build all heads.
    z0=encode_corpus(encoder, vectorizer, sentences[:4])
    f0=make_pair_features(z0); generator(f0); discriminator(tf.concat([f0, tf.zeros([3,3])], -1))
    opt_g=tf.keras.optimizers.Adam(args.lr_generator, beta_1=0.0, beta_2=0.9)
    opt_d=tf.keras.optimizers.Adam(args.lr_discriminator, beta_1=0.0, beta_2=0.9)
    opt_ev=tf.keras.optimizers.Adam(args.lr_encoder)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path=os.path.join(args.output_dir,"training_log.jsonl")
    if os.path.exists(log_path): os.remove(log_path)

    for epoch in range(1,args.epochs+1):
        # Refresh embeddings/evidence each epoch. Evidence is detached from all adversarial updates.
        z=encode_corpus(encoder, vectorizer, sentences)
        vals=tf.reshape(bellman(z, training=False),[-1])
        pair=make_pair_features(z)
        evidence, raw_evidence=evidence_from_embeddings(z, vals, args.window)
        # Discriminator features: pair features plus evidence scalars (shift, variance, residual).
        d_real=tf.concat([pair, raw_evidence],axis=-1)
        # Candidate generator scores; samples are drawn from the generator's proposal distribution.
        d_losses=[]
        for _ in range(args.n_critic):
            with tf.GradientTape() as tape:
                logits=tf.squeeze(generator(pair,training=True),-1)
                probs=tf.nn.softmax(logits / args.generator_temperature)
                ids=sample_indices(tf.shape(pair)[0], args.batch_size)
                fake=tf.gather(d_real,ids)
                #real_ids=tf.random.categorical(tf.math.log(evidence[None,:]+1e-8), tf.shape(pair)[0])[0]
                # Perform multinomial sampling on CPU to avoid GPU kernel failure.
                with tf.device("/CPU:0"):
                    probabilities = tf.nn.softmax(
                        tf.math.log(tf.maximum(evidence, 1e-8))
                    )

                    sampled = tf.random.categorical(
                        tf.math.log(probabilities[None, :]),
                        num_samples=args.batch_size,
                        dtype=tf.int32
                    )

                real_ids = tf.reshape(sampled, [-1])

                real=tf.gather(d_real,real_ids)
                d_fake=discriminator(fake,training=True)
                d_real_score=discriminator(real,training=True)
                gp=gradient_penalty(discriminator,real,fake)
                loss_d=tf.reduce_mean(d_fake)-tf.reduce_mean(d_real_score)+args.gp_weight*gp
            grads=tape.gradient(loss_d,discriminator.trainable_variables)
            opt_d.apply_gradients(zip(grads,discriminator.trainable_variables)); d_losses.append(loss_d)
        # Generator: fool D while staying near evidence and preserving proposal diversity.
        with tf.GradientTape() as tape:
            logits=tf.squeeze(generator(pair,training=True),-1)
            probs=tf.nn.softmax(logits / args.generator_temperature)
            d_scores=tf.squeeze(discriminator(d_real,training=False),-1)
            adversarial=-tf.reduce_sum(probs*d_scores)
            evidence_loss=tf.reduce_sum(probs * (1.0-evidence))
            entropy=-tf.reduce_sum(probs*tf.math.log(probs+1e-8))
            # Encourage non-collapse; entropy floor is softly penalized.
            entropy_penalty=tf.square(tf.maximum(args.entropy_floor-entropy,0.0))
            loss_g=adversarial+args.lambda_evidence*evidence_loss+args.lambda_entropy*entropy_penalty
        grads=tape.gradient(loss_g,generator.trainable_variables)
        opt_g.apply_gradients(zip(grads,generator.trainable_variables))

        # BGSUL encoder/critic update: original self-supervised objective on sampled graph pairs.
        # Context pairs are adjacent within the same document. No labels are involved.
        # and the target critic is Polyak-updated after each step.
        order=np.random.permutation(len(sentences)-1)
        ev_losses=[]
        for start in range(0,len(order),args.batch_size):
            ix=order[start:start+args.batch_size]
            if len(ix)<2: continue
            text_a=tf.constant([sentences[i] for i in ix]); text_b=tf.constant([sentences[i+1] for i in ix])
            ids_a=vectorizer(text_a); ids_b=vectorizer(text_b)
            with tf.GradientTape() as tape:
                za=encoder(ids_a,training=True); zb=encoder(ids_b,training=True)
                # Positive pair = adjacent sentences in same document; filter document crossings.
                same=np.asarray([doc_ids[i]==doc_ids[i+1] for i in ix],dtype=np.float32)
                mask=tf.constant(same[:,None])
                cos=tf.reduce_sum(za*zb,axis=-1,keepdims=True)
                inv=tf.reduce_sum(tf.cast(mask,tf.float32)*(1.0-cos))/tf.maximum(tf.reduce_sum(tf.cast(mask,tf.float32)),1.0)
                nce=bgsul.nt_xent(za,zb,args.temperature)
                va=bellman(za,training=True)
                vb=tf.stop_gradient(target(zb,training=False))
                # Contextual transition has zero immediate reward; critic propagates value.
                resid=va-args.gamma*vb
                bell_loss=tf.reduce_mean(bgsul.huber(resid))
                # Frontier representation regularizer: penalize only false smoothness
                # on evidence-positive pairs, using detached pseudo-frontier evidence.
                ev=tf.gather(evidence,ix)[:,None]
                frontier_loss=tf.reduce_mean(ev*(1.0-cos))
                loss_ev=args.lambda_nce*nce+args.lambda_bellman*bell_loss+args.lambda_frontier*frontier_loss+args.lambda_invariance*inv
            variables=encoder.trainable_variables+bellman.trainable_variables
            grads=tape.gradient(loss_ev,variables)
            pairs=[(g,v) for g,v in zip(grads,variables) if g is not None]
            if pairs:
                gg,vv=zip(*pairs); gg,_=tf.clip_by_global_norm(gg,1.0); opt_ev.apply_gradients(zip(gg,vv))
            for tv,ov in zip(target.variables,bellman.variables): tv.assign((1-args.target_tau)*tv+args.target_tau*ov)
            ev_losses.append(float(loss_ev.numpy()))
        rec={"epoch":epoch,"generator_loss":float(loss_g.numpy()),"discriminator_loss":float(tf.reduce_mean(d_losses).numpy()),
             "evidence_mean":float(tf.reduce_mean(evidence).numpy()),"evidence_entropy":float(entropy.numpy()),
             "bgsul_loss":float(np.mean(ev_losses)) if ev_losses else None}
        with open(log_path,"a",encoding="utf-8") as f:f.write(json.dumps(rec)+"\n")
        print("Epoch %03d | G %.4f | D %.4f | BGSUL %.4f | evidence %.3f | entropy %.3f" %
              (epoch,rec["generator_loss"],rec["discriminator_loss"],rec["bgsul_loss"],rec["evidence_mean"],rec["evidence_entropy"]))

    # Final frontier inference.
    z=encode_corpus(encoder,vectorizer,sentences).numpy().astype(np.float32)
    vals=bellman(tf.constant(z),training=False).numpy().reshape(-1)
    pair=make_pair_features(tf.constant(z))
    evidence, raw=evidence_from_embeddings(tf.constant(z),tf.constant(vals),args.window)
    logits=tf.squeeze(generator(pair,training=False),-1)
    probs=tf.nn.softmax(logits/args.generator_temperature).numpy()
    ev=evidence.numpy()
    # Combine generator proposal, detached evidence and discriminator score.
    dfeat=tf.concat([pair,raw],axis=-1)
    ds=tf.squeeze(discriminator(dfeat,training=False),-1).numpy()
    ds=(ds-np.median(ds))/(np.median(np.abs(ds-np.median(ds)))+1e-6)
    scores=args.score_evidence_weight*ev+args.score_generator_weight*probs+args.score_discriminator_weight/(1+np.exp(-ds))
    # Robust normalize to [0,1]; boundary at t is between sentence t and t+1.
    lo,hi=np.quantile(scores,[0.01,0.99]); scores=np.clip((scores-lo)/(hi-lo+1e-8),0,1)
    frontiers=np.where(scores>=args.threshold)[0].tolist()
    # Ensure document boundaries are included, never merge across documents.
    for i in range(len(doc_ids)-1):
        if doc_ids[i]!=doc_ids[i+1] and i not in frontiers: frontiers.append(i)
    frontiers=sorted(set(frontiers))
    units=[]; start=0
    for i in frontiers:
        if i>=start:
            units.append({"doc_id":int(doc_ids[start]),"start":int(start),"end":int(i+1),
                         "text":" ".join(sentences[start:i+1]),"frontier_score":float(scores[i])})
            start=i+1
    if start<len(sentences): units.append({"doc_id":int(doc_ids[start]),"start":start,"end":len(sentences),
        "text":" ".join(sentences[start:]),"frontier_score":None})
    np.savez_compressed(os.path.join(args.output_dir,"embeddings.npz"),embeddings=z,critic_values=vals,
        frontier_scores=scores,frontier_probabilities=probs,evidence=ev,doc_ids=np.asarray(doc_ids,np.int32),
        sentences=np.asarray(sentences,dtype=object))
    with open(os.path.join(args.output_dir,"frontiers.jsonl"),"w",encoding="utf-8") as f:
        for i,s in enumerate(scores):f.write(json.dumps({"index":i,"left":sentences[i],"right":sentences[i+1],
            "score":float(s),"generator_probability":float(probs[i]),"evidence":float(ev[i]),
            "is_frontier":bool(i in frontiers)},ensure_ascii=False)+"\n")
    with open(os.path.join(args.output_dir,"units.jsonl"),"w",encoding="utf-8") as f:
        for u in units:f.write(json.dumps(u,ensure_ascii=False)+"\n")
    encoder.save_weights(os.path.join(args.output_dir,"encoder.weights.h5"))
    bellman.save_weights(os.path.join(args.output_dir,"critic.weights.h5"))
    target.save_weights(os.path.join(args.output_dir,"target_critic.weights.h5"))
    generator.save_weights(os.path.join(args.output_dir,"generator.weights.h5"))
    discriminator.save_weights(os.path.join(args.output_dir,"discriminator.weights.h5"))
    (Path(args.output_dir)/"vocabulary.json").write_text(json.dumps(vocab,ensure_ascii=False),encoding="utf-8")
    cfgout=vars(args).copy();cfgout.update({"dim":dim,"sentence_count":len(sentences),"frontier_count":len(frontiers),
        "unit_count":len(units),"objective":"unsupervised AFBL; no ASSIN labels"})
    (Path(args.output_dir)/"config.json").write_text(json.dumps(cfgout,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Saved AFBL outputs to {args.output_dir}; frontiers={len(frontiers)}, units={len(units)}")
    print("ASSIN labels were not loaded or used.")


def parse_args():
    p=argparse.ArgumentParser(description="AFBL adversarial frontier Bellman learning (TensorFlow; extends BGSUL)")
    p.add_argument("--pretrained-dir",required=True,help="BGSUL output directory containing config/vocabulary/encoder/critic weights")
    p.add_argument("--file",default=None);p.add_argument("--demo",action="store_true")
    p.add_argument("--dataset-name",default="wikimedia/wikipedia");p.add_argument("--dataset-config",default="20231101.pt")
    p.add_argument("--streaming",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--max-documents",type=int,default=5000);p.add_argument("--min-document-chars",type=int,default=200)
    p.add_argument("--max-sentences",type=int,default=50000);p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=64);p.add_argument("--window",type=int,default=5);p.add_argument("--hidden",type=int,default=256)
    p.add_argument("--n-critic",type=int,default=3);p.add_argument("--gp-weight",type=float,default=10.0)
    p.add_argument("--lr-generator",type=float,default=1e-4);p.add_argument("--lr-discriminator",type=float,default=1e-4)
    p.add_argument("--lr-encoder",type=float,default=2e-5);p.add_argument("--generator-temperature",type=float,default=0.7)
    p.add_argument("--lambda-evidence",type=float,default=1.0);p.add_argument("--lambda-entropy",type=float,default=0.05)
    p.add_argument("--entropy-floor",type=float,default=2.0);p.add_argument("--lambda-nce",type=float,default=0.5)
    p.add_argument("--lambda-bellman",type=float,default=0.25);p.add_argument("--lambda-frontier",type=float,default=0.15)
    p.add_argument("--lambda-invariance",type=float,default=0.1);p.add_argument("--gamma",type=float,default=0.95)
    p.add_argument("--temperature",type=float,default=0.15);p.add_argument("--target-tau",type=float,default=0.02)
    p.add_argument("--score-evidence-weight",type=float,default=0.45);p.add_argument("--score-generator-weight",type=float,default=0.35)
    p.add_argument("--score-discriminator-weight",type=float,default=0.20);p.add_argument("--threshold",type=float,default=0.65)
    p.add_argument("--output-dir",default="afbl_output");p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["auto","gpu","cpu"],default="auto")
    return p.parse_args()

if __name__=="__main__":
    train_afbl(parse_args())
