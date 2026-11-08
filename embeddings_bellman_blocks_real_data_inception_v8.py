import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras import layers, Model

# =========================
# CONFIG
# =========================
SEQ_LEN = 32#64
D_MODEL = 256
N_LAYERS = 4
BATCH_SIZE = 64
PRETRAIN_STEPS = 300
MASK_PROB = 0.15
GAMMA = 0.99
TEMPERATURE = 0.07

PAD_ID = 0
MASK_ID = 1
CLS_ID = 2
SPECIAL_TOKENS = 3

# =========================
# TOKENIZER
# =========================
def build_tokenizer(texts):
    vec = layers.TextVectorization(
        max_tokens=10000 - SPECIAL_TOKENS,
        output_mode="int",
        output_sequence_length=SEQ_LEN - 1,
        standardize="lower_and_strip_punctuation"
    )
    vec.adapt(texts)
    return vec

# =========================
# MLM BATCH
# =========================
def create_mlm_batch(texts, vectorizer):
    x = vectorizer(texts)
    x = tf.cast(x, tf.int32) + SPECIAL_TOKENS

    cls = tf.fill([tf.shape(x)[0], 1], CLS_ID)
    x = tf.concat([cls, x[:, :-1]], axis=1)

    labels = tf.identity(x)

    mask = (tf.random.uniform(tf.shape(x)) < MASK_PROB) & (x > CLS_ID)
    x = tf.where(mask, MASK_ID, x)

    return {"input_ids": x, "labels": labels, "mask": mask}

# =========================
# BELLER BLOCKS
# =========================
def bellman_residue(z):
    td = z[:, :-1] - GAMMA * z[:, 1:]
    td = tf.pad(tf.abs(td), [[0,0],[1,0]])
    return tf.nn.avg_pool1d(td[...,None], 3, 1, "SAME")[...,0]

def bellman_blocks(z, threshold=0.5):
    res = bellman_residue(z)
    boundaries = tf.cast(res > threshold, tf.float32)
    boundaries = tf.concat([tf.ones((tf.shape(z)[0],1)), boundaries[:,1:]], 1)
    ids = tf.cumsum(boundaries, axis=1) - 1
    n_blocks = tf.reduce_max(ids) + 1
    return tf.one_hot(tf.cast(ids, tf.int32), tf.cast(n_blocks, tf.int32))

def pool_blocks(h, mask):
    h = tf.expand_dims(h, 2)
    mask = tf.expand_dims(mask, -1)
    summed = tf.reduce_sum(h * mask, axis=1)
    counts = tf.reduce_sum(mask, axis=1) + 1e-6
    return tf.reduce_mean(summed / counts, axis=1)

# =========================
# ENCODER
# =========================
class BellmanEncoder(Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.token = layers.Embedding(vocab_size, D_MODEL)
        self.pos = layers.Embedding(SEQ_LEN, D_MODEL)

        self.attn = [
            layers.MultiHeadAttention(4, D_MODEL // 4)
            for _ in range(N_LAYERS)
        ]
        self.ffn = [
            tf.keras.Sequential([
                layers.Dense(4 * D_MODEL, activation="gelu"),
                layers.Dense(D_MODEL)
            ]) for _ in range(N_LAYERS)
        ]
        self.norms = [layers.LayerNormalization() for _ in range(2*N_LAYERS)]

        self.value = layers.Dense(1)
        self.sent_proj = layers.Dense(D_MODEL)
        self.final_norm = layers.LayerNormalization()
        self.mlm = layers.Dense(vocab_size)

    def call(self, x, training=False):
        pos = tf.range(tf.shape(x)[1])[None]
        h = self.token(x) + self.pos(pos)

        for i in range(N_LAYERS):
            a = self.attn[i](h, h, training=training)
            h = self.norms[2*i](h + a)
            f = self.ffn[i](h)
            h = self.norms[2*i+1](h + f)

        z = self.value(h)[...,0]
        blocks = bellman_blocks(z)

        block_emb = pool_blocks(h, blocks)
        cls_emb = h[:,0]

        sent = 0.7 * cls_emb + 0.3 * block_emb
        sent = self.final_norm(self.sent_proj(sent))

        return {
            "sent": sent,
            "mlm": self.mlm(h),
            "z": z
        }

# =========================
# LOSSES
# =========================
def mlm_loss(logits, labels, mask):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)

def bellman_loss(z):
    td = z[:, :-1] - GAMMA * z[:, 1:]
    return tf.reduce_mean(tf.square(td))

def info_nce(a, b):
    a = tf.math.l2_normalize(a, axis=1)
    b = tf.math.l2_normalize(b, axis=1)
    logits = tf.matmul(a, b, transpose_b=True) / TEMPERATURE
    labels = tf.range(tf.shape(a)[0])
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    )

# =========================
# PRETRAIN
# =========================
@tf.function
def pretrain_step(model, batch, opt):
    with tf.GradientTape() as tape:
        o1 = model(batch["input_ids"], training=True)
        o2 = model(batch["input_ids"], training=True)

        loss = (
            mlm_loss(o1["mlm"], batch["labels"], batch["mask"])
            + 2.0 * bellman_loss(o1["z"])
            + info_nce(o1["sent"], o2["sent"])
        )

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# =========================
# LOAD SST-2
# =========================
train_ds = tfds.load("glue/sst2", split="train")
val_ds = tfds.load("glue/sst2", split="validation")

train_texts, train_labels = [], []
for x in tfds.as_numpy(train_ds):
    train_texts.append(x["sentence"].decode())
    train_labels.append(int(x["label"]))

val_texts, val_labels = [], []
for x in tfds.as_numpy(val_ds):
    val_texts.append(x["sentence"].decode())
    val_labels.append(int(x["label"]))

vectorizer = build_tokenizer(train_texts)
VOCAB_SIZE = len(vectorizer.get_vocabulary()) + SPECIAL_TOKENS

encoder = BellmanEncoder(VOCAB_SIZE)

opt = tf.keras.optimizers.AdamW(1e-4)

for step in range(PRETRAIN_STEPS):
    batch = np.random.choice(train_texts, BATCH_SIZE)
    loss = pretrain_step(encoder, create_mlm_batch(batch, vectorizer), opt)
    if step % 50 == 0:
        print(f"Pretrain {step}: {loss:.4f}")

# =========================
# CLASSIFIER
# =========================
class Classifier(Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.norm = layers.LayerNormalization()
        self.fc = layers.Dense(256, activation="gelu")
        self.drop = layers.Dropout(0.4)
        self.out = layers.Dense(2)

    def call(self, x, training=False):
        sent = self.encoder(x, training=training)["sent"]
        x = self.drop(self.fc(self.norm(sent)), training=training)
        return self.out(x)

def make_ds(texts, labels):
    x = vectorizer(texts)
    x = tf.cast(x, tf.int32) + SPECIAL_TOKENS
    cls = tf.fill([tf.shape(x)[0],1], CLS_ID)
    x = tf.concat([cls, x[:,:-1]], axis=1)
    return tf.data.Dataset.from_tensor_slices((x, labels)).batch(32)

model = Classifier(encoder)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(2e-5, weight_decay=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(
    make_ds(train_texts, train_labels),
    validation_data=make_ds(val_texts, val_labels),
    epochs=6
)
