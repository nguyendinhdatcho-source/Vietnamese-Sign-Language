"""TensorFlow port of the archived Siformer VSL forward path.

All attention layers below are explicit. In particular, encoder ProbSparse is
not tf.keras MultiHeadAttention. See docs/TENSORFLOW_MIGRATION_REPORT.md.
"""
import math

import tensorflow as tf


class RepeatedFrameUniform(tf.keras.initializers.Initializer):
    """Upstream rand(F,1,1).expand(F,1,D).clone(): tied values initially, free thereafter."""
    def __call__(self, shape, dtype=None):
        frame_values = tf.random.uniform((shape[0], 1), 0, 1, dtype=dtype or tf.float32)
        return tf.broadcast_to(frame_values, shape)


def torch_dense(units, activation=None, trainable=True, name=None):
    # Keras lacks an input-dependent bias initializer hook. A custom Dense build
    # below initializes both kernel and bias with PyTorch's fan-in bound.
    return TorchDense(units, activation=activation, trainable=trainable, name=name)


class TorchDense(tf.keras.layers.Dense):
    def __init__(self, units, activation=None, trainable=True, name=None, **kwargs):
        super().__init__(units, activation=activation, trainable=trainable, name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return
        bound = 1.0 / math.sqrt(int(input_shape[-1]))
        self.kernel_initializer = tf.keras.initializers.RandomUniform(-bound, bound)
        self.bias_initializer = tf.keras.initializers.RandomUniform(-bound, bound)
        super().build(input_shape)


def mha_dense(units):
    # PyTorch MultiheadAttention._reset_parameters initializes the combined
    # in_proj_weight (3W,W) with Xavier uniform, and in_proj_bias with zeros.
    bound = math.sqrt(6.0 / (4.0 * units))
    return tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.RandomUniform(-bound, bound),
                                 bias_initializer="zeros")


class ProbSparseAttention(tf.keras.layers.Layer):
    def __init__(self, width, heads, rng, factor=5, **kwargs):
        super().__init__(**kwargs)
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads, self.factor, self.rng = width, heads, factor, rng
        self.query = torch_dense(width)
        self.key = torch_dense(width)
        self.value = torch_dense(width)
        self.output_projection = torch_dense(width)

    def call(self, x, training=False, sample_indices=None):
        # Active PyTorch AttentionLayer casts source to float32 before projection.
        x = tf.cast(x, tf.float32)
        b, length = tf.shape(x)[0], tf.shape(x)[1]
        depth = self.width // self.heads
        def project(layer):
            value = layer(x)
            return tf.transpose(tf.reshape(value, [b, length, self.heads, depth]), [0, 2, 1, 3])
        q, k, v = project(self.query), project(self.key), project(self.value)
        # Upstream uses ceil(log L), factor=5, clipped to L; the *same* random
        # key indices [L,U] are shared by all batch items and all heads.
        sample_k = tf.minimum(length, tf.cast(self.factor * tf.math.ceil(tf.math.log(tf.cast(length, tf.float32))), tf.int32))
        top_u = sample_k
        if sample_indices is None:
            sample_indices = self.rng.uniform([length, sample_k], maxval=length, dtype=tf.int32)
        else:
            sample_indices = tf.convert_to_tensor(sample_indices, tf.int32)
            tf.debugging.assert_equal(tf.shape(sample_indices), [length, sample_k])
        sampled_k = tf.gather(k, sample_indices, axis=2)  # B,H,L,U,D
        sampled_score = tf.reduce_sum(q[:, :, :, None, :] * sampled_k, axis=-1)
        # Intentional upstream denominator L_K, not U=sample_k.
        sparsity = tf.reduce_max(sampled_score, axis=-1) - tf.reduce_sum(sampled_score, axis=-1) / tf.cast(length, sampled_score.dtype)
        _, chosen = tf.math.top_k(sparsity, k=top_u, sorted=False)
        chosen_q = tf.gather(q, chosen, axis=2, batch_dims=2)
        score = tf.matmul(chosen_q, k, transpose_b=True)
        score = score * tf.math.rsqrt(tf.cast(depth, score.dtype))
        attention = tf.nn.softmax(score, axis=-1)
        updates = tf.matmul(attention, v)
        context = tf.broadcast_to(tf.reduce_mean(v, axis=2, keepdims=True), [b, self.heads, length, depth])
        bi = tf.broadcast_to(tf.range(b)[:, None, None], tf.shape(chosen))
        hi = tf.broadcast_to(tf.range(self.heads)[None, :, None], tf.shape(chosen))
        indices = tf.stack([bi, hi, chosen], axis=-1)
        context = tf.tensor_scatter_nd_update(context, tf.reshape(indices, [-1, 3]), tf.reshape(updates, [-1, depth]))
        merged = tf.reshape(tf.transpose(context, [0, 2, 1, 3]), [b, length, self.width])
        return tf.cast(self.output_projection(merged), tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, width, heads, rng, **kwargs):
        super().__init__(**kwargs)
        self.attention = ProbSparseAttention(width, heads, rng)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.linear1 = torch_dense(2048, activation="relu")
        self.dropout_ff = tf.keras.layers.Dropout(0.1)
        self.linear2 = torch_dense(width)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False, sample_indices=None):
        attn = self.attention(x, training=training, sample_indices=sample_indices)
        x = self.norm1(x + tf.cast(self.dropout1(attn, training=training), x.dtype))
        ff = self.linear2(self.dropout_ff(self.linear1(x), training=training))
        return self.norm2(x + tf.cast(self.dropout2(ff, training=training), x.dtype))


class IsolatedEncoder(tf.keras.layers.Layer):
    def __init__(self, width, heads, classes, frames, layers, rng, **kwargs):
        super().__init__(**kwargs)
        self.layers_list = []
        for i in range(layers):
            layer = EncoderLayer(width, heads, rng, name=f"encoder_layer_{i}")
            setattr(self, f"encoder_layer_{i}", layer)
            self.layers_list.append(layer)
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        # The active PyTorch wrapper freezes these random inner classifiers and
        # projections. There is no auxiliary CE or optimizer update for them.
        self.inner_classifiers = []
        for i in range(layers):
            classifier = torch_dense(classes, trainable=False, name=f"inner_{i}")
            setattr(self, f"inner_{i}", classifier)
            self.inner_classifiers.append(classifier)
        self.projections = []
        for i in range(layers):
            projection = torch_dense(1, trainable=False, name=f"projection_{i}")
            setattr(self, f"projection_{i}", projection)
            self.projections.append(projection)
        self.frames = frames

    def build(self, input_shape):
        for classifier, projection in zip(self.inner_classifiers, self.projections):
            classifier.build((None, self.frames, input_shape[-1]))
            projection.build((None, 100, self.frames))
        super().build(input_shape)

    def call(self, x, training=False, patience=0, samples=None):
        previous, stable, used = None, 0, 0
        for i, layer in enumerate(self.layers_list):
            selection = None if samples is None else samples[i]
            x = layer(x, training=training, sample_indices=selection)
            used += 1
            if patience and not training:
                h = self.final_norm(x)
                # B,F,C -> B,C,F -> B,C -> B
                scores = self.projections[i](tf.transpose(self.inner_classifiers[i](h), [0, 2, 1]))[:, :, 0]
                predicted = tf.argmax(scores, axis=-1, output_type=tf.int32)
                stable = stable + 1 if previous is not None and bool(tf.reduce_all(tf.equal(predicted, previous))) else 0
                previous = predicted
                if stable >= patience:
                    break
        return self.final_norm(x), used


class CrossAttention(tf.keras.layers.Layer):
    """PyTorch nn.MultiheadAttention cross-attention, query length one."""
    def __init__(self, width=108, heads=9, **kwargs):
        super().__init__(**kwargs)
        self.width, self.heads = width, heads
        self.query = mha_dense(width)
        self.key = mha_dense(width)
        self.value = mha_dense(width)
        self.projection = torch_dense(width)
        self.attention_dropout = tf.keras.layers.Dropout(0.1)

    def call(self, query, memory, training=False):
        b, t = tf.shape(query)[0], tf.shape(memory)[1]
        d = self.width // self.heads
        q = tf.transpose(tf.reshape(self.query(query), [b, 1, self.heads, d]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(self.key(memory), [b, t, self.heads, d]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(self.value(memory), [b, t, self.heads, d]), [0, 2, 1, 3])
        score = tf.matmul(q, k, transpose_b=True)
        score = score * tf.math.rsqrt(tf.cast(d, score.dtype))
        weight = self.attention_dropout(tf.nn.softmax(score, -1), training=training)
        context = tf.matmul(weight, v)
        return self.projection(tf.reshape(tf.transpose(context, [0, 2, 1, 3]), [b, 1, self.width]))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, width=108, **kwargs):
        super().__init__(**kwargs)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.cross_attention = CrossAttention(width, 9)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.linear1 = torch_dense(2048, activation="relu")
        self.dropout_ff = tf.keras.layers.Dropout(0.1)
        self.linear2 = torch_dense(width)
        self.dropout3 = tf.keras.layers.Dropout(0.1)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, query, memory, training=False):
        # Vendored DecoderLayer has this unusual first residual and no self-attn.
        query = self.norm1(query + tf.cast(self.dropout1(query, training=training), query.dtype))
        attn = self.cross_attention(query, memory, training=training)
        query = self.norm2(query + tf.cast(self.dropout2(attn, training=training), query.dtype))
        ff = self.linear2(self.dropout_ff(self.linear1(query), training=training))
        return self.norm3(query + tf.cast(self.dropout3(ff, training=training), query.dtype))


class SiformerVSL(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        if cfg["num_classes"] != 100 or cfg["frames"] < 2:
            raise ValueError("Expected 100 classes and frames>=2")
        self.frames = cfg["frames"]
        self.depth = cfg["model"]["encoder_layers"]
        self.decoder_depth = cfg["model"]["decoder_layers"]
        self.attention_rng = tf.random.Generator.from_seed(cfg["seed"] + 17)
        # Equal initial values within each frame; *independent trainable* widths
        # afterward, matching the upstream expanded clone Parameter.
        self.position_left = self.add_weight(name="position_left", shape=(self.frames, 42), initializer=RepeatedFrameUniform())
        self.position_right = self.add_weight(name="position_right", shape=(self.frames, 42), initializer=RepeatedFrameUniform())
        self.position_body = self.add_weight(name="position_body", shape=(self.frames, 24), initializer=RepeatedFrameUniform())
        self.positional = [self.position_left, self.position_right, self.position_body]
        self.class_query = self.add_weight(name="class_query", shape=(1, 1, 108),
                                           initializer=tf.keras.initializers.RandomUniform(0, 1))
        self.left_encoder = IsolatedEncoder(42, 3, 100, self.frames, self.depth, self.attention_rng, name="left_encoder")
        self.right_encoder = IsolatedEncoder(42, 3, 100, self.frames, self.depth, self.attention_rng, name="right_encoder")
        self.body_encoder = IsolatedEncoder(24, 2, 100, self.frames, self.depth, self.attention_rng, name="body_encoder")
        self.encoders = [self.left_encoder, self.right_encoder, self.body_encoder]
        self.decoders = []
        for i in range(self.decoder_depth):
            decoder = DecoderLayer(name=f"decoder_{i}")
            setattr(self, f"decoder_{i}", decoder)
            self.decoders.append(decoder)
        self.decoder_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = TorchDense(100, dtype="float32")
        self.patience = 0
        self.exit_counts = []

    def build_exit_heads(self):
        """Materialize frozen early-exit variables even in static training."""
        for encoder, width in zip(self.encoders, (42, 42, 24)):
            hidden = tf.zeros((1, self.frames, width), tf.float32)
            for classifier, projection in zip(encoder.inner_classifiers, encoder.projections):
                projection(tf.transpose(classifier(hidden), [0, 2, 1]))

    def _forward_batch(self, x, training, patience, samples=None):
        memory, exits = [], []
        for i, (part, encoder, pos) in enumerate(zip((x[:, :, :21], x[:, :, 21:42], x[:, :, 42:]),
                                                       self.encoders, self.positional)):
            flat = tf.reshape(part, [tf.shape(x)[0], self.frames, [42, 42, 24][i]])
            branch_samples = None if samples is None else samples[i]
            encoded, used = encoder(flat + tf.cast(pos[None, :, :], flat.dtype),
                                    training=training, patience=patience, samples=branch_samples)
            memory.append(encoded)
            exits.append(used)
        joined = tf.concat(memory, axis=-1)
        query = tf.broadcast_to(tf.cast(self.class_query, joined.dtype), [tf.shape(x)[0], 1, 108])
        for decoder in self.decoders:
            query = decoder(query, joined, training=training)
        query = self.decoder_norm(query)
        logits = self.head(query)[:, 0, :]
        return logits, exits

    def call(self, coords, training=False, samples=None):
        coords = tf.cast(tf.convert_to_tensor(coords), tf.float32)
        tf.debugging.assert_rank(coords, 4)
        tf.debugging.assert_equal(tf.shape(coords)[1:], [self.frames, 54, 2])
        tf.debugging.assert_all_finite(coords, "Nonfinite Siformer input")
        patience = 0 if training else self.patience
        self.exit_counts = []
        if patience:
            results = []
            for item in tf.unstack(coords, axis=0):
                logits, exits = self._forward_batch(item[None], training=False, patience=patience, samples=samples)
                results.append(logits)
                self.exit_counts.append(exits)
            output = tf.concat(results, axis=0)
        else:
            output, exits = self._forward_batch(coords, training=training, patience=0, samples=samples)
            self.exit_counts = [exits] * (coords.shape[0] or 1)
        tf.debugging.assert_all_finite(output, "Nonfinite Siformer logits")
        return output
