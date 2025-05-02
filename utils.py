import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata


# list of unicode trope symbols, and corresponding names (minus "HEBREW ACCENT")
# includes SOF PASUQ (1473), meaning the :, not the meteg inside the word
trops = [chr(x) for x in range(1425, 1455)] + [chr(1475)]
trop_names = {x: " ".join(unicodedata.name(x).split(" ")[2:]) for x in trops}
trop_names["START"] = "START"
trop_names["END"] = "END"
# trop_names[","] = ","

# TODO fix bigram accuracy metric
import tensorflow as tf

class BigramAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="bigram_accuracy", **kwargs):
        super(BigramAccuracy, self).__init__(name=name, **kwargs)
        self.correct_bigrams = self.add_weight(name="correct_bigrams", initializer="zeros", dtype=tf.float32)
        self.total_bigrams = self.add_weight(name="total_bigrams", initializer="zeros", dtype=tf.float32)

    def _get_numerical_bigrams(self, sequence):
        """Create numerical bigram representation by combining token pairs."""
        # Ensure int32 for proper tensor operations
        sequence = tf.cast(sequence, tf.int32)

        # Create bigrams by combining token pairs into unique integers
        bigrams = sequence[:, :-1] * 10_000 + sequence[:, 1:]  # Combines tokens into unique pairs
        return bigrams

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state with new batch values."""
        y_true = tf.cast(y_true, tf.int32)

        # Handle probabilistic output
        if len(y_pred.shape) == 3:  # Probabilistic output
            y_pred = tf.argmax(y_pred, axis=-1)

        y_pred = tf.cast(y_pred, tf.int32)

        # Ensure same length for comparison
        max_len = tf.minimum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        y_true, y_pred = y_true[:, :max_len], y_pred[:, :max_len]

        # Extract numerical bigrams
        true_bigrams = self._get_numerical_bigrams(y_true)
        pred_bigrams = self._get_numerical_bigrams(y_pred)

        # Flatten for matching
        true_bigrams = tf.reshape(true_bigrams, [-1])
        pred_bigrams = tf.reshape(pred_bigrams, [-1])

        # Count matching bigrams
        matches = tf.reduce_sum(tf.cast(tf.math.reduce_any(tf.equal(true_bigrams[:, tf.newaxis], pred_bigrams), axis=-1), tf.float32))

        # Count total bigrams
        total = tf.cast(tf.size(true_bigrams, out_type=tf.int32), tf.float32)

        # Update metric state
        self.correct_bigrams.assign_add(matches)
        self.total_bigrams.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.correct_bigrams, self.total_bigrams)

    def reset_state(self):
        self.correct_bigrams.assign(0.0)
        self.total_bigrams.assign(0.0)


def bigram_accuracy(y_true, y_pred):
    """
    Computes the bigram accuracy for predicted sequences.

    Args:
    - y_true: Ground truth labels (batch_size, seq_len)
    - y_pred: Model predictions (batch_size, seq_len, vocab_size) or (batch_size, seq_len)

    Returns:
    - Bigram accuracy as a scalar tensor.
    """
    # Handle probabilistic output
    if len(y_pred.shape) == 3:  # Probabilistic output (batch_size, seq_len, vocab_size)
        y_pred = tf.argmax(y_pred, axis=-1)

    # Ensure int32 for consistency
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    # Ensure same length for comparison
    max_len = tf.minimum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
    y_true, y_pred = y_true[:, :max_len], y_pred[:, :max_len]

    # Create numerical bigrams
    true_bigrams = y_true[:, :-1] * 10_000 + y_true[:, 1:]
    pred_bigrams = y_pred[:, :-1] * 10_000 + y_pred[:, 1:]

    # Flatten for matching
    true_bigrams = tf.reshape(true_bigrams, [-1])
    pred_bigrams = tf.reshape(pred_bigrams, [-1])

    # Count matching bigrams
    matches = tf.reduce_sum(
        tf.cast(tf.math.reduce_any(tf.equal(true_bigrams[:, tf.newaxis], pred_bigrams), axis=-1), tf.float32)
    )

    # Count total bigrams with correct casting
    total = tf.cast(tf.size(true_bigrams, out_type=tf.int32), tf.float32)

    # Compute bigram accuracy
    accuracy = tf.math.divide_no_nan(matches, total)

    return accuracy


def plot_attention_head(in_tokens, translated_tokens, attention):
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(
      labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)


def plot_attention_weights(sentence, translated_tokens, attention_heads, tokenizers):
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.pt.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()
