import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata


# list of unicode trope symbols, and corresponding names (minus "HEBREW ACCENT")
# includes SOF PASUQ (1473), meaning the :, not the meteg inside the word
trops = [chr(x) for x in range(1425, 1455)] + [chr(1475)]
trop_names = {x: " ".join(unicodedata.name(x).split(" ")[2:]) for x in trops}
trop_names["START"] = "START"
trop_names["END"] = "END"


class BigramAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="bigram_accuracy", **kwargs):
        super(BigramAccuracy, self).__init__(name=name, **kwargs)
        self.correct_bigrams = self.add_weight(name="correct_bigrams", initializer="zeros", dtype=tf.float32)
        self.total_bigrams = self.add_weight(name="total_bigrams", initializer="zeros", dtype=tf.float32)

    def _get_bigrams(self, sequence):
        """Generate bigrams from a tensor of token IDs."""
        bigrams = tf.strings.join([sequence[:, :-1], sequence[:, 1:]], separator="|")
        return bigrams

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_true and y_pred are token sequences in integer format.
        """
        # Ensure the sequences have the same length
        max_len = tf.minimum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        y_true, y_pred = y_true[:, :max_len], y_pred[:, :max_len]

        # Extract bigrams
        true_bigrams = self._get_bigrams(tf.strings.as_string(y_true))
        pred_bigrams = self._get_bigrams(tf.strings.as_string(y_pred))

        # Calculate matching bigrams
        matches = tf.reduce_sum(
            tf.cast(tf.reduce_any(tf.equal(true_bigrams[:, :, tf.newaxis], pred_bigrams[:, tf.newaxis, :]), axis=-1),
                    tf.float32)
        )

        # Count total bigrams
        total = tf.reduce_sum(tf.cast(tf.shape(true_bigrams)[1], tf.float32))

        # Update state
        self.correct_bigrams.assign_add(matches)
        self.total_bigrams.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.correct_bigrams, self.total_bigrams)

    def reset_state(self):
        self.correct_bigrams.assign(0.0)
        self.total_bigrams.assign(0.0)


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
