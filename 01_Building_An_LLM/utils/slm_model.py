"""
We define two specific layers: 
- TokenAndPositionEmbedding: This layer combines token embeddings and positional embeddings. It takes the input tokens, converts them to embeddings, and adds positional information to them.
- TransformerBlock: This layer implements a single transformer block, which consists of multi-head self-attention followed by a feed-forward neural network. It also includes layer normalization and dropout for regularization.
"""

import time
import numpy as np
import tensorflow as tf

# First class
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        """
        Initializes the TokenAndPositionEmbedding layer.
        maxlen: Maximum length of the input sequences.
        vocab_size: Size of the vocabulary (number of unique tokens).
        embed_dim: Dimensionality of the token embeddings.
        """
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        # Defines the token embedding layer 
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # Defines the positional embedding layer
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x): 
        maxlen = tf.shape(x)[-1]
        # Create a range of positions from 0 to maxlen-1 and get their positional embeddings
        positions = tf.range(start=0, limit=maxlen, delta=1)
        # Passing these positions through the positional embedding layer to get the positional embeddings
        positions = self.pos_emb(positions)
        # Token embeddings 
        x = self.token_emb(x)
        # Add the token embeddings and positional embeddings together
        return x + positions
    
# Second class
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        """
        Initializer for the TransformerBlock layer.
        embed_dim: Dimensionality of the input embeddings.
        num_heads: Number of attention heads in the multi-head attention mechanism.
        ff_dim: Dimensionality of the feed-forward network.
        rate: Dropout rate for regularization.
        """
        super(TransformerBlock, self).__init__(**kwargs)
        # Defining the multi-head attention layer
        self.attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),  # First layer of the feed-forward network
            tf.keras.layers.Dense(embed_dim)  # Second layer of the feed-forward network
        ])
        # Normalization layers for the attention and feed-forward outputs
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout layers for regularization
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, training= False): # training false during inference, true during training (for dropout)
        # Multi-head attention
        attention_output = self.attention_layer(inputs, inputs)  # Self-attention
        attention_output = self.dropout1(attention_output, training=training)  # Apply dropout
        attention_output = self.layernorm1(inputs + attention_output)  # Add & Normalize
        # Feed-forward network
        feed_forward_output = self.feed_forward(attention_output)  # Pass through feed-forward network
        feed_forward_output = self.dropout2(feed_forward_output, training=training)  # Apply
        feed_forward_output = self.layernorm2(attention_output + feed_forward_output)  # Add & Normalize
        return feed_forward_output
    
# Taking the same loss from AI foundations course
 
class CustomMaskPadLoss(tf.keras.losses.Loss):
    def __init__(self, pad_token_id,  **kwargs):
        super(CustomMaskPadLoss, self).__init__(name= "custom_mask_pad_loss", **kwargs)
        self.pad_token_id = pad_token_id
    
    def call(self, y_true, y_pred):
        """
        Basically sparse categorical crossentropy loss, but we ignore the padding tokens in the loss calculation.
        """
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                ignore_class=self.pad_token_id,
                                                                reduction="sum_over_batch_size")
        return loss_fn(y_true, y_pred)
    
# The model (Functional API):

def create_slm_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, pad_token_id, rate=0.1):
    """
    Creates the SLM model using the Functional API.
    maxlen: Maximum length of the input sequences.
    vocab_size: Size of the vocabulary (number of unique tokens).
    embed_dim: Dimensionality of the token embeddings.
    num_heads: Number of attention heads in the multi-head attention mechanism.
    ff_dim: Dimensionality of the feed-forward network.
    pad_token_id: ID of the padding token.
    rate: Dropout rate for regularization.
    """
    # Building the model architecture using the Functional API
    inputs = tf.keras.layers.Input(shape=(maxlen,))  # Input layer
    x = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, name="token_and_position_embedding")(inputs)  # Token and positional embeddings
    x = TransformerBlock(embed_dim, num_heads, ff_dim, rate, name="transformer_block")(x)  # Transformer block
    outputs = tf.keras.layers.Dense(vocab_size, activation='linear')(x)  # Output logits for each token position
    model = tf.keras.Model(name="slm_model" , inputs=inputs, outputs=outputs)  # Create the model
    # Compiling the model : 
    loss_function = CustomMaskPadLoss(pad_token_id=pad_token_id)  # Assuming 0 is the padding token ID
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4,
                                          weight_decay=0.005,
                                          gradient_accumulation_steps=None)  # Using AdamW optimizer
    model.compile(optimizer=optimizer, loss=loss_function)
    return model

# Text generation class (reusable outside of callbacks)

class TextGenerator:
    def __init__(self, model, tokenizer, temperature=0.7, top_k=0, top_p=1.0):
        """
        Reusable text generator for an SLM model.
        model: The trained Keras model.
        tokenizer: The tokenizer (SimpleWordTokenizer) used to encode/decode.
        temperature: Controls randomness of sampling. Lower = more deterministic, higher = more random.
        top_k: If > 0, only sample from the top-k most probable tokens.
        top_p: If < 1.0, only sample from the smallest set of tokens whose cumulative probability exceeds top_p (nucleus sampling).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def generate(self, prompt, max_tokens=50):
        """
        Generate text from a prompt.
        prompt: The seed string to start generation from.
        max_tokens: Maximum number of tokens to generate.
        Returns: A tuple (generated_text, token_probabilities) where:
            - generated_text: The full generated text as a string.
            - token_probabilities: A list of dicts, one per generated token, each containing:
                - "token": The sampled token string.
                - "token_id": The sampled token ID.
                - "probability": The probability of the sampled token (after temperature/top-k/top-p).
                - "all_probabilities": Dict mapping every token in the vocabulary to its probability.
        """
        # Set a different random seed each call based on current time
        np.random.seed(int(time.time() * 1000) % (2**32))
        # Encode the prompt into token IDs
        token_ids = self.tokenizer.encode(prompt)
        maxlen = self.model.input_shape[1]  # Get the model's expected input length
        token_probabilities = []

        for _ in range(max_tokens):
            # Pad/truncate to model's input length
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [token_ids], maxlen=maxlen, padding='post',
                truncating='post', value=self.tokenizer.pad_token_id
            )
            # Get model predictions (logits)
            logits = self.model.predict(padded, verbose=0)
            # Get logits for the next token position (length of current sequence)
            next_token_logits = logits[0, len(token_ids) - 1, :]
            # Apply temperature scaling
            scaled_logits = next_token_logits / self.temperature
            # Convert to probabilities via softmax
            probabilities = tf.nn.softmax(scaled_logits).numpy()

            # Apply top-k filtering
            if self.top_k > 0:
                top_k_indices = np.argsort(probabilities)[-self.top_k:]
                mask = np.zeros_like(probabilities)
                mask[top_k_indices] = probabilities[top_k_indices]
                probabilities = mask

            # Apply top-p (nucleus) filtering
            if self.top_p < 1.0:
                sorted_indices = np.argsort(probabilities)[::-1]
                sorted_probs = probabilities[sorted_indices]
                cumulative_probs = np.cumsum(sorted_probs)
                # Find the cutoff index where cumulative probability exceeds top_p
                cutoff_index = np.searchsorted(cumulative_probs, self.top_p) + 1
                # Zero out tokens beyond the cutoff
                removed_indices = sorted_indices[cutoff_index:]
                probabilities[removed_indices] = 0.0

            # Re-normalize probabilities
            prob_sum = probabilities.sum()
            if prob_sum > 0:
                probabilities = probabilities / prob_sum
            else:
                # Fallback: uniform over all tokens
                probabilities = np.ones_like(probabilities) / len(probabilities)

            # Sample from the probability distribution
            next_token_id = int(np.random.choice(len(probabilities), p=probabilities))

            # Build full probability map: token string -> probability for every vocab entry
            all_probabilities = {
                self.tokenizer.decode(int(idx)): float(probabilities[idx])
                for idx in range(len(probabilities))
                if probabilities[idx] > 0
            }

            token_probabilities.append({
                "token": self.tokenizer.decode(next_token_id),
                "token_id": next_token_id,
                "probability": float(probabilities[next_token_id]),
                "all_probabilities": all_probabilities,
            })
            token_ids.append(next_token_id)
            # Stop if we hit the padding token
            if next_token_id == self.tokenizer.pad_token_id:
                break

        generated_text = self.tokenizer.decode(token_ids)
        return generated_text, token_probabilities


# Defining a callback function (runs an example generation at the end of each epoch to see how the model is doing)


class TextGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, prompt, tokenizer, max_tokens=50, every_n_epochs=10, temperature=0.7, top_k=0, top_p=1.0):
        """
        Callback that generates text from a prompt every N epochs.
        prompt: The seed string to start generation from.
        tokenizer: The tokenizer (SimpleWordTokenizer) used to encode/decode.
        max_tokens: Maximum number of tokens to generate.
        every_n_epochs: Run generation every N epochs.
        temperature: Controls randomness of sampling. Lower = more deterministic, higher = more random.
        top_k: If > 0, only sample from the top-k most probable tokens.
        top_p: If < 1.0, nucleus sampling — only sample from tokens whose cumulative probability exceeds top_p.
        """
        super(TextGenerationCallback, self).__init__()
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.every_n_epochs = every_n_epochs
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.last_callback_time = None
        self._generator = None

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        # Lazily create the TextGenerator on first use
        if self._generator is None:
            self._generator = TextGenerator(
                self.model, self.tokenizer,
                temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
            )
        generated_text, _ = self._generator.generate(self.prompt, max_tokens=self.max_tokens)
        loss = logs.get("loss", "N/A")
        now = time.time()
        elapsed = now - self.last_callback_time if self.last_callback_time else 0
        self.last_callback_time = now
        total_epochs = self.params.get("epochs", "?")
        print(f"--- Epoch {epoch + 1} / {total_epochs} | Loss: {loss} | Time: {elapsed:.2f}s | Generated text ---")
        print(generated_text)
        print("---")
