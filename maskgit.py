import jax
import jax.numpy as jnp
import flax.linen as nn

LAYERNORM_EPSILON = 1e-12


def truncated_normal(stddev, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return jax.random.truncated_normal(
            key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev
    return init


class Attention(nn.Module):
    hidden_size: int
    hidden_dropout_prob: float
    num_heads: int
    attention_probs_dropout_prob: float

    @nn.compact
    def __call__(self, layer_input, input_mask, train=True):
        attention_mask = nn.make_attention_mask(input_mask, input_mask)
        attention_output = nn.attention.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.attention_probs_dropout_prob,
            deterministic=not train,
            kernel_init=truncated_normal(0.02),
            bias_init=jax.nn.initializers.zeros,
            name='self_attention',
        )(layer_input, attention_mask)

        attention_output = nn.Dropout(rate=self.hidden_dropout_prob)(
            attention_output, deterministic=not train)
        attention_output = nn.LayerNorm(
            epsilon=LAYERNORM_EPSILON, name='attention_output_ln')(
            attention_output + layer_input)

        return attention_output


class MLP(nn.Module):
    input_dim: int
    dropout_prob: float

    @nn.compact
    def __call__(self, x, train=True):
        residual = x
        x = nn.Dense(int(4*self.input_dim), kernel_init=truncated_normal(0.02))(x)
        x = nn.gelu(x)
        x = nn.Dense(self.input_dim, kernel_init=truncated_normal(0.02))(x)
        x = nn.Dropout(self.dropout_prob)(x, deterministic=not train)
        x = nn.LayerNorm(epsilon=LAYERNORM_EPSILON, name='MLP_ln')(x + residual)
        return x


class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, layer_input, input_mask, train=True):
        attn_out = Attention(
            hidden_size=self.input_dim,
            num_heads=self.num_heads,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
        )(layer_input, input_mask, train=train)

        layer_output = MLP(self.input_dim, self.dropout)(attn_out, train=train)
        return layer_output


class MlmLayer(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, last_layer, embeddings):
        mlm_hidden = nn.Dense(
            features=self.hidden_size,
            kernel_init=truncated_normal(0.02),
            name='mlm_dense')(
                last_layer)
        mlm_hidden = nn.gelu(mlm_hidden)
        mlm_hidden = nn.LayerNorm(
            epsilon=LAYERNORM_EPSILON, name='mlm_ln')(
                mlm_hidden)
        output_weights = jnp.transpose(embeddings)
        logits = mlm_hidden @ output_weights
        return logits


class MaskGit(nn.Module):
    num_codebook: int
    n_heads: int = 8
    n_layers: int = 6
    embed_dim: int = 768
    dropout: float = 0.1

    @nn.compact
    def __call__(self, input_ids, train=True):  # x: [N, L]
        input_ids = input_ids.astype('int32')
        N, L = input_ids.shape
        word_embeddings = nn.Embed(self.num_codebook + 2, self.embed_dim,
                                   embedding_init=truncated_normal(0.02), name='word_embeddings')(
            input_ids)  # [N, L, E]
        pos_ids = jnp.expand_dims(jnp.arange(L), axis=0)  # [1, L]
        pos_embed = nn.Embed(L, self.embed_dim,
                             embedding_init=truncated_normal(0.02), name='position_embeddings')(pos_ids)  # [1, L, E]

        input_embeddings = nn.LayerNorm(epsilon=LAYERNORM_EPSILON, name='embeddings_ln')(
            word_embeddings + pos_embed
        )

        input_embeddings = nn.Dropout(self.dropout)(input_embeddings, deterministic=not train)

        layer_input = input_embeddings
        for _ in range(self.n_layers):
            layer_outputs = EncoderBlock(
                input_dim=self.embed_dim,
                num_heads=self.n_heads,
                dropout=self.dropout,
            )(layer_input, input_mask=jnp.ones_like(input_ids, dtype=jnp.int32), train=train)

            layer_input = layer_outputs

        word_embeddings_w = self.variables['params']['word_embeddings']['embedding']  # [C, E]

        logits = MlmLayer(self.embed_dim)(layer_input, word_embeddings_w)

        return logits  # [N, L, C]
