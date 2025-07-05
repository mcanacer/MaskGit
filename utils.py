import jax
import jax.numpy as jnp
import math

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np


def mask_by_random_topk(rng, mask_len, probs, temperature=1.0):
    confidence = jnp.log(probs) + temperature * jax.random.gumbel(
        rng, probs.shape)
    sorted_confidence = jnp.sort(confidence, axis=-1)
    cut_off = jnp.take_along_axis(sorted_confidence, mask_len.astype(jnp.int32), axis=-1)
    masking = (confidence < cut_off)
    return masking


def mask_indices(key, indices, num_codebook):
    firs_key, second_key, last_key = jax.random.split(key, 3)

    N, L = indices.shape

    sos_token_id = num_codebook
    mask_token_id = num_codebook + 1

    sos_tokens = sos_token_id * jnp.ones((N, 1))  # [N, 1]

    r = math.floor(gamma_func(jax.random.uniform(firs_key)) * L)
    sample = jax.lax.top_k(jax.random.uniform(second_key, indices.shape), r)[1]  # [N, R]
    mask = jnp.zeros((N, L), dtype=jnp.bool)  # [N, L]
    mask = mask.at[jnp.expand_dims(jnp.arange(N), axis=-1), sample].set(True)  # [N, L]

    masked_indices = mask_token_id * jnp.ones((N, L), dtype=jnp.int32)  # [N, L]
    m_indices = mask * indices + (~mask) * masked_indices  # [N, L]

    m_indices = jnp.concatenate([sos_tokens, m_indices], axis=-1)  # [N, L + 1]
    target_indices = jnp.concatenate([sos_tokens, indices], axis=-1)  # [N, L + 1]

    return m_indices.astype(int), target_indices.astype(int)


def gamma_func(r):
    return jax.lax.cos(r * jnp.pi / 2)


def decode(inputs, rng, tokens_to_logits, mask_token_id, num_iter=12, choice_temperature=1.0):
    num_samples, seq_len = inputs.shape
    unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)

    outputs = jnp.zeros((num_samples, num_iter, seq_len), dtype=jnp.int32)

    curr_ids = inputs  # [N, seq_len]
    for t in range(num_iter):
        logits = tokens_to_logits(curr_ids)  # [N, seq_len, 1026]
        rng, sample_rng = jax.random.split(rng, 2)

        sampled_ids = jax.random.categorical(sample_rng, logits)

        unknown_map = (curr_ids == mask_token_id)  # [N, seq_len]
        sampled_ids = jnp.where(unknown_map, sampled_ids, curr_ids)  # [N, seq_len]

        ratio = 1. * (t + 1) / num_iter
        mask_ratio = gamma_func(ratio * jnp.pi / 2)

        outputs = jax.lax.dynamic_update_slice(
            outputs, jnp.expand_dims(sampled_ids, axis=1), (0, t, 0))

        probs = jax.nn.softmax(logits, axis=-1)  # [N, seq_len, 1024]
        selected_probs = jnp.squeeze(
            jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids, axis=-1), axis=-1), axis=-1
        )

        selected_probs = jnp.where(unknown_map, selected_probs, jnp.inf)

        mask_len = jnp.expand_dims(
            jnp.floor(unknown_number_in_the_beginning * mask_ratio), axis=1)

        mask_len = jnp.maximum(
            1,
            jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len)
        )

        rng, choice_rng = jax.random.split(rng)
        masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                      choice_temperature * (1. - ratio))

        curr_ids = jnp.where(masking, mask_token_id, sampled_ids)  # [N, seq_len, 1024]

    return outputs  # [N, num_iter, seq_len]


def apply_label_smoothing(one_hot_labels, label_smoothing=0.0):
    num_classes = one_hot_labels.shape[-1]
    pos = 1.0 - label_smoothing
    neg = label_smoothing / num_classes
    return pos * one_hot_labels + neg


def get_sequences(model, checkpoint_path, train_loader):

    def make_predict_fn(*, apply_fn, method):
        def predict_fn(params, inputs):
            return apply_fn(params, inputs, method=method)

        return jax.pmap(predict_fn, axis_name='batch', donate_argnums=())

    def shard(x):
        n, *s = x.shape
        return x.reshape((num_devices, n // num_devices, *s))

    def unshard(x):
        d, b, *s = x.shape
        return x.reshape((d * b, *s))

    key = jax.random.PRNGKey(42)
    key, sub_key = jax.random.split(key)

    vqvae_params = model.init(sub_key, jnp.ones((2, 256, 256, 3)))

    with open(checkpoint_path, 'rb') as f:
        vqvae_params = serialization.from_bytes(vqvae_params, f.read())

    devices = jax.local_devices()
    num_devices = len(devices)
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    predict_fn = make_predict_fn(apply_fn=model.apply, method=model.encode)

    params_repl = replicate(vqvae_params)

    all_sequences = []

    for img in train_loader:
        inputs = jax.tree_util.tree_map(lambda x: shard(np.array(x)), img)  # [D, B', H, W, C]
        encoded = predict_fn(params_repl, inputs)  # [D, B', H, W]
        encoded = jax.tree_util.tree_map(lambda x: unshard(np.array(x)), encoded)  # [N, H, W]
        seq = encoded.reshape(encoded.shape[0], -1)  # [N, L]
        all_sequences.append(seq)

    all_sequences = jnp.concatenate(all_sequences, axis=0)  # [N, L]
    jnp.save("celeb_sequences.npy", all_sequences)


def load_params(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params = serialization.from_bytes(None, f.read())
    return params
