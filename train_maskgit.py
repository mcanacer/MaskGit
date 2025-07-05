import sys
import importlib

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from utils import mask_indices, apply_label_smoothing, get_sequences

import numpy as np


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def get_batches(data, batch_size, key):
    indices = jax.random.permutation(key, data.shape[0])
    shuffled = data[indices]
    for i in range(0, len(shuffled), batch_size):
        yield shuffled[i:i + batch_size]


def make_update_fn(*, apply_fn, optimizer):
    def update_fn(params, opt_state, inputs, targets, drop_key):
        def loss_fn(params):
            logits = apply_fn(
                params,
                inputs,
                train=True,
                rngs={'dropout': drop_key}
            )  # [N, L, C]
            num_classes = logits.shape[-1]

            one_hot_labels = jax.nn.one_hot(targets, num_classes)
            smooth_labels = apply_label_smoothing(one_hot_labels, label_smoothing=0.1)
            loss = -jnp.sum(smooth_labels * jax.nn.log_softmax(logits), axis=-1)

            return loss.mean()

        loss, grad = jax.value_and_grad(loss_fn)(params)

        loss, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, grad),
        )

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path, args):
    evy = get_everything(config_path, args)

    seed = evy['seed']

    train_loader = evy['train_loader']

    model = evy['model']
    maskgit = evy['maskgit']
    optimizer = evy['optimizer']
    label_smoothing = evy['label_smoothing']
    epochs = evy['epochs']
    num_codebook = evy['num_codebook']

    run = evy['run']

    checkpoint_path = evy['checkpoint_path']
    transformer_path = evy['transformer_path']
    seq_path = evy['seq_path']

    get_sequences(model, checkpoint_path, train_loader)

    key = jax.random.PRNGKey(seed)
    sub_key, params_key, drop_key = jax.random.split(key, 3)

    params = maskgit.init({'params': params_key, 'dropout': drop_key}, jnp.ones((2, 257), dtype=jnp.int32),
                              train=False)

    sequences = jnp.load(seq_path)

    opt_state = optimizer.init(params)

    devices = jax.local_devices()
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    update_fn = make_update_fn(apply_fn=maskgit.apply, optimizer=optimizer)

    params_repl = replicate(params)
    opt_state_repl = replicate(opt_state)

    del params
    del opt_state

    num_devices = jax.local_device_count()

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (num_devices, n // num_devices, *s))

    def unshard(inputs):
        num_devices, batch_size, *shape = inputs.shape
        return jnp.reshape(inputs, (num_devices * batch_size, *shape))

    for epoch in range(epochs):
        key, subkey, rng = jax.random.split(key, 3)
        for seq in get_batches(sequences, batch_size=512, key=subkey):
            rng, mask_rng, drop_key = jax.random.split(rng, 3)

            inputs, targets = mask_indices(mask_rng, seq, num_codebook)

            inputs = jax.tree_util.tree_map(lambda x: shard(np.array(x)), inputs)
            targets = jax.tree_util.tree_map(lambda x: shard(np.array(x)), targets)

            drop_keys = jax.random.split(drop_key, num_devices)

            (
                params_repl,
                opt_state_repl,
                loss,
            ) = update_fn(
                params_repl,
                opt_state_repl,
                inputs,
                targets,
                drop_keys,
            )

            loss = unreplicate(loss)

            run.log({
                "total_loss": loss,
                "epoch": epoch})

        bytes_output = serialization.to_bytes(unreplicate(params_repl))

        with open(transformer_path, "wb") as f:
            f.write(bytes_output)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
