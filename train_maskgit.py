import sys
import importlib
import os
from jax._src.api import F
import yaml

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from utils import mask_indices, apply_label_smoothing
import math

import numpy as np


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())

def gamma_func(r):
  return jax.lax.cos(r * jnp.pi / 2)


def make_predict_fn(*, apply_fn, method):
  def predict_fn(params, images):
    indices = apply_fn(
        params,
        images,
        method=method,
    )

    indices = indices.reshape(indices.shape[0], -1)

    return indices

  return jax.pmap(predict_fn, axis_name='batch', donate_argnums=())

def make_update_fn(*, apply_fn, optimizer):
    def update_fn(params, opt_state, inputs, targets, drop_rng):
        def loss_fn(params):

            logits = apply_fn(
                params,
                inputs,
                train=True,
                rngs={'dropout': drop_rng}
            )  # [N, T, C]

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
    maskgit_path = evy['maskgit_path']

    vqgan_params = load_checkpoint(checkpoint_path, None)['ema_params']

    key = jax.random.PRNGKey(seed)
    key, params_key, drop_key = jax.random.split(key, 3)

    maskgit_params = maskgit.init({'params': params_key, 'dropout': drop_key}, jnp.ones((2, 65), dtype=jnp.int32),
                          train=False)

    opt_state = optimizer.init(maskgit_params)

    devices = jax.local_devices()
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    update_fn = make_update_fn(
        apply_fn=maskgit.apply,
        optimizer=optimizer,
    )

    predict_fn = make_predict_fn(
      apply_fn=model.apply,
      method=model.encode,
    )

    vqgan_params_repl = replicate(vqgan_params)
    maskgit_params_repl = replicate(maskgit_params)
    opt_state_repl = replicate(opt_state)

    del vqgan_params
    del maskgit_params
    del opt_state

    num_devices = jax.local_device_count()

    state_template = {
        "params": unreplicate(maskgit_params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "epoch": 0,
    }

    loaded_state = load_checkpoint(maskgit_path, state_template)
    start_epoch = 0
    if loaded_state:
        maskgit_params_repl = replicate(loaded_state['params'])
        opt_state_repl = replicate(loaded_state['opt_state'])
        start_epoch = loaded_state['epoch'] + 1

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (num_devices, n // num_devices, *s))

    def unshard(inputs):
        num_devices, batch_size, *shape = inputs.shape
        return jnp.reshape(inputs, (num_devices * batch_size, *shape))

    for epoch in range(start_epoch, epochs):
        for images in train_loader:
            key, mask_rng, drop_key = jax.random.split(key, 3)

            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)

            indices = predict_fn(vqgan_params_repl, images)

            inputs, targets = mask_indices(mask_rng, unshard(indices), num_codebook)

            inputs = jax.tree_util.tree_map(lambda x: shard(np.array(x)), inputs)
            targets = jax.tree_util.tree_map(lambda x: shard(np.array(x)), targets)

            drop_keys = jax.random.split(drop_key, num_devices)

            (
                maskgit_params_repl,
                opt_state_repl,
                loss,
            ) = update_fn(
                maskgit_params_repl,
                opt_state_repl,
                inputs,
                targets,
                drop_keys,
            )

            loss = unreplicate(loss)

            run.log({
                "total_loss": loss,
                "epoch": epoch})

        save_checkpoint(maskgit_path, {
            "params": unreplicate(maskgit_params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "epoch": epoch,
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])

