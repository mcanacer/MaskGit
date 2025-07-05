import sys
import importlib

import jax
import jax.numpy as jnp
import optax
import operator
from flax import serialization

import numpy as np


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def make_update_fn(*, apply_fn, optimizer):
    def update_fn(params, opt_state, images):
        def loss_fn(params):
            images_recon, quantized_latents, commitment_loss, embedding_loss = apply_fn(
                params,
                images,
            )

            recon_loss = jnp.mean((images_recon - images) ** 2)

            losses = recon_loss, commitment_loss, embedding_loss

            loss = jax.tree_util.tree_reduce(operator.add, losses)

            return loss, losses

        ((loss, losses), grad) = jax.value_and_grad(loss_fn, has_aux=True)(params)

        loss, losses, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, losses, grad),
        )

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, losses

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path, args):
    evy = get_everything(config_path, args)

    seed = evy['seed']

    train_loader = evy['train_loader']

    loader_iter = iter(train_loader)
    inputs = next(loader_iter)

    model = evy['model']
    optimizer = evy['optimizer']
    epochs = evy['epochs']

    run = evy['run']

    checkpoint_path = evy['checkpoint_path']

    key = jax.random.PRNGKey(seed)

    params = model.init(key, np.array(inputs))

    opt_state = optimizer.init(params)

    devices = jax.local_devices()
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    update_fn = make_update_fn(apply_fn=model.apply, optimizer=optimizer)

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
        for step, images in enumerate(train_loader):
            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)

            (
                params_repl,
                opt_state_repl,
                loss,
                losses,
            ) = update_fn(
                params_repl,
                opt_state_repl,
                images,
            )

            loss = unreplicate(loss)
            losses = unreplicate(losses)

            recon_loss, commitment_loss, embedding_loss = losses

            run.log({
                "reconstruct_loss": recon_loss,
                "commitment_loss": commitment_loss,
                "embedding_loss": embedding_loss,
                "total_loss": loss,
                "epoch": epoch})

        bytes_output = serialization.to_bytes(unreplicate(params_repl))

        with open(checkpoint_path, "wb") as f:
            f.write(bytes_output)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
