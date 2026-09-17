"""Load a checkpoint, generate 1-NFE samples, save a PNG grid.

Independent of the training loop, so it also serves as a standalone check that
a checkpoint restores and samples correctly.

Usage:
  python tools/sample_grid.py --config=configs/load_config.py:smoke_train \
      --ckpt=/path/to/workdir --out=samples.png
"""
import sys

sys.path.insert(0, ".")

import jax
import numpy as np
from absl import app, flags
from flax import jax_utils
from ml_collections import config_flags
from PIL import Image

import train as train_mod
from meanflow import MeanFlow
from utils.ckpt_util import restore_checkpoint
from utils.vae_util import LatentManager

FLAGS = flags.FLAGS
flags.DEFINE_string("ckpt", None, "Directory holding checkpoint_* (default: config.load_from).")
flags.DEFINE_string("out", "samples.png", "Output PNG path.")
flags.DEFINE_integer("grid", 5, "Grid side; grid**2 images are drawn.")
config_flags.DEFINE_config_file("config", None, "Config file.", lock_config=True)


def main(argv):
  del argv
  config = FLAGS.config
  ckpt = FLAGS.ckpt or config.load_from
  if ckpt is None:
    raise ValueError("Pass --ckpt or set load_from in the config.")

  n = FLAGS.grid ** 2
  model_config = config.model.to_dict()
  model_str = model_config.pop("cls")
  model = MeanFlow(model_str=model_str, model_config=model_config,
                   **config.sampling, **config.method)

  rng = jax.random.key(config.training.seed)
  state = train_mod.create_train_state(
      rng, config, model, config.dataset.image_size,
      lr_value=config.training.learning_rate)
  state = restore_checkpoint(state, ckpt)
  print(f"restored at step {int(state.step)}")
  state = jax_utils.replicate(state)

  latent_manager = LatentManager(config.dataset.vae, n, config.dataset.image_size)
  from functools import partial
  p_sample_step = jax.pmap(
      partial(train_mod.sample_step, model=model,
              rng_init=jax.random.PRNGKey(config.sampling.seed),
              device_batch_size=n, config=config),
      axis_name="batch")

  idx = jax.process_index() * jax.local_device_count() + np.arange(jax.local_device_count())
  imgs = train_mod.run_p_sample_step(p_sample_step, state, idx, latent_manager, ema=True)
  imgs = np.asarray(imgs)[:n]                       # (n, H, W, C) uint8

  g, H, W, C = FLAGS.grid, imgs.shape[1], imgs.shape[2], imgs.shape[3]
  canvas = imgs.reshape(g, g, H, W, C).transpose(0, 2, 1, 3, 4).reshape(g * H, g * W, C)
  Image.fromarray(canvas).save(FLAGS.out)
  print(f"wrote {FLAGS.out}  ({g}x{g} grid of {H}x{W})")


if __name__ == "__main__":
  flags.mark_flags_as_required(["config"])
  app.run(main)
