"""Compute FID from a saved checkpoint, with EMA weights or with raw params.

Useful for diagnosing EMA problems: on a short run, `ema_val` tuned for a long run
leaves a large weight on the random initialization (0.9999^7200 = 0.49), so the EMA
weights can be far worse than the raw ones.

Usage:
  python tools/eval_fid.py --config=configs/load_config.py:local_testbed \
      --ckpt=$(pwd)/runs/base_s42 --which=both
"""
import sys
from functools import partial

sys.path.insert(0, ".")

import jax
import numpy as np
from absl import app, flags
from flax import jax_utils
from ml_collections import config_flags

import train as train_mod
from meanflow import MeanFlow
from utils import fid_util, sample_util
from utils.ckpt_util import restore_checkpoint
from utils.vae_util import LatentManager

FLAGS = flags.FLAGS
flags.DEFINE_string("ckpt", None, "Directory holding checkpoint_* (default: config.load_from).")
flags.DEFINE_enum("which", "both", ["ema", "raw", "both"], "Which weights to score.")
config_flags.DEFINE_config_file("config", None, "Config file.", lock_config=True)


def main(argv):
  del argv
  config = FLAGS.config
  ckpt = FLAGS.ckpt or config.load_from
  if ckpt is None:
    raise ValueError("Pass --ckpt or set load_from in the config.")

  model_config = config.model.to_dict()
  model_str = model_config.pop("cls")
  model = MeanFlow(model_str=model_str, model_config=model_config,
                   **config.sampling, **config.method)

  rng = jax.random.key(config.training.seed)
  state = train_mod.create_train_state(rng, config, model, config.dataset.image_size,
                                       lr_value=config.training.learning_rate)
  state = restore_checkpoint(state, ckpt)
  step = int(state.step)
  state = jax_utils.replicate(state)

  latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size,
                                 config.dataset.image_size)
  p_sample_step = jax.pmap(
      partial(train_mod.sample_step, model=model,
              rng_init=jax.random.PRNGKey(config.sampling.seed),
              device_batch_size=config.fid.device_batch_size, config=config),
      axis_name="batch")

  inception_net = fid_util.build_jax_inception()
  stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)
  runner = partial(train_mod.run_p_sample_step, latent_manager=latent_manager)

  modes = ["ema", "raw"] if FLAGS.which == "both" else [FLAGS.which]
  results = {}
  for mode in modes:
    samples = sample_util.generate_fid_samples(
        state, "/tmp", config, p_sample_step, runner, ema=(mode == "ema"))
    mu, sigma = fid_util.compute_stats(samples, inception_net)
    results[mode] = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    print(f"RESULT  ckpt={ckpt}  step={step}  weights={mode}  "
          f"n={samples.shape[0]}  FID={results[mode]:.4f}", flush=True)

  if len(results) == 2:
    print(f"RESULT  delta(ema - raw) = {results['ema'] - results['raw']:+.4f}")


if __name__ == "__main__":
  flags.mark_flags_as_required(["config"])
  app.run(main)
