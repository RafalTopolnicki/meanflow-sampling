# Reproducing MeanFlow on a single NVIDIA GPU

Notes from reproducing [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)
(Geng, Deng, Bai, Kolter, He — arXiv 2505.13447) on one RTX 4090.

The upstream repo is written for TPU pods. Everything here is on branch `repro-gpu`;
`git diff main` shows exactly what deviates from upstream (29 lines, all environmental —
nothing touches the method).

## Results

| What | Reference | This machine |
|---|---|---|
| MF-B/4 sanity check, 1-NFE FID @ 50k | 11.4 (README, TPU) / 11.29 (PR #5, GPU) | **11.306** |
| Training throughput, DiT-B/4, batch 256, fp32 | — | **2.55 steps/s** |
| VAE encoding throughput (256px → 32x32x4) | — | **113 img/s** |

Hardware: 1x RTX 4090 (24 GB), 14 cores, 51 GB RAM.

## Environment

Upstream `install.sh` is TPU-only. Use the GPU recipe from
[PR #5](https://github.com/Gsunshine/meanflow/pull/5), pinned to the JAX version it validated:

```bash
conda create -y -n meanflow python=3.11        # TF 2.15 does not support 3.12
conda activate meanflow
# order matters: TF first, then override ml-dtypes/tensorstore, then jax last
pip install pillow clu "tensorflow==2.15.0" "keras<3" "torch<=2.4" torchvision \
            tensorflow_datasets "matplotlib==3.9.2"
pip install "orbax-checkpoint==0.4.4" "ml-dtypes==0.5.0" "tensorstore==0.1.67"
pip install "diffusers==0.29.2" dm-tree cached_property gdown scipy
pip install -U "jax[cuda12]==0.6.2"
```

Resulting versions: jax 0.6.2, flax 0.10.4, optax 0.2.5, numpy 1.26.4, torch 2.4.0+cu121,
tensorflow 2.15.0, diffusers 0.29.2.

pip warns that torch 2.4 wants `nvidia-cudnn-cu12==9.1.0.70` while jax installs 9.26.
Harmless here — torch is only used for data loading and CPU-side resizing.

`diffusers` is pinned because `FlaxAutoencoderKL` is needed; newer releases drop Flax support.

## Patches applied (branch `repro-gpu`)

| File | Change | Why |
|---|---|---|
| `main.py`, `prepare_dataset.py` | `jax.distributed.initialize()` behind `MEANFLOW_MULTIHOST=1` | unconditional multi-host init fails on one GPU |
| `train.py` | skip building the training dataloader when `eval_only` | otherwise eval demands the 1.28M-file latent dataset just to compute `steps_per_epoch` |
| `train.py`, `utils/ema_util.py` | `jax.tree_leaves`/`jax.tree_map` -> `jax.tree_util.*` | removed in JAX >= 0.6 |
| `meanflow.py` | `jnp.clip(t-r, a_min=, a_max=)` -> positional | kwargs removed in JAX >= 0.6 |
| `utils/vae_util.py` | `cost_analysis()` handles dict and list | returns a list on jax<=0.4, a dict on jax>=0.5 |

New files: `configs/eval_b4.yml`, `configs/smoke_train.yml`, `scripts/launch_eval_gpu.sh`,
`tools/check_identity.py`, `tools/sample_grid.py`.

## Gotchas (single-GPU)

**1. `fid.device_batch_size` must be a multiple of 16 when `sample_on_training: True`.**
`utils/vis_util.py:14` asserts `n % (col*row) == 0` with `grid=4`, and the fallback at line 11
assumes at least `col*row*max_bz = 128` images to trim from. On a TPU host that holds
(8 devices x 20 = 160 -> trimmed to 128); on one GPU you get `device_batch_size x 1`.
Never fires under `eval_only`, which skips visualization entirely.

**2. Track `v_loss`, not `loss`.** With `norm_p=1.0` the adaptive weighting computes
`loss / stop_grad(loss + norm_eps)^p`, so once the raw loss exceeds ~0.01 the logged `loss`
is pinned at 1.0 forever and carries no signal. `v_loss` (plain velocity MSE, monitoring-only)
is the real curve. It is written to TensorBoard but not to stdout. **This applies on every
machine, not just single-GPU.**

**3. Scalars in the event files are tensor summaries**, not `simple_value` — read them with
`tf.make_ndarray(v.tensor)`. `tools/dump_curve.py <workdir>` does this for you and prints `v_loss`
with an ASCII bar chart; `tensorboard --logdir <workdir>` also works.

**4. VAE decoding is the memory bottleneck**, not the DiT. `device_batch_size` drives both
sampling and decoding; at 50 the decoder asked XLA for 141 GiB. 25–32 is right for 24 GB.

**5. The FID stats "zip" from Google Drive is already an `.npz`** (an npz *is* a zip of
`.npy` files). Rename it, do not extract it.

## Reproducing the FID number (no ImageNet needed)

This measures the authors' trained weights; it trains nothing. Real images enter only as
precomputed Inception statistics.

```bash
# assets (~2 GB), kept outside the repo
mkdir -p ../meanflow-assets && cd ../meanflow-assets
gdown 19MR1WLycyqc627gsOJF4yzR4ZkU8fvOu -O mfb4_ckpt.zip    # 1.9 GB checkpoint
gdown 1sESI-bE4SpB0noJ6VQ_OE6XAI_o4c-qK -O fid_stats.npz    # 33 MB, already an npz
unzip -q mfb4_ckpt.zip -d ckpt                              # -> ckpt/checkpoint_1200960
# Inception-v3 FID weights auto-download to $TMPDIR/jax_fid on first run
```

Set `load_from` and `fid.cache_ref` in `configs/eval_b4.yml`, then:

```bash
bash scripts/launch_eval_gpu.sh myrun     # ~11 min -> FID w/ EMA at 50000 samples
```

The checkpoint is `checkpoint_1200960`; 1,281,167 // 256 = 5004 steps/epoch x 240 epochs
= 1,200,960, confirming 240 epochs at batch 256.

What the run does:

```
1. 50,000 Gaussian noise tensors        eps ~ N(0,1), each 32x32x4
2. 50,000 random class labels           uniform over 1000 classes
3. ONE forward pass                     u = u_theta(eps, r=0, t=1, y);  x = eps - u
4. VAE decode                           32x32x4 -> 256x256x3
5. Inception-v3                         -> 2048 features per image
6. mu_g, Sigma_g over the 50,000 features
7. Frechet distance vs downloaded mu_r, Sigma_r
```

Caveat on comparability: the reference statistics were taken on trust from the authors'
Drive link (md5 `84e8422726059c1db567d764f6ce5841`). They were not independently verified
against ImageNet. The README says they follow ADM/guided-diffusion, while
`scripts/prepare_data.sh` would compute stats over **all** 1.28M training images if you
generate your own — those are different references and FID shifts between them. Keep using
this file if you want numbers comparable to 11.4.

## Verifying the math (no checkpoint, no data)

```bash
python tools/check_identity.py      # ~10 s
```

Four independent checks:

1. **MeanFlow Identity** `u = v - (t-r) du/dt` on an analytic field, via RK4 integration.
2. **`jax.jvp` vs central finite differences** — validates the total-derivative trick.
3. **`r == t` degenerates to Flow Matching** — target collapses to `v_g` exactly.
4. **One-step sampler** equals `x = eps - u(eps, 0, 1)`.

Check 1 must run in **float64**. In float32 the RK4 increments over the finite-difference
window (`h*v ~ 3e-8`) fall below the resolution of `z ~ 0.7` and round away to zero, which
silently zeroes the derivative and fails the identity for entirely numerical reasons.

## Training smoke test (~6 min, 326 MB of data)

Uses [Imagenette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) — 9,469
real ImageNet images in 10 classes, already in WNID `ImageFolder` layout.

```bash
curl -sSLO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf imagenette2-320.tgz

python prepare_dataset.py --imagenet_root=./imagenette2-320 --output_dir=./smoke_latents \
       --batch_size=32 --vae_type=mse --image_size=256 \
       --compute_latent=True --compute_fid=False         # ~84 s

# edit dataset.root in configs/smoke_train.yml, then
python main.py --workdir=./smoke_run --config=configs/load_config.py:smoke_train

python tools/sample_grid.py --config=configs/load_config.py:smoke_train \
       --ckpt=./smoke_run --grid=5 --out=samples.png
```

Expected: `v_loss` falls ~6759 -> ~3975 over 780 steps (41%), checkpoints at steps 360/720,
samples are textured noise (correct at 0.06% of a real run — what matters is that they are
not NaN, not uniform grey, not saturated).

Resume is tested by pointing `load_from` at the workdir; a correct restore shows **no
discontinuity** in `v_loss` across the seam (720: 3994.4 -> 740: 3994.6), which is the proof
that params, EMA and Adam state all came back.

## Full training: what it needs

**Data.** ImageNet-1k (ILSVRC-2012) train split, 1,281,167 JPEGs, ~145 GB.
`prepare_dataset.py` center-crops to 256x256 (ADM crop), VAE-encodes, and writes
**one `.pt` per image** holding `(8,32,32)` float32 = latent mean(4) concat std(4), plus the
label. Storing mean *and* std lets training resample the latent each epoch (`cached_encode`).

```
34,206 bytes/file x 1,281,167 = 43.8 GB  and  1.28 million inodes
```

**Time on one RTX 4090**, from the measured 2.55 steps/s:

| Run | Steps | Wall clock |
|---|---|---|
| 240 epochs (the 11.4 model) | 1,200,960 | ~5.5 days |
| 80 epochs (paper Table 1) | 400,320 | ~1.8 days |

Plus ~3.1 h to preprocess (extrapolated from 113 img/s).

Those figures were benchmarked against local-SSD data in page cache. Real training reads
653 separate 34 KB files per second; on network storage the file-per-image layout is likely
to make you I/O-bound rather than GPU-bound. **Sharding the dataset is the first thing to
change for a serious run.**

**Cheaper alternative.** A 100-class subset (~128k images) is 4.4 GB of latents and ~13 h for
240 epochs. Absolute FID will not be comparable to the paper, but relative comparisons
between your own runs are what ablations need.

**CIFAR-10 is not a shortcut in this repo.** `create_cifar_split` exists, but the training
step assumes 8-channel VAE latents (`cached_encode` splits channels into mean/std), so the
CIFAR path needs surgery. The authors' PyTorch repo
[py-meanflow](https://github.com/Gsunshine/py-meanflow) is the better base for CIFAR.

## Suggested first experiment

Reproduce the paper's Table 1(a) ablation rather than chasing 11.35: set
`method.data_proportion: 1.0`, which forces `r == t` always and reduces MeanFlow to plain
Flow Matching. 1-NFE FID should collapse (~300 in the paper) against the 0.75 baseline.
That isolates the single mechanism MeanFlow adds and gives a floor and ceiling to place any
modification between.

Most of the interesting knobs live in `meanflow.py`: `sample_tr()` (the `r,t` coupling and
time distribution), `guidance_fn()` (omega/kappa), and `forward()` (the JVP target and the
adaptive weight).
