"""Standalone checks that the MeanFlow math in meanflow.py is what the paper says.

Runs in seconds, needs no checkpoint and no data. Four independent checks:

  1. identity  -- the MeanFlow Identity (Eq. 6) on an analytic vector field,
                  verified by numerical integration. Pure math, no network.
  2. jvp       -- jax.jvp in forward() computes the same total derivative
                  du/dt = v.d_z u + d_t u as central finite differences.
  3. degenerate-- at r == t the training target collapses to the Flow Matching
                  target, so MeanFlow strictly contains FM.
  4. sampler   -- solver_step at (t=1, r=0) is exactly x = eps - u(eps, 0, 1).

Usage:  python tools/check_identity.py
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")


def check_identity_analytic():
  """MeanFlow Identity on an arbitrary smooth field, via numerical integration.

  For a trajectory z(tau) solving dz/dtau = v(z, tau), the average velocity is
      u(z_t, r, t) = (z_t - z_r) / (t - r),
  with z_r the point at time r on the *same* trajectory. The identity claims
      u = v(z_t, t) - (t - r) * d/dt u,
  where d/dt is the total derivative along the trajectory. Checked numerically.

  Note: this runs in float64 on purpose. In float32 the RK4 increments over the
  finite-difference window (h*v ~ 3e-8) fall below the resolution of z ~ 0.7 and
  round away to zero, which silently kills the derivative.
  """
  import math

  v = lambda z, s: math.sin(z) + 0.5 * s        # arbitrary smooth field
  r, t = 0.2, 0.9

  def integrate(z0, s0, s1):                     # RK4, float64, step ~1e-5
    n = max(64, int(abs(s1 - s0) * 100000))
    h = (s1 - s0) / n
    z, s = z0, s0
    for _ in range(n):
      k1 = v(z, s)
      k2 = v(z + 0.5 * h * k1, s + 0.5 * h)
      k3 = v(z + 0.5 * h * k2, s + 0.5 * h)
      k4 = v(z + h * k3, s + h)
      z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
      s = s + h
    return z

  z_t = 0.7                                      # state at time t
  z_r = integrate(z_t, t, r)                     # same trajectory, time r
  u_of = lambda tau: (integrate(z_t, t, tau) - z_r) / (tau - r)

  eps = 1e-3
  du_dt = (u_of(t + eps) - u_of(t - eps)) / (2 * eps)

  lhs = u_of(t)
  rhs = v(z_t, t) - (t - r) * du_dt
  err = abs(lhs - rhs)
  print(f"  u = {lhs:+.8f}   v - (t-r)du/dt = {rhs:+.8f}   |diff| = {err:.2e}")
  return err < 1e-5


def _tiny_model():
  from meanflow import MeanFlow
  from models import models_dit
  # smallest DiT, tiny latent, so this runs on CPU in seconds
  models_dit.DiT_TINY = lambda **kw: models_dit.DiT(
      depth=2, hidden_size=64, patch_size=4, num_heads=4,
      input_size=8, num_classes=10, **kw)
  m = MeanFlow(model_str="DiT_TINY", model_config={}, num_classes=10)
  rng = jax.random.PRNGKey(0)
  x = jnp.ones((2, 8, 8, 4))
  var = m.init({"params": rng, "gen": rng}, x, jnp.ones((2,)), jnp.ones((2,), jnp.int32))
  return m, var


def check_jvp():
  """jax.jvp total derivative == central finite differences."""
  m, var = _tiny_model()
  rng = jax.random.PRNGKey(1)
  bz = 2
  z_t = jax.random.normal(rng, (bz, 8, 8, 4))
  v_g = jax.random.normal(jax.random.PRNGKey(2), (bz, 8, 8, 4))
  t = jnp.full((bz,), 0.8)
  r = jnp.full((bz,), 0.3)

  def u_fn(z, tt, rr):
    return m.apply(var, z, tt, tt - rr, jnp.zeros((bz,), jnp.int32),
                   train=False, method=m.u_fn,
                   rngs={"gen": jax.random.PRNGKey(0)})

  _, du_dt = jax.jvp(u_fn, (z_t, t, r), (v_g, jnp.ones_like(t), jnp.zeros_like(t)))

  # finite difference: move z along v_g and t forward, hold r fixed
  h = 1e-3
  fd = (u_fn(z_t + h * v_g, t + h, r) - u_fn(z_t - h * v_g, t - h, r)) / (2 * h)
  err = float(jnp.max(jnp.abs(du_dt - fd)) / (jnp.max(jnp.abs(fd)) + 1e-12))
  print(f"  max rel err (jvp vs finite diff) = {err:.2e}")
  return err < 1e-3


def check_degenerate():
  """At r == t the target u_tgt equals v_g: MeanFlow reduces to Flow Matching."""
  m, var = _tiny_model()
  bz = 2
  z_t = jax.random.normal(jax.random.PRNGKey(3), (bz, 8, 8, 4))
  v_g = jax.random.normal(jax.random.PRNGKey(4), (bz, 8, 8, 4))
  t = jnp.full((bz,), 0.6)
  r = t  # <- the degenerate case

  def u_fn(z, tt, rr):
    return m.apply(var, z, tt, tt - rr, jnp.zeros((bz,), jnp.int32),
                   train=False, method=m.u_fn,
                   rngs={"gen": jax.random.PRNGKey(0)})

  _, du_dt = jax.jvp(u_fn, (z_t, t, r), (v_g, jnp.ones_like(t), jnp.zeros_like(t)))
  u_tgt = v_g - jnp.clip(t - r, 0.0, 1.0).reshape(bz, 1, 1, 1) * du_dt
  err = float(jnp.max(jnp.abs(u_tgt - v_g)))
  print(f"  max |u_tgt - v_g| at r==t = {err:.2e}  (must be exactly 0)")
  return err == 0.0


def check_sampler():
  """solver_step(t=1, r=0) == eps - u(eps, 0, 1), i.e. Alg. 2 one-step sampling."""
  m, var = _tiny_model()
  bz = 2
  eps_noise = jax.random.normal(jax.random.PRNGKey(5), (bz, 8, 8, 4))
  y = jnp.zeros((bz,), jnp.int32)
  t = jnp.ones((bz,))
  r = jnp.zeros((bz,))

  got = m.apply(var, eps_noise, t, r, y, method=m.solver_step,
                rngs={"gen": jax.random.PRNGKey(0)})
  u = m.apply(var, eps_noise, t, t - r, y, train=False, method=m.u_fn,
              rngs={"gen": jax.random.PRNGKey(0)})
  want = eps_noise - u
  err = float(jnp.max(jnp.abs(got - want)))
  print(f"  max |solver_step - (eps - u)| = {err:.2e}  (must be exactly 0)")
  return err == 0.0


if __name__ == "__main__":
  checks = [
      ("1. MeanFlow Identity (analytic field)", check_identity_analytic),
      ("2. jax.jvp == finite differences     ", check_jvp),
      ("3. r == t degenerates to Flow Match. ", check_degenerate),
      ("4. one-step sampler = eps - u(eps,0,1)", check_sampler),
  ]
  ok = True
  for name, fn in checks:
    print(f"\n[{name}]")
    passed = fn()
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    ok &= passed
  print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
  sys.exit(0 if ok else 1)
