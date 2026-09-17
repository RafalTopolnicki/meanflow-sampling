"""Print scalar curves from a run's TensorBoard event files.

The repo logs `loss` (pinned at ~1.0 by the adaptive weighting -- no signal) and
`v_loss` (plain velocity MSE -- the curve you actually want). Scalars are stored
as tensor summaries, so they need tf.make_ndarray, not v.simple_value.

Usage:
  python tools/dump_curve.py <workdir> [tag ...]      # default tag: v_loss
"""
import glob
import os
import sys

import tensorflow as tf
from tensorflow.core.util import event_pb2


def read(workdir):
  series = {}
  files = sorted(glob.glob(os.path.join(workdir, "events.out.tfevents.*")))
  if not files:
    sys.exit(f"no event files in {workdir}")
  for f in files:
    for rec in tf.data.TFRecordDataset(f):
      e = event_pb2.Event.FromString(rec.numpy())
      for v in e.summary.value:
        try:
          series.setdefault(v.tag, []).append((e.step, float(tf.make_ndarray(v.tensor))))
        except Exception:
          pass                      # images and other non-scalar summaries
  for pts in series.values():
    pts.sort()
  return series


def main():
  if len(sys.argv) < 2:
    sys.exit(__doc__)
  workdir, tags = sys.argv[1], sys.argv[2:] or ["v_loss"]
  series = read(workdir)
  print(f"tags available: {', '.join(sorted(series))}\n")
  for tag in tags:
    pts = series.get(tag)
    if not pts:
      print(f"--- {tag}: not found ---")
      continue
    print(f"--- {tag} ---")
    lo, hi = min(v for _, v in pts), max(v for _, v in pts)
    for s, v in pts:
      bar = "#" * int(40 * (v - lo) / (hi - lo)) if hi > lo else ""
      print(f"  {s:7d}  {v:12.4f}  {bar}")
    first, last = pts[0][1], pts[-1][1]
    if first:
      print(f"\n  {first:.4g} -> {last:.4g}   ({100*(1-last/first):+.1f}%)\n")


if __name__ == "__main__":
  main()
