#!/usr/bin/env python3
"""Report how much of the validate.py tolerance each program actually uses.

For every official case the program is run once and compared against the
expected output with the same rule as validate.py (rtol=1e-3, atol=1e-4).
Besides PASS/FAIL it prints the worst-case tolerance usage

    max |out - exp| / (atol + rtol * |exp|)

which must stay below 1; a value near 1 means the result barely passes.

Usage: tolerance_margin.py PROGRAM [PROGRAM ...]   (run from the repo root)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

CASES = [
    ("default", ["--n_fft", "8192", "--n_mels", "256", "--f_min", "0", "--f_max", "22050"]),
    ("small_mels", ["--n_fft", "8192", "--n_mels", "64", "--f_min", "0", "--f_max", "22050"]),
    ("large_fft", ["--n_fft", "16384", "--n_mels", "256", "--f_min", "0", "--f_max", "22050"]),
    ("narrow_band", ["--n_fft", "16384", "--n_mels", "256", "--f_min", "300", "--f_max", "8000"]),
]
RTOL, ATOL = 1e-3, 1e-4


def main():
    for program in sys.argv[1:]:
        for name, extra in CASES:
            fd, out = tempfile.mkstemp(suffix=".bin")
            os.close(fd)
            try:
                subprocess.run([program, "data/audio.pcm", out, "--sr", "96000",
                                "--hop_length", "128", *extra], check=True)
                got = np.fromfile(out, dtype="<f4").astype(np.float64)
            finally:
                os.unlink(out)
            exp = np.fromfile(f"data/expected/{name}.bin", dtype="<f4").astype(np.float64)
            diff = np.abs(got - exp)
            usage = diff / (ATOL + RTOL * np.abs(exp))
            worst = int(np.argmax(usage))
            print(f"{os.path.basename(os.path.dirname(program)) or program:>10} {name:<12} "
                  f"max_abs={diff.max():.3e} max_usage={usage.max():.4f} "
                  f"(exp={exp[worst]:.4e}) rel_l2={np.linalg.norm(got - exp) / np.linalg.norm(exp):.3e} "
                  f"{'PASS' if usage.max() <= 1.0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
