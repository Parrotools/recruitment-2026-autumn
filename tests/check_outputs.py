#!/usr/bin/env python3
"""Compare the executable with an independent NumPy FFT and Mel formula."""

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np


SYNTHETIC_CASES = [
    {"sr": 8000, "n_fft": 16, "hop_length": 5, "n_mels": 7,
     "f_min": 0.0, "f_max": 4000.0},
    {"sr": 8000, "n_fft": 32, "hop_length": 8, "n_mels": 12,
     "f_min": 100.0, "f_max": 5100.0},
    {"sr": 8000, "n_fft": 64, "hop_length": 19, "n_mels": 17,
     "f_min": 300.0, "f_max": 3900.0},
    {"sr": 8000, "n_fft": 128, "hop_length": 32, "n_mels": 24,
     "f_min": 0.0, "f_max": 6500.0},
    {"sr": 32, "n_fft": 1, "hop_length": 1, "n_mels": 5,
     "f_min": 0.0, "f_max": 28.0},
    {"sr": 32, "n_fft": 2, "hop_length": 1, "n_mels": 7,
     "f_min": 1.0, "f_max": 16.0},
    {"sr": 32, "n_fft": 8, "hop_length": 3, "n_mels": 13,
     "f_min": 0.0, "f_max": 25.0},
]

FORMAL_SHORT_CASES = [
    {"sr": 96000, "n_fft": 8192, "hop_length": 128, "n_mels": 256,
     "f_min": 0.0, "f_max": 22050.0},
    {"sr": 96000, "n_fft": 8192, "hop_length": 128, "n_mels": 64,
     "f_min": 0.0, "f_max": 22050.0},
    {"sr": 96000, "n_fft": 16384, "hop_length": 128, "n_mels": 256,
     "f_min": 0.0, "f_max": 22050.0},
    {"sr": 96000, "n_fft": 16384, "hop_length": 128, "n_mels": 256,
     "f_min": 300.0, "f_max": 8000.0},
]

REFERENCE_CASES = [
    ("default", {"sr": 44100, "n_fft": 2048, "hop_length": 512,
                 "n_mels": 128, "f_min": 0.0, "f_max": 22050.0}),
    ("small_mels", {"sr": 44100, "n_fft": 2048, "hop_length": 512,
                    "n_mels": 32, "f_min": 0.0, "f_max": 22050.0}),
    ("large_fft", {"sr": 44100, "n_fft": 4096, "hop_length": 512,
                   "n_mels": 128, "f_min": 0.0, "f_max": 22050.0}),
    ("narrow_band", {"sr": 44100, "n_fft": 2048, "hop_length": 512,
                     "n_mels": 128, "f_min": 300.0, "f_max": 8000.0}),
]


def dense_filterbank(sr, n_fft, n_mels, f_min, f_max):
    """Build the triangular filters directly from the mathematical formula."""
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    frequencies = np.arange(n_fft // 2 + 1, dtype=np.float64) * sr / n_fft
    hz_points = mel_to_hz(mel_points)
    weights = np.zeros((n_mels, frequencies.size), dtype=np.float64)
    for band in range(n_mels):
        left, center, right = hz_points[band:band + 3]
        rising = (frequencies - left) / (center - left)
        falling = (right - frequencies) / (right - center)
        weights[band] = np.maximum(0.0, np.minimum(rising, falling))
        weights[band] *= 2.0 / (right - left)
    return weights


def expected_spectrogram(samples, params):
    sr = params["sr"]
    n_fft = params["n_fft"]
    hop = params["hop_length"]
    n_frames = 1 + (samples.size - n_fft) // hop
    starts = np.arange(n_frames, dtype=np.int64) * hop
    frames = np.stack([samples[start:start + n_fft] for start in starts])
    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(n_fft, dtype=np.float64) / n_fft
    )
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    weights = dense_filterbank(sr, n_fft, params["n_mels"],
                               params["f_min"], params["f_max"])
    return weights @ power.T


def invoke(program, pcm_path, output_path, params):
    command = [str(program), str(pcm_path), str(output_path)]
    for name, value in params.items():
        command.extend([f"--{name}", str(value)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    expected_size = params["n_mels"] * (
        1 + (pcm_path.stat().st_size // 2 - params["n_fft"]) //
        params["hop_length"]
    )
    actual = np.fromfile(output_path, dtype="<f4")
    if actual.size != expected_size:
        raise AssertionError(
            f"{program}: expected {expected_size} floats, got {actual.size}"
        )
    return actual.reshape(params["n_mels"], -1)


def compare_case(program, pcm_path, params, rtol, atol, expected=None):
    with tempfile.TemporaryDirectory(prefix="melspec-check-") as directory:
        output_path = Path(directory) / "output.bin"
        actual = invoke(program, pcm_path, output_path, params)
    if expected is None:
        samples = np.fromfile(pcm_path, dtype="<i2").astype(np.float64) / 32768.0
        expected = expected_spectrogram(samples, params)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch for {params}: actual={actual.shape}, expected={expected.shape}"
        )
    if not np.all(np.isfinite(actual)):
        raise AssertionError(f"non-finite output for {params}")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    maximum = float(difference.max(initial=0.0))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"output mismatch for {params}: max_abs_error={maximum:.8g}"
        )
    print(f"PASS {params} max_abs_error={maximum:.8g}")


def find_reference_root(path):
    if (path / "audio.pcm").is_file():
        return path
    if (path / "data" / "audio.pcm").is_file():
        return path / "data"
    raise FileNotFoundError(f"could not find audio.pcm under {path}")


def run_synthetic_cases(program, rtol, atol):
    rng = np.random.default_rng(20261029)
    pcm = rng.integers(-32768, 32768, size=521, dtype=np.int16)
    with tempfile.TemporaryDirectory(prefix="melspec-input-") as directory:
        pcm_path = Path(directory) / "synthetic.pcm"
        pcm.astype("<i2").tofile(pcm_path)
        for params in SYNTHETIC_CASES:
            compare_case(program, pcm_path, params, rtol, atol)

        # Exercise the current four parameter sets without allocating the full
        # official input or its roughly 192k output frames.
        short_frames = 9
        for index, params in enumerate(FORMAL_SHORT_CASES):
            sample_count = (
                (short_frames - 1) * params["hop_length"] + params["n_fft"]
            )
            formal_short = rng.integers(
                -32768, 32768, size=sample_count, dtype=np.int16
            )
            short_path = Path(directory) / f"formal-short-{index}.pcm"
            formal_short.astype("<i2").tofile(short_path)
            compare_case(program, short_path, params, rtol, atol)


def run_reference_cases(program, reference_data, rtol, atol):
    data_root = find_reference_root(reference_data)
    pcm_path = data_root / "audio.pcm"
    for name, params in REFERENCE_CASES:
        expected_path = data_root / "expected" / f"{name}.bin"
        raw_expected = np.fromfile(expected_path, dtype="<f4")
        sample_count = pcm_path.stat().st_size // 2
        n_frames = 1 + (sample_count - params["n_fft"]) // params["hop_length"]
        shape = (params["n_mels"], n_frames)
        if raw_expected.size != shape[0] * shape[1]:
            raise AssertionError(
                f"{expected_path}: expected {shape[0] * shape[1]} values, "
                f"got {raw_expected.size}"
            )
        compare_case(program, pcm_path, params, rtol, atol,
                     raw_expected.reshape(shape))
        print(f"PASS reference data: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True, type=Path)
    parser.add_argument("--reference-data", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if not args.program.exists():
        parser.error(f"program does not exist: {args.program}")
    run_synthetic_cases(args.program.resolve(), args.rtol, args.atol)
    if args.reference_data is not None:
        run_reference_cases(args.program.resolve(), args.reference_data.resolve(),
                            args.rtol, args.atol)


if __name__ == "__main__":
    main()
