"""Calibrate the FluidSynth MIDI velocity → loudness curve.

Plays each MIDI velocity (1–127) through FluidSynth for a fixed note,
measures the RMS loudness of each, fits a smoothing spline, and saves the
calibration data to:
    robopianist/music/velocity_calibration.npz

Usage:
    python examples/calibrate_velocity.py
    python examples/calibrate_velocity.py --note 60 --output path/to/out.npz
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline

from robopianist.music import midi_message, synthesizer
from robopianist.music.constants import SAMPLING_RATE

_NOTE_DURATION = 0.8  # seconds each note is held
_DEFAULT_NOTE = 60    # Middle C


def measure_rms(samples_int16: np.ndarray) -> float:
    """Compute RMS on the middle 50 % of the audio to skip attack/release."""
    float_samples = samples_int16.astype(np.float64) / float(np.iinfo(np.int16).max)
    n = len(float_samples)
    segment = float_samples[n // 4 : 3 * n // 4]
    return float(np.sqrt(np.mean(segment ** 2)))


def synthesize_note(
    synth: synthesizer.Synthesizer, note: int, velocity: int, duration: float
) -> np.ndarray:
    events = [
        midi_message.NoteOn(note=note, velocity=velocity, time=0.0),
        midi_message.NoteOff(note=note, time=duration),
    ]
    return synth.get_samples(events, normalize=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--note",
        type=int,
        default=_DEFAULT_NOTE,
        help="MIDI note number to use for calibration (default: 60 = Middle C)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="robopianist/music/velocity_calibration.npz",
        help="Output path for the calibration data (.npz)",
    )
    args = parser.parse_args()

    print(f"Calibrating velocity → loudness curve for MIDI note {args.note}...")
    synth = synthesizer.Synthesizer(sample_rate=SAMPLING_RATE)

    velocities = np.arange(1, 128, dtype=np.int32)
    rms_values = np.zeros(len(velocities), dtype=np.float64)

    for i, vel in enumerate(velocities):
        samples = synthesize_note(synth, args.note, int(vel), _NOTE_DURATION)
        rms_values[i] = measure_rms(samples)
        if vel % 16 == 1:
            print(f"  velocity {vel:3d}/127  RMS = {rms_values[i]:.6f}")

    synth.stop()

    # Fit a cubic smoothing spline over velocity → RMS.
    # Smoothing factor: allow ~5 % of std-dev residual per point.
    s = len(velocities) * (rms_values.std() * 0.05) ** 2
    spline = UnivariateSpline(velocities.astype(float), rms_values, k=3, s=s)
    rms_fitted = np.clip(spline(velocities.astype(float)), 0, None)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        velocities=velocities,
        rms_values=rms_values,
        rms_fitted=rms_fitted,
        note=np.array([args.note]),
    )

    print(f"\nCalibration saved to: {output_path}")
    print(f"  Velocity range : {velocities[0]} – {velocities[-1]}")
    print(f"  RMS range (raw): {rms_values.min():.6f} – {rms_values.max():.6f}")
    print(f"  RMS range (fit): {rms_fitted.min():.6f} – {rms_fitted.max():.6f}")
    print("\nDone. Now use VelocityCalibration.load() in your RL reward.")


if __name__ == "__main__":
    main()
