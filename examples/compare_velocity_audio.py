"""Compare MIDI reference audio vs. robot-simulated velocity audio.

Loads a MIDI file and renders two versions:
  (a) reference  — original GT MIDI velocities
  (b) robot      — velocities the robot would produce when trying to match GT

Reports per-note and aggregate statistics in both raw-velocity space
and (if velocity_calibration.npz is available) perceptual loudness space.
Two WAV files are saved for direct listening comparison.

The robot modes simulate different tracking behaviours:

  perfect         robot perfectly matches GT velocity (sanity-check baseline;
                  the two WAV files should be identical)          [DEFAULT]
  noise:<sigma>   GT velocity + N(0, sigma) clipped to [1, 127]
                  — represents imperfect RL tracking error
  flat:<V>        every note is played at constant velocity V
                  — represents a robot that completely ignores dynamics
  csv:<path>      load per-note velocities from a CSV with columns:
                    pitch (MIDI), start_time (s), velocity (int)
                  notes are matched by (pitch, nearest start_time)
                  — for actual robot simulation logs

Usage examples:
    # Sanity check: robot perfectly tracks GT → WAVs should be identical
    python examples/compare_velocity_audio.py \\
        --midi robopianist/music/data/rousseau/twinkle-twinkle-trimmed.mid

    # Realistic RL: robot tracks GT with ±15 velocity error
    python examples/compare_velocity_audio.py \\
        --midi robopianist/music/data/rousseau/nocturne-trimmed.mid \\
        --robot noise:15

    # Worst case: robot ignores dynamics completely
    python examples/compare_velocity_audio.py \\
        --midi robopianist/music/data/rousseau/nocturne-trimmed.mid \\
        --robot flat:64

    # Real robot data from a simulation log
    python examples/compare_velocity_audio.py \\
        --midi path/to/song.mid \\
        --robot csv:path/to/robot_velocities.csv

Output (in --outdir, default ./tmp/):
    reference.wav   — rendered with GT MIDI velocities
    robot.wav       — rendered with robot velocities
    comparison.txt  — per-note table and summary statistics
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from note_seq import midi_io

from robopianist.music import midi_message, synthesizer
from robopianist.music.constants import SAMPLING_RATE

_NOTE_CLIP_DURATION = 0.6   # seconds: each note clip for per-note RMS
_NOTE_CLIP_TAIL = 0.3       # seconds: silence after note-off in clip
_MAX_KEY_VEL = 5.0          # rad/s → vel 127 (matches midi_module.py)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class NoteRecord(NamedTuple):
    pitch: int
    velocity: int
    start_time: float
    end_time: float


# ---------------------------------------------------------------------------
# MIDI loading
# ---------------------------------------------------------------------------

def load_notes(midi_path: Path) -> List[NoteRecord]:
    seq = midi_io.midi_file_to_note_sequence(midi_file=midi_path)
    notes = sorted(
        [
            NoteRecord(
                pitch=n.pitch,
                velocity=n.velocity,
                start_time=n.start_time,
                end_time=n.end_time,
            )
            for n in seq.notes
        ],
        key=lambda n: (n.start_time, n.pitch),
    )
    return notes


# ---------------------------------------------------------------------------
# Robot velocity generators
# ---------------------------------------------------------------------------

def _clamp_velocity(v: float) -> int:
    return int(np.clip(round(v), 1, 127))


def make_robot_velocities_flat(notes: List[NoteRecord], flat_vel: int) -> List[int]:
    return [flat_vel] * len(notes)


def make_robot_velocities_noise(
    notes: List[NoteRecord], sigma: float, rng: np.random.Generator
) -> List[int]:
    noise = rng.normal(0, sigma, size=len(notes))
    return [_clamp_velocity(n.velocity + d) for n, d in zip(notes, noise)]


def make_robot_velocities_csv(
    notes: List[NoteRecord], csv_path: Path
) -> List[int]:
    """Match CSV rows to GT notes by (pitch, nearest start_time)."""
    csv_rows: List[Tuple[int, float, int]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append(
                (int(row["pitch"]), float(row["start_time"]), int(row["velocity"]))
            )

    # Group CSV rows by pitch.
    by_pitch: Dict[int, List[Tuple[float, int]]] = {}
    for pitch, t, vel in csv_rows:
        by_pitch.setdefault(pitch, []).append((t, vel))

    robot_vels: List[int] = []
    for note in notes:
        candidates = by_pitch.get(note.pitch, [])
        if not candidates:
            robot_vels.append(64)   # fallback
            continue
        # Pick the candidate with the nearest start_time.
        best_t, best_vel = min(candidates, key=lambda x: abs(x[0] - note.start_time))
        robot_vels.append(best_vel)
    return robot_vels


def parse_robot_mode(
    mode_str: str, notes: List[NoteRecord], rng: np.random.Generator
) -> List[int]:
    if mode_str == "perfect":
        return [n.velocity for n in notes]
    elif mode_str.startswith("flat:"):
        return make_robot_velocities_flat(notes, int(mode_str.split(":", 1)[1]))
    elif mode_str.startswith("noise:"):
        sigma = float(mode_str.split(":", 1)[1])
        return make_robot_velocities_noise(notes, sigma, rng)
    elif mode_str.startswith("csv:"):
        return make_robot_velocities_csv(notes, Path(mode_str.split(":", 1)[1]))
    else:
        raise ValueError(
            f"Unknown --robot mode '{mode_str}'. "
            "Expected: perfect | flat:<V> | noise:<sigma> | csv:<path>"
        )


# ---------------------------------------------------------------------------
# Audio synthesis helpers
# ---------------------------------------------------------------------------

def synthesize_full(
    synth: synthesizer.Synthesizer,
    notes: List[NoteRecord],
    velocities: List[int],
) -> np.ndarray:
    """Render all notes onto a shared timeline, preserving relative timing.

    Shifts all events so that the first note starts at t=0 to avoid leading
    silence when the MIDI file has a non-zero start offset.
    """
    events: List[midi_message.MidiMessage] = []
    for note, vel in zip(notes, velocities):
        events.append(
            midi_message.NoteOn(note=note.pitch, velocity=vel, time=note.start_time)
        )
        events.append(
            midi_message.NoteOff(note=note.pitch, time=note.end_time)
        )
    # Sort by time so the synthesizer sees events in order.
    events.sort(key=lambda e: e.time)
    # Shift to t=0: get_samples treats event_list[0].time as the leading
    # silence offset, so subtract the first event's absolute time.
    t0 = events[0].time
    for e in events:
        e.time -= t0
    return synth.get_samples(events, normalize=False)


def synthesize_note_clip(
    synth: synthesizer.Synthesizer,
    pitch: int,
    velocity: int,
    duration: float,
    tail: float,
) -> np.ndarray:
    """Render a single note in isolation (used for per-note RMS)."""
    events = [
        midi_message.NoteOn(note=pitch, velocity=velocity, time=0.0),
        midi_message.NoteOff(note=pitch, time=duration),
    ]
    # Temporarily: we need a full clip including tail silence.  The synthesizer
    # appends 1 s of silence internally; pass tail via the last event's time.
    events[-1].time = duration  # keep as absolute; get_samples converts to relative
    return synth.get_samples(events, normalize=False)


def measure_rms(samples_int16: np.ndarray) -> float:
    """RMS on the middle 50 % of the buffer (avoids attack / release artefacts)."""
    float_samples = samples_int16.astype(np.float64) / float(np.iinfo(np.int16).max)
    n = len(float_samples)
    segment = float_samples[n // 4 : 3 * n // 4]
    if segment.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(segment**2)))


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# Loudness calibration (optional)
# ---------------------------------------------------------------------------

def load_calibration(calib_path: Path) -> Optional[np.ndarray]:
    """Return rms_fitted[0..126] indexed by velocity-1, or None if not found."""
    if not calib_path.exists():
        return None
    data = np.load(calib_path)
    return data["rms_fitted"]   # shape (127,), index 0 = velocity 1


def loudness(rms_table: np.ndarray, velocity: int) -> float:
    idx = np.clip(velocity - 1, 0, len(rms_table) - 1)
    rms_max = rms_table.max()
    if rms_max == 0:
        return 0.0
    return float(rms_table[idx] / rms_max)   # normalized to [0, 1]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    notes: List[NoteRecord],
    robot_vels: List[int],
    gt_rms: List[float],
    robot_rms: List[float],
    rms_table: Optional[np.ndarray],
    out_txt: Path,
) -> None:
    from robopianist.music.midi_file import midi_number_to_note_name

    has_calib = rms_table is not None
    header_parts = [
        f"{'#':>4}",
        f"{'Note':<6}",
        f"{'GT vel':>6}",
        f"{'Robot vel':>9}",
        f"{'ΔVel':>6}",
        f"{'GT RMS':>8}",
        f"{'Rbt RMS':>8}",
        f"{'ΔRMS':>8}",
    ]
    if has_calib:
        header_parts += [f"{'GT loud':>8}", f"{'Rbt loud':>8}", f"{'ΔLoud':>8}"]
    header = "  ".join(header_parts)
    separator = "-" * len(header)

    lines = [separator, header, separator]

    vel_errors, rms_errors, loud_errors = [], [], []

    for i, (note, rv, gr, rr) in enumerate(
        zip(notes, robot_vels, gt_rms, robot_rms)
    ):
        dv = rv - note.velocity
        dr = rr - gr
        vel_errors.append(abs(dv))
        rms_errors.append(abs(dr))

        row = [
            f"{i+1:>4}",
            f"{midi_number_to_note_name(note.pitch):<6}",
            f"{note.velocity:>6}",
            f"{rv:>9}",
            f"{dv:>+6}",
            f"{gr:>8.5f}",
            f"{rr:>8.5f}",
            f"{dr:>+8.5f}",
        ]
        if has_calib:
            gl = loudness(rms_table, note.velocity)
            rl = loudness(rms_table, rv)
            dl = rl - gl
            loud_errors.append(abs(dl))
            row += [f"{gl:>8.4f}", f"{rl:>8.4f}", f"{dl:>+8.4f}"]

        lines.append("  ".join(row))

    lines.append(separator)
    lines.append("")
    lines.append("Summary")
    lines.append(f"  Notes          : {len(notes)}")
    lines.append(f"  Vel MAE        : {np.mean(vel_errors):.2f}")
    lines.append(f"  Vel RMSE       : {np.sqrt(np.mean(np.array(vel_errors)**2)):.2f}")
    lines.append(f"  RMS MAE        : {np.mean(rms_errors):.6f}")
    if has_calib:
        lines.append(f"  Loudness MAE   : {np.mean(loud_errors):.4f}")
        lines.append(f"  Loudness RMSE  : {np.sqrt(np.mean(np.array(loud_errors)**2)):.4f}")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    out_txt.write_text(text)
    print(f"Report saved to: {out_txt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--midi",
        required=True,
        help="Path to the MIDI file (.mid)",
    )
    parser.add_argument(
        "--robot",
        default="perfect",
        help=(
            "Robot velocity mode: "
            "perfect (default) | flat:<V> | noise:<sigma> | csv:<path.csv>"
        ),
    )
    parser.add_argument(
        "--calib",
        default="robopianist/music/velocity_calibration.npz",
        help="Path to velocity_calibration.npz (optional; enables loudness stats)",
    )
    parser.add_argument(
        "--outdir",
        default="./tmp",
        help="Directory for output WAV files and report (default: ./tmp)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used only in noise mode)",
    )
    parser.add_argument(
        "--max-notes",
        type=int,
        default=None,
        help="Limit analysis to the first N notes (useful for long pieces)",
    )
    args = parser.parse_args()

    midi_path = Path(args.midi)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MIDI: {midi_path}")
    notes = load_notes(midi_path)
    if args.max_notes is not None:
        notes = notes[: args.max_notes]
    print(f"  {len(notes)} notes loaded")

    rng = np.random.default_rng(args.seed)
    robot_vels = parse_robot_mode(args.robot, notes, rng)


    # Load calibration if available.
    calib_path = Path(args.calib)
    rms_table = load_calibration(calib_path)
    if rms_table is not None:
        print(f"Calibration loaded from: {calib_path}")
    else:
        print(f"No calibration file found at {calib_path} — skipping loudness stats")

    # -----------------------------------------------------------------------
    # Per-note RMS via isolated clips
    # -----------------------------------------------------------------------
    print("Synthesizing per-note clips for RMS measurement...")
    synth = synthesizer.Synthesizer(sample_rate=SAMPLING_RATE)
    gt_rms: List[float] = []
    robot_rms: List[float] = []

    for i, (note, rv) in enumerate(zip(notes, robot_vels)):
        duration = min(note.end_time - note.start_time, _NOTE_CLIP_DURATION)
        duration = max(duration, 0.05)  # guard against zero-length notes

        gt_clip = synthesize_note_clip(synth, note.pitch, note.velocity, duration, _NOTE_CLIP_TAIL)
        robot_clip = synthesize_note_clip(synth, note.pitch, rv, duration, _NOTE_CLIP_TAIL)

        gt_rms.append(measure_rms(gt_clip))
        robot_rms.append(measure_rms(robot_clip))

        if (i + 1) % 20 == 0 or (i + 1) == len(notes):
            print(f"  {i+1}/{len(notes)} notes done")

    # -----------------------------------------------------------------------
    # Full-piece rendering for WAV output
    # -----------------------------------------------------------------------
    print("Rendering full-piece WAVs...")
    gt_audio = synthesize_full(synth, notes, [n.velocity for n in notes])
    robot_audio = synthesize_full(synth, notes, robot_vels)
    synth.stop()

    # Normalize both by the GT peak so relative levels are preserved.
    int16_max = float(np.iinfo(np.int16).max)
    gt_peak = np.abs(gt_audio).max()
    if gt_peak > 0:
        scale = int16_max / gt_peak
        gt_audio = np.array(gt_audio.astype(np.float64) * scale, dtype=np.int16)
        robot_audio = np.array(
            np.clip(robot_audio.astype(np.float64) * scale, -int16_max, int16_max),
            dtype=np.int16,
        )

    ref_wav = outdir / "reference.wav"
    robot_wav = outdir / "robot.wav"
    save_wav(ref_wav, gt_audio, SAMPLING_RATE)
    save_wav(robot_wav, robot_audio, SAMPLING_RATE)
    print(f"Saved: {ref_wav}")
    print(f"Saved: {robot_wav}")
    print(f"Listen: aplay {ref_wav} && aplay {robot_wav}")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n--- Per-note comparison ---\n")
    out_txt = outdir / "comparison.txt"
    print_report(notes, robot_vels, gt_rms, robot_rms, rms_table, out_txt)


if __name__ == "__main__":
    main()
