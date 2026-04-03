"""Generate a wav file comparing low / medium / high velocity in sequence.

Usage:
    python examples/gen_velocity_audio.py
Output:
    ./tmp/velocity_comparison.wav  -- three notes played back-to-back:
                                      velocity 10 → 64 → 127
"""

import wave

import numpy as np

from robopianist.music import midi_message, synthesizer

# Middle C (key 39 on 88-key piano = MIDI note 60).
_NOTE = 60
_DURATION = 1.5   # seconds each note is held
_GAP = 0.5        # silence between notes (seconds)
_SAMPLE_RATE = 44100


def make_events(velocity: int, t_start: float):
    return [
        midi_message.NoteOn(note=_NOTE, velocity=velocity, time=t_start),
        midi_message.NoteOff(note=_NOTE, time=t_start + _DURATION),
    ]


def save_wav(path: str, samples: np.ndarray) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(samples.tobytes())
    print(f"Saved: {path}")


if __name__ == "__main__":
    synth = synthesizer.Synthesizer(sample_rate=_SAMPLE_RATE)

    velocities = [10, 64, 127]
    stride = _DURATION + _GAP

    # Build one event list with all three notes placed sequentially.
    events = []
    for i, vel in enumerate(velocities):
        events.extend(make_events(vel, t_start=i * stride))

    raw = synth.get_samples(events, normalize=False)
    synth.stop()

    # Normalize by the shared peak so relative volumes are preserved.
    int16_max = float(np.iinfo(np.int16).max)
    peak = np.abs(raw).max()
    samples = np.array(raw.astype(np.float64) * (int16_max / peak), dtype=np.int16)

    out_path = "./tmp/velocity_comparison.wav"
    save_wav(out_path, samples)
    print(f"Three notes in sequence: vel={velocities[0]} → {velocities[1]} → {velocities[2]}")
    print(f"Play with: aplay {out_path}")
