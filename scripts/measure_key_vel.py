"""Measure maximum achievable qvel and QVEL_MIN for black vs white piano keys.

Method: for each key, inject qvel directly and step until activation or key returns.
Find (1) minimum qvel to activate, (2) relationship between injected qvel and
measured qvel at activation moment.
"""

import numpy as np
from dm_control import mjcf
from robopianist.models.piano import piano as piano_mod, piano_constants as consts

BLACK_KEY_PATTERN = (False, True, False, False, True, False, True, False, False, True, False, True)

def is_black(key_id: int) -> bool:
    return BLACK_KEY_PATTERN[key_id % 12]


def measure_min_activation_qvel(key_id: int, n_trials: int = 5) -> float:
    """Binary search for minimum qvel that activates key_id."""
    lo, hi = 0.0, 20.0
    for _ in range(20):
        mid = (lo + hi) / 2
        activated = _try_activate(key_id, mid)
        if activated:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.005:
            break
    return hi


def measure_qvel_at_activation(key_id: int, injected_qvel: float) -> float | None:
    """Returns the measured qvel at the moment of activation, or None if not activated."""
    p = piano_mod.Piano()
    physics = mjcf.Physics.from_mjcf_model(p.mjcf_model)
    p.initialize_episode(physics, random_state=np.random.RandomState(0))

    joint = p.keys[key_id].joint[0]
    joint_id = physics.model.name2id(joint.full_identifier, 'joint')

    # Inject velocity
    physics.data.qvel[joint_id] = injected_qvel

    threshold = physics.model.jnt_range[joint_id][1] - np.deg2rad(0.5)

    for _ in range(200):
        physics.step()
        qpos = physics.data.qpos[joint_id]
        qvel = physics.data.qvel[joint_id]
        if qpos >= threshold:
            return float(qvel)
    return None


def _try_activate(key_id: int, injected_qvel: float) -> bool:
    return measure_qvel_at_activation(key_id, injected_qvel) is not None


# --- Main ---
# Sample keys: 3 white, 3 black across registers
white_keys = [0, 12, 36, 60, 80]   # A0, A1, C4, C6, G#6
black_keys = [1, 13, 37, 61, 81]   # Bb0, Bb1, Db4, Db6, A6

print("=== QVEL_MIN (minimum injection to activate) ===")
results = {}
for key_id in white_keys + black_keys:
    ktype = "BLACK" if is_black(key_id) else "WHITE"
    qvel_min = measure_min_activation_qvel(key_id)
    results[key_id] = qvel_min
    print(f"  key {key_id:3d} ({ktype}): QVEL_MIN = {qvel_min:.4f} rad/s")

white_mins = [results[k] for k in white_keys]
black_mins = [results[k] for k in black_keys]
print(f"\n  White mean QVEL_MIN: {np.mean(white_mins):.4f} ± {np.std(white_mins):.4f}")
print(f"  Black mean QVEL_MIN: {np.mean(black_mins):.4f} ± {np.std(black_mins):.4f}")

print("\n=== qvel AT ACTIVATION for various injection levels ===")
test_qvels = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0]
sample_white = 36  # C4
sample_black = 37  # Db4 (adjacent black key)

print(f"\n  White key {sample_white} vs Black key {sample_black} (adjacent pair):")
print(f"  {'injected':>10s}  {'white_measured':>15s}  {'black_measured':>15s}  {'ratio':>8s}")
for inj in test_qvels:
    wm = measure_qvel_at_activation(sample_white, inj)
    bm = measure_qvel_at_activation(sample_black, inj)
    wm_str = f"{wm:.4f}" if wm is not None else "  (no act)"
    bm_str = f"{bm:.4f}" if bm is not None else "  (no act)"
    ratio = f"{bm/wm:.3f}" if (wm and bm) else "    -"
    print(f"  {inj:10.1f}  {wm_str:>15s}  {bm_str:>15s}  {ratio:>8s}")
