# fxdsp_compressor

```python
# fxdsp_compressor.py
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Literal
from jax import lax,jit


# ---------------------------------------------------------------------
# === Detector path (Rectifier + Envelope + Gain Computer) ===
# ---------------------------------------------------------------------

FORWARD_BIAS = 0
REVERSE_BIAS = 1
FULL_WAVE   = 2

PEAK = 0
RMS  = 1

def diode_rectifier_init(bias: int = FULL_WAVE, threshold: float = 0.3) -> Dict:
    state = {
        "bias": jnp.array(bias, dtype=jnp.int32),   # numeric, JAX-safe
        "threshold": 0.0,
        "vt": 0.0,
        "scale": 0.0
    }
    return diode_rectifier_set_threshold(state, threshold)

@jit
def diode_rectifier_set_threshold(state: Dict, threshold: float) -> Dict:
    th = jnp.clip(jnp.abs(threshold), 0.01, 0.9)

    # Extract bias flag
    bias = state["bias"]

    # Reverse bias flips sign
    sign = jnp.where(bias == REVERSE_BIAS, -1.0, 1.0)
    th = th * sign
    sc = (1.0 - jnp.abs(th)) * sign

    # JAX-safe diode equation
    vt = -0.1738 * th + 0.1735
    sc = sc / jnp.exp(1.0 / vt - 1.0)

    # Return a NEW state dict (critical!)
    return {
        "bias": bias,
        "threshold": th,
        "vt": vt,
        "scale": sc,
    }


@jit
def diode_rectifier_process_block(x: jnp.ndarray, state: Dict) -> Tuple[jnp.ndarray, Dict]:
    vt = state["vt"]
    scale = state["scale"]
    bias = state["bias"]

    # Precompute the 3 possible modes
    x_full = jnp.abs(x)
    x_rev  = -x
    x_fwd  = x

    # bias ∈ {0 = forward, 1 = reverse, 2 = full}
    is_full    = bias == FULL_WAVE
    is_reverse = bias == REVERSE_BIAS

    # Select mode with nested where (JAX-safe)
    x_proc = jnp.where(
        is_full,
        x_full,
        jnp.where(is_reverse, x_rev, x_fwd)
    )

    y = jnp.exp((x_proc / vt) - 1.0) * scale
    return y, state



def dynamic_envelope_init(attack: float, release: float, sr: float, mode=RMS) -> Dict:
    atk_coef = jnp.exp(-1.0 / (attack * sr)) if attack > 0 else 0.0
    rel_coef = jnp.exp(-1.0 / (release * sr)) if release > 0 else 0.0
    
    return {
        "attack_coef": atk_coef,
        "release_coef": rel_coef,
        "env": 0.0,
        "mode": mode,
    }

@jit
def dynamic_envelope_step(x: float, state: Dict) -> Tuple[float, Dict]:
    y_prev = state["env"]
    mode = state["mode"]  # 0=peak, 1=rms

    # x_in selection
    x_in = jnp.where(mode == 0, jnp.abs(x), x * x)

    # choose attack or release
    use_rel = x_in <= y_prev
    coef = jnp.where(use_rel, state["release_coef"], state["attack_coef"])

    y = (1 - coef) * x_in + coef * y_prev

    # RMS only
    y = jnp.where(mode == 1, jnp.sqrt(y), y)

    # immutable update
    new_state = dict(state)
    new_state["env"] = y

    return y, new_state

@jit
def dynamic_envelope_process(x: jnp.ndarray, state: Dict) -> Tuple[jnp.ndarray, Dict]:
    def step(carry_state, x_t):
        y_t, new_state = dynamic_envelope_step(x_t, carry_state)
        return new_state, y_t

    final_state, y_block = lax.scan(step, state, x)
    return y_block, final_state


def envelope_detector_init(bias=FULL_WAVE, threshold=0.3, attack=0.01, release=0.1, sr=48000, mode=RMS) -> Dict:
    diode = diode_rectifier_init(bias, threshold)
    env = dynamic_envelope_init(attack, release, sr, mode)
    return {"diode": diode, "env": env, "sr": sr}

@jit
def envelope_detector_process_block(x: jnp.ndarray, state: Dict) -> Tuple[jnp.ndarray, Dict]:
    y_rect, diode_state = diode_rectifier_process_block(x, state["diode"])
    y_env, env_state    = dynamic_envelope_process(y_rect, state["env"])

    new_state = {
        "diode": diode_state,
        "env":   env_state,
        "sr":    state["sr"],
    }
    return y_env, new_state


def compressor_gain_computer(level_db, threshold_db, ratio, knee_db):
    """
    JAX-safe static compressor gain computer (dB-domain).
    """

    # Hard knee gain (used above the soft knee region)
    hard_gain = threshold_db + (level_db - threshold_db) / ratio - level_db

    # Soft knee boundaries
    lower = threshold_db - knee_db / 2.0
    upper = threshold_db + knee_db / 2.0
    delta = level_db - lower

    # Soft knee quadratic curve
    soft_gain = (1.0 / ratio - 1.0) * (delta**2) / (2.0 * knee_db + 1e-12)

    # Region masks
    below     = level_db < lower
    in_knee   = (level_db >= lower) & (level_db <= upper)
    hard_part = level_db > upper

    # Combine them with where()
    gain_db = jnp.where(
        below,
        0.0,
        jnp.where(
            in_knee,
            soft_gain,
            hard_gain
        )
    )

    # Special case: knee = 0 → pure hard knee
    gain_db = jnp.where(
        knee_db == 0.0,
        jnp.where(level_db < threshold_db, 0.0, hard_gain),
        gain_db
    )

    return gain_db


def compressor_detector_init(threshold_db=-20, ratio=4, knee_db=6, attack=0.01, release=0.2, sr=48000, mode=RMS) -> Dict:
    env = envelope_detector_init(FULL_WAVE, 0.3, attack, release, sr, mode)
    return {"env_detector": env, "threshold_db": threshold_db, "ratio": ratio, "knee_db": knee_db}

@jit
def compressor_detector_process_block(
    x: jnp.ndarray,
    state: Dict,
) -> Tuple[jnp.ndarray, Dict, jnp.ndarray]:
    # Envelope detector ------------------------
    env, env_state = envelope_detector_process_block(x, state["env_detector"])

    # Immutable updated state
    new_state = {
        "env_detector": env_state,
        "threshold_db": state["threshold_db"],
        "ratio":        state["ratio"],
        "knee_db":      state["knee_db"],
    }

    # Level in dB
    eps      = 1e-12
    level_db = 20.0 * jnp.log10(jnp.maximum(env, eps))

    # Vectorized gain computer: works on whole level_db array
    gain_db = compressor_gain_computer(
        level_db,
        new_state["threshold_db"],
        new_state["ratio"],
        new_state["knee_db"],
    )

    # dB → linear
    gain_lin = jnp.power(10.0, gain_db / 20.0)

    return gain_lin, new_state, env

# ---------------------------------------------------------------------
# === Full Compressor (audio path) ===
# ---------------------------------------------------------------------
@jit
def compressor_process_block(
    x: jnp.ndarray,
    state: Dict,
    makeup_db: float = 0.0,
    mix: float = 1.0,
) -> Tuple[jnp.ndarray, Dict, jnp.ndarray, jnp.ndarray]:
    gain, state, env = compressor_detector_process_block(x, state)

    makeup_lin = jnp.power(10.0, makeup_db / 20.0)
    y_comp     = x * gain * makeup_lin
    y_mix      = mix * y_comp + (1.0 - mix) * x

    return y_mix, state, env, gain


# ---------------------------------------------------------------------
# === Visualization and Test ===
# ---------------------------------------------------------------------

def plot_compressor_behavior():
    sr = 48000
    t = np.linspace(0, 0.6, int(sr * 0.6))
    amp = np.concatenate(
        [
            np.linspace(0.05, 1.0, int(0.3 * sr)),
            np.linspace(1.0, 0.2, int(0.3 * sr)),
        ]
    )
    x = amp * np.sin(2 * np.pi * 220 * t)

    state = compressor_detector_init(threshold_db=-20, ratio=4, knee_db=6, attack=0.005, release=0.2, sr=sr)
    y, state, env, gain = compressor_process_block(x, state, makeup_db=6.0, mix=1.0)

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, x, color="gray", lw=0.8, label="Input")
    plt.plot(t, env, color="orange", lw=1.5, label="Detector Env")
    plt.title("Compressor Input and Envelope")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, gain, lw=1.5, color="red", label="Gain Reduction (linear)")
    plt.ylabel("Gain")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, x, color="gray", alpha=0.6, label="Input")
    plt.plot(t, y, color="blue", lw=1.2, label="Output (Compressed)")
    plt.title("Compressed Output with Make-up Gain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_compressor_behavior()

### How to Use
"""
import numpy as np

sr = 48000
t = np.linspace(0, 1, sr)
x = np.sin(2*np.pi*220*t) * np.linspace(0.1, 1.0, sr)

state = compressor_detector_init(threshold_db=-18, ratio=4, knee_db=6, attack=0.01, release=0.2, sr=sr)
y, state, env, gain = compressor_process_block(x, state, makeup_db=6.0, mix=0.8)
"""
```
