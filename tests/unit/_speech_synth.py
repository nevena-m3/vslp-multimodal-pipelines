"""Source-filter synthetic DDK and sustained /a/ with known ground truth (test helper)."""
import numpy as np
from scipy import signal

def glottal(n, sr, f0, jitter=0.005, shimmer=0.03, rng=None):
    rng = rng or np.random.default_rng(0)
    out = np.zeros(n); t = 0.0
    while t < n / sr:
        period = 1 / (f0 * (1 + jitter * rng.standard_normal()))
        i = int(t * sr)
        if i < n: out[i] = 1 + shimmer * rng.standard_normal()
        t += period
    # glottal pulse shaping (-12 dB/oct)
    b, a = signal.butter(2, 300 / (sr / 2)); return signal.lfilter(b, a, out) * 30

def formants(x, sr, fs=((700, 110), (1220, 120), (2600, 160), (3300, 200))):
    y = x
    for f, bw in fs:
        r = np.exp(-np.pi * bw / sr); th = 2 * np.pi * f / sr
        y = signal.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    return y

BURST = {"p": (300, 1500), "t": (3500, 6000), "k": (1500, 2600)}

def ddk(sr=44100, syllables="pa"*10, rate=6.0, dip_db=60, snr_db=40, decay=0.0,
        rate_jitter=0.05, f0=120, lead=1.0, seed=0):
    rng = np.random.default_rng(seed)
    cons = [c for c in syllables if c in "ptk"]
    n_syl = len(cons)
    period = 1 / rate
    total = lead + n_syl * period + 1.0
    x = np.zeros(int(total * sr)); truth = []
    src = formants(glottal(len(x), sr, f0, rng=rng), sr)
    src /= np.max(np.abs(src))
    env = np.full(len(x), 10 ** (-dip_db / 20))  # closure floor (slurring = shallow dip)
    env[:int(lead * sr)] = 0; env[int((lead + n_syl * period) * sr):] = 0
    t = lead
    for i, c in enumerate(cons):
        p = period * (1 + rate_jitter * rng.standard_normal())
        clos = 0.35 * p; vowel_on = t + clos; vowel_off = t + p * 0.95
        a, b = int(vowel_on * sr), int(vowel_off * sr)
        amp = 1 - decay * i / n_syl
        ramp = np.minimum(1, np.minimum(np.arange(b - a) / (0.015 * sr), (b - a - np.arange(b - a)) / (0.02 * sr)))
        env[a:b] = np.maximum(env[a:b], amp * ramp)
        floor_lvl = 10 ** (-dip_db / 20) * amp  # closure floor scales with the syllable
        env[int(t * sr):a] = np.minimum(env[int(t * sr):a], floor_lvl) if i else env[int(t * sr):a]
        # release burst
        lo, hi = BURST[c]; bn = int(0.012 * sr)
        bb, ba = signal.butter(2, [lo / (sr / 2), min(hi, sr / 2 * 0.95) / (sr / 2)], btype="band")
        burst = signal.lfilter(bb, ba, rng.standard_normal(bn)) * 0.6 * amp
        x[a - bn:a] += burst
        truth.append({"cons": c, "vowel_on": vowel_on, "vowel_off": vowel_off})
        t += p
    x += src * env
    noise = rng.standard_normal(len(x)); sig_rms = np.sqrt(np.mean((src * env)[env > 0.5] ** 2))
    x += noise * sig_rms / 10 ** (snr_db / 20)
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32), truth

def vowel(sr=44100, trials=((1.0, 5.0),), total=None, f0=130, jitter=0.01, shimmer=0.05,
          snr_db=35, breaks=(), onset_ramp=0.3, wobble_start=None, seed=1):
    rng = np.random.default_rng(seed)
    total = total or trials[-1][1] + 1.0
    n = int(total * sr)
    src = formants(glottal(n, sr, f0, jitter, shimmer, rng), sr); src /= np.max(np.abs(src))
    env = np.zeros(n)
    for a, b in trials:
        i, j = int(a * sr), int(b * sr); k = np.arange(j - i) / sr
        env[i:j] = np.minimum(1, np.minimum(k / onset_ramp, (b - a - k) / 0.2))
    for a, b in breaks: env[int(a * sr):int(b * sr)] = 0
    x = src * env
    if wobble_start is not None:  # unstable voice (tremor) after wobble_start
        t = np.arange(n) / sr; m = (t > wobble_start); x[m] *= 1 + 0.6 * np.sin(2 * np.pi * 7 * t[m])
    x += rng.standard_normal(n) * 10 ** (-snr_db / 20) * 0.3
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32)
