"""
8-bit 芯片音乐合成器 —— 方波 / 三角波 / 噪声，全部用 NumPy 现场计算，不依赖任何音色库。

和画面共用同一条时间轴（cues），所以打字音效、开始音效、流星音效都能和画面对上。
"""
import wave

import numpy as np

SR = 44100
RNG = np.random.default_rng(11)


def mtof(m):
    """MIDI 音高 -> 频率"""
    return 440.0 * 2 ** ((m - 69) / 12)


def smooth(a, b, x):
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------ 振荡器
def tone(m, dur, kind="square", duty=0.5, a=0.004, d=0.09, s=0.55, r=0.03,
         vib=0.0, slide_to=None):
    n = max(1, int(dur * SR))
    t = np.arange(n) / SR
    f0 = mtof(m)
    if slide_to is not None:
        f = f0 * (mtof(slide_to) / f0) ** (t / dur)
    else:
        f = np.full(n, f0)
    if vib:  # 延迟颤音
        f = f * (1 + vib * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.12) / 0.15, 0, 1))
    ph = np.cumsum(f) / SR % 1.0
    if kind == "square":
        w = np.where(ph < duty, 1.0, -1.0)
    elif kind == "tri":
        w = 1 - 4 * np.abs(ph - 0.5)
        w = np.round(w * 7.5) / 7.5          # 4-bit 量化：FC 三角波的"台阶"味
    else:
        raise ValueError(kind)
    env = np.minimum(1, t / a) * (s + (1 - s) * np.exp(-t / d)) * np.clip((dur - t) / r, 0, 1)
    return w * env


def noise(dur, decay, hp=True):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = RNG.uniform(-1, 1, n)
    if hp:
        x = np.diff(x, prepend=0.0) * 0.5
    return x * np.exp(-t * decay)


def kick():
    t = np.arange(int(0.28 * SR)) / SR
    f = 45 + 120 * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12)


def snare():
    t = np.arange(int(0.2 * SR)) / SR
    return noise(0.2, 20, hp=False) * 0.7 + np.sin(2 * np.pi * 185 * t) * np.exp(-t * 28) * 0.5


# ------------------------------------------------------------------ 混音台
class Mixer:
    def __init__(self, dur):
        self.n = int(dur * SR)
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)

    def add(self, sig, t0, gain=1.0, pan=0.0):
        i0 = int(round(t0 * SR))
        if i0 >= self.n:
            return
        if i0 < 0:
            sig, i0 = sig[-i0:], 0
        i1 = min(self.n, i0 + len(sig))
        s = sig[: i1 - i0] * gain
        self.L[i0:i1] += s * min(1.0, 1.0 - pan)
        self.R[i0:i1] += s * min(1.0, 1.0 + pan)


# ------------------------------------------------------------------ 乐谱
ROOT = {"C": 36, "G": 43, "Am": 45, "F": 41}
TONES = {"C": [60, 64, 67, 72], "G": [59, 62, 67, 71],
         "Am": [57, 60, 64, 69], "F": [57, 60, 65, 69]}
CHORDS = ["C", "G", "Am", "F", "C", "G", "Am", "FG", "C"]

# (MIDI 音高, 八分音符个数)，每小节 8 个八分音符
MELODY = [
    [(76, 2), (79, 2), (84, 2), (83, 1), (81, 1)],
    [(79, 3), (74, 1), (79, 2), (81, 1), (83, 1)],
    [(84, 2), (83, 1), (81, 1), (76, 3), (72, 1)],
    [(77, 2), (76, 1), (74, 1), (72, 2), (74, 2)],
    [(76, 2), (79, 2), (84, 2), (86, 1), (88, 1)],
    [(86, 3), (83, 1), (79, 2), (81, 1), (83, 1)],
    [(84, 2), (81, 2), (76, 2), (79, 1), (81, 1)],
    [(77, 2), (81, 2), (79, 3), (None, 1)],
    [(79, 2), (76, 2), (72, 4)],          # 篝火旁：轻声收尾
]

BAR, E8 = 2.0, 0.25   # 120 BPM


def render(path, cues, dur):
    mx = Mixer(dur)
    T0 = cues["scene_t0"]

    # ---------------- 片头：夜色里的琶音 + 打字音效
    mx.add(tone(48, T0, "tri", a=1.2, d=1, s=1, r=0.4), 0.0, 0.10)
    arp = [60, 64, 67, 71]
    k = 0
    while k * E8 < T0 - 0.2:
        g = 0.05 + 0.05 * (k * E8 / T0)
        mx.add(tone(arp[k % 4] + 12, 0.24, "tri", d=0.12, s=0.2), k * E8, g, pan=-0.3)
        k += 1
    for i, tt in enumerate(cues["cn_drops"]):          # 大标题落下："咚"
        mx.add(tone(60, 0.09, "tri", s=0.0, d=0.05, slide_to=40), tt + 0.12, 0.28)
        mx.add(tone([72, 76, 79, 84][i % 4], 0.10, "square", duty=0.5, s=0.2), tt + 0.12, 0.07)
    for i, tt in enumerate(cues["en_types"]):          # 英文打字："嘀"
        mx.add(tone(88 if i % 2 else 91, 0.03, "square", duty=0.25, s=0.3), tt, 0.045)
    for m_, dt in ((84, 0.0), (91, 0.08)):             # 副标题浮现
        mx.add(tone(m_, 0.9, "tri", d=0.3, s=0.0, r=0.2), cues["sub_t"] + dt, 0.10)
    pt = cues["press_t"]                               # 按下 START：投币音
    mx.add(tone(83, 0.07, "square", duty=0.5, s=0.8), pt, 0.09)
    mx.add(tone(88, 0.40, "square", duty=0.5, d=0.12, s=0.0, r=0.1), pt + 0.07, 0.09)
    rise = tone(48, T0 - 2.3, "square", duty=0.5, s=1.0, slide_to=72)
    mx.add(rise * np.linspace(0, 1, len(rise)) ** 2, 2.3, 0.035)
    nr = noise(0.7, 0.0)
    mx.add(nr * np.linspace(0, 1, len(nr)) ** 3, T0 - 0.7, 0.08)
    mx.add(noise(1.6, 3.0), T0, 0.12, pan=0.2)       # 镲片

    # ---------------- 主段：C - G - Am - F
    for b, chord in enumerate(CHORDS):
        tb = T0 + b * BAR
        for e in range(8):
            ch = ("F" if e < 4 else "G") if chord == "FG" else chord
            te = tb + e * E8
            if b < 8:
                mx.add(tone(ROOT[ch] + (12 if e % 2 else 0), 0.22, "tri", s=0.8, d=0.2), te, 0.30)
                for s16 in (0, 1):
                    m_ = TONES[ch][(e * 2 + s16) % 4] + 12
                    mx.add(tone(m_, 0.11, "square", duty=0.125, s=0.4, d=0.05),
                           te + s16 * E8 / 2, 0.042, pan=-0.35)
            else:
                if e in (0, 4):
                    mx.add(tone(ROOT[ch], 0.95, "tri", d=0.6, s=0.3, r=0.2), te, 0.26)
                mx.add(tone(TONES[ch][e % 4] + 12, 0.4, "tri", d=0.18, s=0.0, r=0.1), te, 0.06, pan=-0.3)
            # 鼓：前 6 小节完整，入夜后变稀疏，篝火旁停下
            if b <= 5:
                if e in (0, 3, 4):
                    mx.add(kick(), te, 0.55)
                if e in (2, 6):
                    mx.add(snare(), te, 0.22)
                mx.add(noise(0.05, 90), te, 0.07 if e % 2 else 0.045, pan=0.3)
            elif b <= 7:
                if e == 0:
                    mx.add(kick(), te, 0.4)
                if e % 2 == 0:
                    mx.add(noise(0.05, 90), te, 0.035, pan=0.3)
        # 主旋律 + 回声
        tcur = tb
        for m_, n8 in MELODY[b]:
            ln = n8 * E8
            if m_ is not None:
                if b < 8:
                    sig = tone(m_, ln * 0.92, "square", duty=0.25, d=0.12, s=0.6, vib=0.006)
                    mx.add(sig, tcur, 0.11)
                    mx.add(sig, tcur + 0.375, 0.035, pan=0.5)
                else:
                    sig = tone(m_, ln * 0.95, "tri", d=0.4, s=0.5, vib=0.004, r=0.15)
                    mx.add(sig, tcur, 0.17)
                    mx.add(sig, tcur + 0.375, 0.05, pan=0.5)
            tcur += ln

    # ---------------- 流星："叮铃"
    for i, m_ in enumerate([96, 93, 91, 88, 84, 79]):
        mx.add(tone(m_, 0.25, "tri", d=0.08, s=0.0), cues["shoot_t"] + i * 0.045, 0.06, pan=0.4 - i * 0.16)

    # ---------------- 篝火噼啪声
    t = 13.5
    while t < cues["end_t"] + 0.6:
        vis = smooth(14.0, 18.0, t) * (1 - smooth(cues["end_t"], cues["end_t"] + 0.6, t))
        mx.add(noise(0.006, 700), t, 0.25 * vis * RNG.uniform(0.3, 1.0), pan=RNG.uniform(-0.2, 0.5))
        t += RNG.exponential(1 / 9)

    # ---------------- 片尾：钟声琶音
    te = cues["end_t"]
    for i, m_ in enumerate([72, 76, 79, 84, 88]):
        mx.add(tone(m_, 1.8, "tri", a=0.002, d=0.55, s=0.0, r=0.3), te + i * 0.12, 0.15, pan=-0.3 + i * 0.15)
    mx.add(tone(36, 2.6, "tri", d=1.2, s=0.0, r=0.4), te, 0.26)
    mx.add(tone(84, 1.4, "tri", a=0.002, d=0.5, s=0.0, r=0.3), te + 1.6, 0.10)
    mx.add(tone(91, 1.2, "tri", a=0.002, d=0.5, s=0.0, r=0.3), te + 1.72, 0.07)

    # ---------------- 母带：软限幅 + 淡入淡出
    tt = np.arange(mx.n) / SR
    fade = np.clip(tt / 0.05, 0, 1) * (1 - smooth(dur - 0.8, dur, tt))
    out = np.stack([mx.L, mx.R], 1) * fade[:, None]
    out = np.tanh(out * 1.2)
    out *= 0.89 / np.max(np.abs(out))
    pcm = (out * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
