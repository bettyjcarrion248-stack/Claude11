#!/usr/bin/env python3
"""
像素旅途 · PIXEL VOYAGE —— 一段 24 秒、画面和音乐全部由代码生成的像素风短片。

原理：
  1. 在 320x180 的小画布上用 NumPy 逐帧"画"出每个像素：
     有序抖动(Bayer)的渐变天空、视差滚动的山脉、程序化生成的松树和草地、
     手写字符画的精灵角色、昼夜循环的调色板、抖动光晕……
  2. 每个元素都只是时间 t 的函数，所以每一帧可以独立、并行地计算。
  3. 用最近邻插值放大 6 倍到 1920x1080，保留像素颗粒感。
  4. chiptune.py 用方波/三角波/噪声合成 8-bit 配乐，并按同一时间轴对齐音效。
  5. 把原始 RGB 帧通过管道喂给 FFmpeg，和音频一起编码成 MP4。

用法：
  python3 pixel_voyage.py                      # 渲染完整视频 -> out/pixel_voyage.mp4
  python3 pixel_voyage.py --stills 1.5 8 19    # 只导出这几个时刻的静帧（调试用）
"""
import argparse
import math
import os
import subprocess
import sys
import time
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import chiptune

# ================================================================== 基本参数
W, H = 320, 180          # 真正"作画"的分辨率
SCALE = 6                # 放大 6 倍 -> 1920x1080
FPS = 30
DURATION = 24.0
NFRAMES = int(DURATION * FPS)
GROUND = 152             # 地面所在的行
CHAR_X = 108             # 主角在屏幕上的位置

# ================================================================== 时间轴
T_SCENE0, T_SCENE1 = 3.0, 21.5     # 主场景：清晨 -> 深夜
TITLE_OUT = (2.95, 3.55)           # 标题 -> 主场景（抖动溶解）
END_IN = (21.0, 21.6)              # 主场景 -> 片尾
FADE_OUT = (23.3, 24.0)
SPEED = 36.0                       # 镜头速度：像素/秒
DECEL_T, DECEL_LEN = 17.5, 2.0     # 走到篝火旁慢慢停下

TITLE_CN = "像素旅途"
TITLE_EN = "PIXEL VOYAGE"
SUBTITLE = "每一帧画面与音乐，都由代码生成"
CN_DROP_T = [0.30 + i * 0.18 for i in range(len(TITLE_CN))]
EN_TYPE_T0, EN_TYPE_DT = 1.20, 0.05
SUB_T = 1.85
START_T = 2.30
PRESS_T = 2.80
SHOOT_T = 19.9


def cam_x(t):
    tt = max(0.0, t - T_SCENE0)
    d0 = DECEL_T - T_SCENE0
    if tt <= d0:
        return SPEED * tt
    u = min(tt - d0, DECEL_LEN)
    return SPEED * d0 + SPEED * (u - u * u / (2 * DECEL_LEN))


CAM_END = cam_x(1e9)
FIRE_X = int(CAM_END) + CHAR_X + 36      # 篝火的世界坐标：正好停在主角身前
TENT_X = FIRE_X + 22
HOUSES = [150, 330, 600]
LAMPS = [90, 270, 450, 588]


# ================================================================== 工具函数
def smooth(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def bayer(n):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


B4 = np.tile(bayer(4), (H // 4 + 1, W // 4 + 1))[:H, :W].astype(np.float32)
B8 = np.tile(bayer(8), (H // 8 + 1, W // 8 + 1))[:H, :W].astype(np.float32)
YS = np.arange(H)[:, None]


def hsh(n):
    """确定性伪随机：同一个输入永远得到同一个 [0,1) 值，保证每帧可独立重算。"""
    n = np.asarray(n, dtype=np.float64)
    v = np.sin(n * 12.9898 + 78.233) * 43758.5453
    return v - np.floor(v)


def hx(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def pal(*hs):
    return np.stack([hx(h) for h in hs])


def rect(img, x, y, w, h, c):
    x, y = int(x), int(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + int(w)), min(H, y + int(h))
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = c


def px(img, x, y, c):
    x, y = int(x), int(y)
    if 0 <= x < W and 0 <= y < H:
        img[y, x] = c


def blit(img, m, x, y, color, alpha=1.0):
    """把布尔蒙版 m 以颜色 color（单色或逐行渐变）贴到 (x, y)。alpha 用抖动实现。"""
    x, y = int(round(x)), int(round(y))
    h, w = m.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
    if alpha < 1.0:
        sub = sub & (B4[y0:y1, x0:x1] < alpha)
    reg = img[y0:y1, x0:x1]
    color = np.asarray(color, np.float32)
    if color.ndim == 2:
        col = np.broadcast_to(color[y0 - y:y1 - y, None, :], reg.shape)
        reg[sub] = col[sub]
    else:
        reg[sub] = color


def dilate(m):
    p = np.pad(m, 1)
    out = p.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= np.roll(np.roll(p, dy, 0), dx, 1)
    return out


def disc_mask(cx, cy, r):
    x0, y0 = int(math.floor(cx - r)), int(math.floor(cy - r))
    ys, xs = np.mgrid[y0:int(math.ceil(cy + r)) + 1, x0:int(math.ceil(cx + r)) + 1]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r, x0, y0


def glow(img, cx, cy, r, color, strength=1.0):
    """像素风光晕：3 级量化 + Bayer 抖动，而不是平滑渐变。"""
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(H, int(cy + r) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / r
    lvl = np.clip(1 - d, 0, 1) ** 1.4 * strength * 3
    a = np.clip(np.floor(lvl + B4[y0:y1, x0:x1]) / 3 * 0.42, 0, 0.6)[..., None]
    reg = img[y0:y1, x0:x1]
    reg += (np.asarray(color, np.float32) - reg) * a


def wipe(a, b, p):
    """对角线抖动转场：p 从 0 到 1，画面从 a 过渡到 b。"""
    xs = np.arange(W)[None, :]
    thr = B8 * 0.45 + (xs + YS) / (W + H) * 0.55
    return np.where((thr < p * 1.0001)[..., None], b, a)


def ease_out_bounce(u):
    n1, d1 = 7.5625, 2.75
    if u < 1 / d1:
        return n1 * u * u
    if u < 2 / d1:
        u -= 1.5 / d1
        return n1 * u * u + 0.75
    if u < 2.5 / d1:
        u -= 2.25 / d1
        return n1 * u * u + 0.9375
    u -= 2.625 / d1
    return n1 * u * u + 0.984375


# ================================================================== 文字
FONT5 = {
    'A': [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    'B': ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
    'C': [" ####", "#    ", "#    ", "#    ", "#    ", "#    ", " ####"],
    'D': ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    'E': ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
    'F': ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    'G': [" ####", "#    ", "#    ", "#  ##", "#   #", "#   #", " ### "],
    'H': ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    'I': ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"],
    'J': ["  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "],
    'K': ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
    'L': ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    'M': ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"],
    'N': ["#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"],
    'O': [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    'P': ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
    'Q': [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"],
    'R': ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
    'S': [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
    'T': ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    'U': ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    'V': ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
    'W': ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
    'X': ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
    'Y': ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
    'Z': ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
    '0': [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    '1': ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    '2': [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
    '3': ["#### ", "    #", "    #", " ### ", "    #", "    #", "#### "],
    '4': ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    '5': ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    '6': [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "],
    '7': ["#####", "    #", "   # ", "  #  ", "  #  ", "  #  ", "  #  "],
    '8': [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    '9': [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
    ' ': ["     "] * 7,
    '.': ["     "] * 6 + ["  #  "],
    '!': ["  #  "] * 5 + ["     ", "  #  "],
    '-': ["     "] * 3 + [" ### "] + ["     "] * 3,
    '+': ["     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "],
    ':': ["     ", "  #  ", "     ", "     ", "     ", "  #  ", "     "],
}


@lru_cache(maxsize=None)
def text5(s, scale=1):
    m = np.zeros((7, len(s) * 6 - 1), bool)
    for i, ch in enumerate(s):
        for r, row in enumerate(FONT5[ch]):
            for c, v in enumerate(row):
                if v == '#':
                    m[r, i * 6 + c] = True
    return np.repeat(np.repeat(m, scale, 0), scale, 1)


UNIFONT = "/usr/share/fonts/opentype/unifont/unifont.otf"


@lru_cache(maxsize=None)
def cjk(s, scale=1):
    """Unifont 是 16x16 的点阵字体，关掉抗锯齿后每个像素都是硬边。"""
    font = ImageFont.truetype(UNIFONT, 16)
    w = int(font.getlength(s)) + 2
    im = Image.new('1', (w, 16), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = '1'
    d.text((0, 14), s, font=font, fill=1, anchor='ls')
    m = np.array(im, dtype=bool)
    m = m[:, :np.where(m.any(0))[0][-1] + 1]
    return np.repeat(np.repeat(m, scale, 0), scale, 1)


def grad(h, top, bot, steps=4):
    f = np.floor(np.linspace(0, 1, h) * (steps - 0.01)) / (steps - 1)
    return hx(top) * (1 - f[:, None]) + hx(bot) * f[:, None]


def styled(img, m, x, y, fill, outline=None, shadow=None, alpha=1.0):
    if outline is not None or shadow is not None:
        om = dilate(m)
        if shadow is not None:
            blit(img, om, x - 1, y + 1, hx(shadow), alpha)
        if outline is not None:
            blit(img, om, x - 1, y - 1, hx(outline), alpha)
    blit(img, m, x, y, fill if isinstance(fill, np.ndarray) else hx(fill), alpha)


# ================================================================== 调色板（昼夜循环）
THEMES = {
    'dawn': dict(
        sky=pal('#2b3770', '#6a5a9e', '#e88a8a', '#ffc98b'), far2=pal('#9a86b8'),
        far=pal('#6a5f9a', '#4f4a84', '#ead8f0'), hill=pal('#4a8060', '#386650'),
        tree=pal('#2f6b4f', '#224d3e', '#4a3030'), grass=pal('#7ab85a', '#4a8848'),
        dirt=pal('#6b4a3a', '#553a2f', '#87654f'), cloud=pal('#ffd6cc', '#d6a0b8'),
        sun=pal('#ffd890'), tint=pal('#f0dcdc')),
    'day': dict(
        sky=pal('#3a74d6', '#5aa0ee', '#8fd0ff', '#c8f0ff'), far2=pal('#a6c8ea'),
        far=pal('#7a98c8', '#5f7cb0', '#ffffff'), hill=pal('#56ad5c', '#3d8a50'),
        tree=pal('#2f8a4a', '#1f6a3c', '#5a3a28'), grass=pal('#8ad45e', '#56a64a'),
        dirt=pal('#8a5a3c', '#6e4630', '#a77b58'), cloud=pal('#ffffff', '#c8dcf0'),
        sun=pal('#fff4c4'), tint=pal('#ffffff')),
    'sunset': dict(
        sky=pal('#3d2f78', '#8a3f8a', '#f06a5a', '#ffb35a'), far2=pal('#b06a8a'),
        far=pal('#7e4a7e', '#5a3466', '#f7c0a8'), hill=pal('#5e6a48', '#454f38'),
        tree=pal('#3a5a3a', '#2a432c', '#4a2a22'), grass=pal('#a0a44a', '#6c7a3c'),
        dirt=pal('#7a4a34', '#5e382a', '#9a6a4a'), cloud=pal('#ffb08a', '#d0607a'),
        sun=pal('#ff8a5a'), tint=pal('#ffd0b8')),
    'dusk': dict(
        sky=pal('#151a45', '#3a2a6a', '#7a3a7a', '#c0607a'), far2=pal('#4e3c6e'),
        far=pal('#3c3264', '#2c2550', '#9a90c0'), hill=pal('#2f3f4a', '#24323c'),
        tree=pal('#22343a', '#18272e', '#2a1e22'), grass=pal('#466050', '#2f483c'),
        dirt=pal('#3e2c30', '#2f2226', '#54403e'), cloud=pal('#6a4a7a', '#4a3560'),
        sun=pal('#ff6a5a'), tint=pal('#9a90b8')),
    'night': dict(
        sky=pal('#070a1e', '#0e1434', '#1a2350', '#2a3570'), far2=pal('#1e2850'),
        far=pal('#1c2448', '#141a38', '#6a78a8'), hill=pal('#142a30', '#0f2026'),
        tree=pal('#0f2228', '#0a181d', '#1a1418'), grass=pal('#24443e', '#172e2e'),
        dirt=pal('#1e1a24', '#17141c', '#2c2632'), cloud=pal('#2a3260', '#1c2448'),
        sun=pal('#ff6a5a'), tint=pal('#5a6690')),
}
KEYS = [(0.0, 'dawn'), (0.15, 'day'), (0.45, 'day'), (0.6, 'sunset'),
        (0.7, 'dusk'), (0.82, 'night'), (1.0, 'night')]


def theme_at(phi):
    for (p0, a), (p1, b) in zip(KEYS, KEYS[1:]):
        if phi <= p1:
            f = smooth(p0, p1, phi)
            A, B = THEMES[a], THEMES[b]
            return {k: A[k] * (1 - f) + B[k] * f for k in A}
    return THEMES['night']


# ================================================================== 场景元素
SKY_BANDS = 9


def draw_sky(img, stops):
    """分成 9 条色带，色带交界处用 Bayer 有序抖动 —— 经典像素天空。"""
    n = SKY_BANDS
    pos = np.linspace(0, 1, n) * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i)[:, None]
    bands = stops[i] * (1 - f) + stops[i + 1] * f
    p = np.clip(YS / (H * 0.78), 0, 1) * (n - 1)
    k = np.floor(p)
    f2 = np.clip((p - k - 0.5) * 2.5 + 0.5, 0, 1)
    idx = np.clip(k + (f2 > B4), 0, n - 1).astype(int)
    img[:] = bands[idx]


_rs = np.random.default_rng(3)
STARS = [(int(_rs.uniform(0, W)), int(_rs.uniform(0, 120)), _rs.uniform(0, 6.28),
          _rs.uniform(1.5, 4.0), _rs.uniform(0, 1)) for _ in range(150)]
STAR_C = hx('#fffbe6')


def draw_stars(img, t, vis):
    if vis <= 0.01:
        return
    for x, y, ph, sp, th in STARS:
        if th > vis:
            continue
        tw = (0.55 + 0.45 * math.sin(t * sp + ph)) * min(1.0, (vis - th) * 4)
        img[y, x] += (STAR_C - img[y, x]) * tw
        if th < 0.07 and tw > 0.75:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    img[yy, xx] += (STAR_C - img[yy, xx]) * 0.45


def draw_moon(img, mx, my, strength=1.0):
    glow(img, mx, my, 22, hx('#c8d0ff'), 0.7 * strength)
    x0, y0 = int(mx) - 9, int(my) - 9
    ys, xs = np.mgrid[y0:y0 + 19, x0:x0 + 19]
    m1 = (xs - mx) ** 2 + (ys - my) ** 2 <= 7.5 ** 2
    m2 = (xs - mx - 4) ** 2 + (ys - my + 2.5) ** 2 <= 6.5 ** 2
    blit(img, m1 & ~m2, x0, y0, hx('#f4f0c8'))


def make_cloud(rng, w):
    hgt = w // 3 + 4
    ys, xs = np.mgrid[0:hgt, 0:w]
    m = np.zeros((hgt, w), bool)
    n = int(rng.integers(4, 7))
    for k in range(n):
        cx = w * (0.15 + 0.7 * k / (n - 1)) + rng.uniform(-2, 2)
        r = w * 0.17 * (1 - abs(k / (n - 1) - 0.5)) + rng.uniform(1.5, 4)
        cy = hgt - 3 - r * 0.6
        m |= (xs - cx) ** 2 + ((ys - cy) * 1.2) ** 2 <= r * r
    m &= ys <= hgt - 3
    shade = m & ~np.roll(m, -3, axis=0)
    return m, shade


_rc = np.random.default_rng(5)
CLOUDS = []
for _i in range(6):
    _m, _s = make_cloud(_rc, int(_rc.integers(28, 52)))
    CLOUDS.append((_m, _s, _i * 75 + _rc.uniform(0, 30), int(_rc.uniform(12, 62)),
                   _rc.uniform(0.15, 0.3), _rc.uniform(2, 5)))


def draw_clouds(img, t, cam, P):
    span = W + 140
    for m, shade, bx, y, par, drift in CLOUDS:
        x = (bx - cam * par - t * drift) % span - 70
        blit(img, m, x, y, P['cloud'][0])
        blit(img, shade, x, y, P['cloud'][1])


BIRD = [["k...k", ".k.k.", "..k.."], [".....", "kk.kk", "..k.."], [".....", "..k..", ".k.k."]]
BIRD_M = [np.array([[c == 'k' for c in row] for row in f]) for f in BIRD]


def draw_birds(img, t, tint):
    if not 5.5 < t < 15.5:
        return
    for i in range(4):
        x = W + 10 - (t - 5.5) * 40 + i * 10 + (i % 2) * 4
        y = 46 + i * 4 + math.sin(t * 2 + i) * 2
        fr = [0, 1, 2, 1][int(t * 8 + i) % 4]
        blit(img, BIRD_M[fr], x, y, hx('#2a2a3a') * tint)


def far2_h(wx):
    return 92 + 16 * np.sin(wx * 0.012 + 0.5) + 9 * np.sin(wx * 0.029 + 2.0) + 4 * np.sin(wx * 0.071)


def far_h(wx):
    return (114 - 26 * (0.5 + 0.5 * np.sin(wx * 0.016)) - 16 * (0.5 + 0.5 * np.sin(wx * 0.043 + 1.3))
            - 5 * np.sin(wx * 0.11 + 0.7) - 2 * np.sin(wx * 0.31))


def hill_h(wx):
    return 134 - 7 * np.sin(wx * 0.021) - 5 * np.sin(wx * 0.053 + 2) - 2.5 * np.sin(wx * 0.13 + 1)


def mountain_layer(img, fn, off, light, dark=None, snow=None, snow_y=None):
    wx = np.arange(W) + off
    h = fn(wx)
    m = YS >= np.floor(h)[None, :]
    if dark is None:
        img[m] = light
        return
    # 明暗面：朝左的山坡迎着光；分界线从山峰斜着往下走，而不是竖直切开
    wxe = np.arange(-64, W) + off
    lit_e = fn(wxe + 6) < fn(wxe - 6)
    depth = np.maximum(YS - np.floor(h)[None, :], 0)
    col = np.clip(np.arange(W)[None, :] - (depth * 0.6).astype(int) + 64, 0, W + 63)
    lit = lit_e[col]
    img[m & lit] = light
    img[m & ~lit] = dark
    if snow is not None:
        sl = snow_y + (hsh(np.floor(wx / 3)) * 4).astype(int)
        sm = m & (YS < sl[None, :])
        img[sm & lit] = snow
        img[sm & ~lit] = snow * 0.82


def pine(img, x, base, hgt, light, dark, trunk):
    rect(img, x, base - 2, 1, 3, trunk)
    top = base - hgt
    tier = max(3, hgt // 3)
    for y in range(top, base - 2):
        r = y - top
        w = int(1 + (r % tier) * 0.7 + (r // tier) * 1.0)
        rect(img, x - w, y, w + 1, 1, light)
        rect(img, x + 1, y, w, 1, dark)


def draw_hills(img, cam, P):
    off = int(cam * 0.45)
    wx = np.arange(W) + off
    h = np.floor(hill_h(wx))
    m = YS >= h[None, :]
    img[m] = P['hill'][0]
    depth = (YS - h[None, :] - 5) / 9.0
    img[m & (B4 < depth)] = P['hill'][1]
    for k in range((off - 12) // 11, (off + W + 12) // 11 + 1):
        if hsh(k * 1.7) > 0.6:
            continue
        twx = k * 11 + hsh(k * 3.1) * 6
        pine(img, int(twx - off), int(hill_h(twx)) + 3, int(9 + hsh(k * 5.3) * 10),
             P['tree'][0], P['tree'][1], P['tree'][2])


def draw_land(img, cam, P):
    mountain_layer(img, far2_h, int(cam * 0.08), P['far2'][0])
    mountain_layer(img, far_h, int(cam * 0.18), P['far'][0], P['far'][1], P['far'][2], 80)
    draw_hills(img, cam, P)


FLOWER_C = [hx('#e43b44'), hx('#feae34'), hx('#f4f4f4'), hx('#8f7dff')]


def draw_ground(img, ci, P, tint):
    wx = np.arange(W) + ci
    sub = img[GROUND:]
    yy = np.arange(H - GROUND)[:, None]
    sub[:] = P['dirt'][0]
    bx = np.floor(wx / 5)[None, :]
    stone = hsh(bx * 1.3 + np.floor((yy + (bx % 2) * 2) / 3) * 7.7) > 0.86
    sub[stone] = P['dirt'][2]
    sub[hsh(wx[None, :] * 3.7 + yy * 11.3) > 0.94] = P['dirt'][1]
    gd = 2 + (hsh(wx * 0.37) * 3).astype(int)
    sub[yy < gd[None, :]] = P['grass'][1]
    sub[0] = P['grass'][0]
    tuft = hsh(wx * 7.13)
    c1 = np.where(tuft > 0.80)[0]
    c2 = np.where(tuft > 0.93)[0]
    img[GROUND - 1, c1] = P['grass'][0]
    img[GROUND - 2, c2] = P['grass'][0]
    for x in np.where(hsh(wx * 3.31 + 1.7) > 0.955)[0]:
        img[GROUND - 1, x] = P['grass'][1]
        img[GROUND - 2, x] = P['grass'][1]
        img[GROUND - 3, x] = FLOWER_C[int(hsh(wx[x] * 9.1) * 4)] * tint


def draw_house(img, x, t, tint, nf, seed, lights):
    b = GROUND
    c = lambda h: hx(h) * tint
    rect(img, x, b - 16, 26, 16, c('#e8d5b0'))
    rect(img, x + 23, b - 16, 3, 16, c('#c4a888'))
    rect(img, x, b - 16, 26, 1, c('#8a6a4a'))
    rect(img, x, b - 2, 26, 2, c('#8a8a9a'))
    for r in range(10):                                    # 屋顶：逐行收窄
        rect(img, x - 3 + r, b - 17 - r, 32 - 2 * r, 1, c('#b5483a') if r % 2 == 0 else c('#9a3a30'))
    rect(img, x - 3, b - 17, 32, 1, c('#7a2a24'))
    rect(img, x + 18, b - 28, 4, 8, c('#8a7a7a'))           # 烟囱
    rect(img, x + 17, b - 29, 6, 1, c('#5a5050'))
    rect(img, x + 4, b - 11, 6, 11, c('#6b4026'))           # 门
    px(img, x + 8, b - 6, c('#e8c070'))
    lit = nf > 0.12 + 0.25 * float(hsh(seed))
    rect(img, x + 13, b - 12, 8, 7, c('#5a3a2a'))           # 窗
    rect(img, x + 14, b - 11, 6, 5, hx('#ffd36a') if lit else c('#8fc8e8'))
    rect(img, x + 16, b - 11, 1, 5, c('#5a3a2a'))
    rect(img, x + 14, b - 9, 6, 1, c('#5a3a2a'))
    if lit:
        lights.append((x + 17, b - 9, 16, hx('#ffc65a'), 0.8))
    for i in range(7):                                      # 炊烟
        age = (t * 0.5 + i / 7 + seed * 0.13) % 1.0
        sx = x + 20 - age * 14 + math.sin(age * 6 + i) * 1.5
        sy = b - 31 - age * 26
        m, mx0, my0 = disc_mask(sx, sy, 1.0 + age * 2.6)
        blit(img, m, mx0, my0, hx('#d8d8e0') * tint, alpha=0.85 * (1 - age))


def draw_lamp(img, x, tint, nf, seed, lights):
    b = GROUND
    c = lambda h: hx(h) * tint
    rect(img, x, b - 22, 2, 22, c('#3a3a4a'))
    rect(img, x - 1, b - 2, 4, 2, c('#2a2a3a'))
    rect(img, x - 2, b - 25, 6, 1, c('#2a2a3a'))
    on = nf > 0.3 + 0.15 * float(hsh(seed))
    rect(img, x - 1, b - 24, 4, 3, hx('#ffe08a') if on else c('#a0a8b8'))
    if on:
        lights.append((x + 1, b - 23, 26, hx('#ffcf6a'), 0.9))


def draw_tent(img, x, tint):
    b = GROUND
    for r in range(15):
        half = int((15 - r) * 0.75)
        rect(img, x + 11 - half, b - 1 - r, half, 1, hx('#d8783a') * tint)
        rect(img, x + 11, b - 1 - r, half + 1, 1, hx('#a8562a') * tint)
    for r in range(8):
        half = int((8 - r) * 0.5)
        rect(img, x + 11 - half, b - 1 - r, 2 * half + 1, 1, hx('#3a2420') * tint)
    rect(img, x + 11, b - 18, 1, 4, hx('#5a3a2a') * tint)


def draw_fire(img, x, t, tint, lights):
    b = GROUND
    st, std = hx('#8a8a9a') * tint, hx('#5a5a6a') * tint
    for dx in (-8, -5, 5, 8):
        rect(img, x + dx - 1, b - 2, 3, 2, st)
        rect(img, x + dx - 1, b - 1, 3, 1, std)
    for i in range(10):                                     # 交叉的木柴
        rect(img, x - 5 + i, b - 1 - i // 3, 1, 2, hx('#6b4026') * tint)
        rect(img, x + 4 - i, b - 1 - i // 3, 1, 2, hx('#7a4a2a') * tint)
    fi = int(t * 12)
    X, Y = np.meshgrid(np.arange(-5, 6), np.arange(-17, -2))
    hgt = 13 * np.clip(1 - (np.abs(X) / 6.0) ** 1.4, 0, 1) * (0.85 + 0.25 * hsh(X * 3.1 + fi * 7.7))
    v = 1 - (-Y - 3) / np.maximum(hgt, 0.1)
    v += (hsh(X * 1.7 + Y * 3.3 + fi * 5.1) - 0.5) * 0.35
    for thr, col in ((0.08, '#e43b44'), (0.3, '#ff8a2a'), (0.55, '#ffd23f'), (0.78, '#fff3b0')):
        blit(img, v > thr, x - 5, b - 17, hx(col))
    for i in range(10):                                     # 火星
        age = (t * 0.9 + i / 10) % 1.0
        if hsh(i * 3.3 + int(t * 0.9 + i / 10)) < 0.35:
            continue
        px(img, x + math.sin(i * 7 + age * 5) * 4, b - 8 - age * 30,
           hx('#ffd23f') if age < 0.5 else hx('#e43b44'))
    lights.append((x, b - 8, 34, hx('#ff9a3c'), 0.85 + 0.12 * math.sin(t * 13) + 0.08 * math.sin(t * 7.3)))


# ------------------------------------------------------------------ 主角精灵（手写字符画）
CHAR_PAL = {k: hx(v) for k, v in {
    'k': '#1a1c2c', 'b': '#f4f4f4', 's': '#c2c3c7', 'e': '#1a1c2c', 'p': '#ff9aa2',
    'r': '#e43b44', 'R': '#a22633', 'l': '#333c57', 'g': '#7a8a9a', 'G': '#ffd36a'}.items()}
BODY = [
    "....kkkkkk....",
    "...kbbbbbbk...",
    "..kbbbbbbbbk..",
    "..kbbbbbbbbk..",
    "..kbbbbebebk..",
    "..kbbbbebebk..",
    "..kbbbpbbbpk..",
    "..kbbbbbbbbk..",
    "..krrrrrrrrk..",
    "..kRRRRRRRRk..",
    "..ksbbbbbbsk..",
    "..ksbbbbbbsk..",
    "...kssssssk...",
]
LEGS = {'spread': ["...ll....ll...", "..ll......ll.."],
        'pass': ["....ll..ll....", "....ll..ll...."]}
TAILS = [{(8, 0): 'r', (8, 1): 'r', (9, 0): 'R', (9, 1): 'R'},
         {(8, 1): 'r', (9, 0): 'r', (9, 1): 'R', (10, 0): 'R'}]
LANTERN = [".k..", "kkkk", "kggk", "kggk", "kkkk"]


def draw_rows(img, rows, x, y, cmap):
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch != '.':
                px(img, x + c, y + r, cmap[ch])


def draw_char(img, x, gy, t, step, walking, tint, lantern_on, lights):
    ctint = 1 - (1 - tint) * 0.8
    cmap = {k: v * ctint for k, v in CHAR_PAL.items()}
    cmap['G'] = CHAR_PAL['G']
    frame = int(step) % 4 if walking else 0
    bob = (1 if frame in (1, 3) else 0) if walking else (1 if (t % 1.2) < 0.6 else 0)
    legs = LEGS['spread'] if walking and frame in (0, 2) else LEGS['pass']
    if bob:
        legs = [legs[0]] + legs
    draw_rows(img, legs, x, gy - len(legs), cmap)
    top = gy - 15 - bob
    body = list(BODY)
    if (t % 3.1) < 0.12:                    # 眨眼
        body[4] = body[4].replace('e', 'b')
    draw_rows(img, body, x, top, cmap)
    for (r, c), ch in TAILS[(frame // 2) if walking else int(t * 5) % 2].items():
        px(img, x + c, top + r, cmap[ch])
    lan = [row.replace('g', 'G') for row in LANTERN] if lantern_on else LANTERN
    draw_rows(img, lan, x + 12, top + 8, cmap)
    if lantern_on:
        lights.append((x + 14, top + 11, 22, hx('#ffcf6a'), 0.8 + 0.06 * math.sin(t * 9)))


def draw_fireflies(img, t, ci, nf, lights):
    if nf < 0.4:
        return
    for i in range(18):
        bx = 420 + i * 19 + hsh(i * 2.7) * 12
        by = 118 + hsh(i * 4.1) * 28
        x = bx - ci + 8 * math.sin(t * 0.6 + i * 1.3)
        y = by + 5 * math.sin(t * 1.1 + i * 2.1)
        if math.sin(t * 2.3 + i * 1.7) > -0.3 and 0 <= x < W:
            px(img, x, y, hx('#e8ff8a'))
            lights.append((x, y, 6, hx('#c8ff6a'), 0.7 * (nf - 0.4) / 0.6))


def draw_shooting_star(img, t):
    u = (t - SHOOT_T) / 0.7
    if not 0 <= u <= 1:
        return
    hx0, hy0 = 250 - u * 90, 16 + u * 38
    for k in range(16):
        a = (1 - k / 16) * (1 - max(0, u - 0.7) / 0.3)
        x, y = int(hx0 + k * 2.4), int(hy0 - k * 1.0)
        if 0 <= x < W and 0 <= y < H:
            img[y, x] += (STAR_C - img[y, x]) * a


def draw_hud(img, phi, alpha):
    hour = 5.5 + phi * 16.5
    hh, mm = int(hour), int((hour % 1) * 6) * 10
    label = ("清晨" if hour < 8 else "上午" if hour < 11 else "正午" if hour < 13.5
             else "午后" if hour < 17 else "黄昏" if hour < 19 else "夜晚")
    m = cjk(f"{hh:02d}:{mm:02d} {label}")
    w = m.shape[1] + 20
    sel = B4[4:24, 4:4 + w] < alpha
    reg = img[4:24, 4:4 + w]
    reg[sel] *= 0.45
    if hour < 18.5:
        dm, x0, y0 = disc_mask(12, 13.5, 3.2)
        blit(img, dm, x0, y0, hx('#ffd23f'), alpha)
    else:
        x0, y0 = 8, 9
        ys, xs = np.mgrid[y0:y0 + 10, x0:x0 + 10]
        mm_ = ((xs - 12) ** 2 + (ys - 13.5) ** 2 <= 12.5) & ~((xs - 14) ** 2 + (ys - 12) ** 2 <= 9)
        blit(img, mm_, x0, y0, hx('#f4f0c8'), alpha)
    blit(img, m, 20, 6, hx('#f4f4f4'), alpha)


# ================================================================== 三个"镜头"
def scene_frame(t):
    phi = min(max((t - T_SCENE0) / (T_SCENE1 - T_SCENE0), 0.0), 1.0)
    P = theme_at(phi)
    tint = P['tint'][0] / 255.0
    nf = smooth(0.6, 0.8, phi)
    cam = cam_x(t)
    ci = int(cam)
    img = np.empty((H, W, 3), np.float32)
    lights = []

    draw_sky(img, P['sky'])
    draw_stars(img, t, nf)
    draw_shooting_star(img, t)
    s = phi / 0.68                                     # 太阳沿弧线东升西落
    if s <= 1.0:
        sx, sy = 40 + s * 240, 126 - math.sin(s * math.pi) * 96
        sun = P['sun'][0]
        glow(img, sx, sy, 28, sun, 0.9)
        m, x0, y0 = disc_mask(sx, sy, 9)
        blit(img, m, x0, y0, sun)
        m, x0, y0 = disc_mask(sx - 2, sy - 2, 5)
        blit(img, m, x0, y0, sun * 0.4 + 255 * 0.6)
    if phi > 0.6:                                      # 月亮升起
        u = min(1.0, (phi - 0.6) / 0.4)
        draw_moon(img, 262 - u * 50, 120 - math.sin(u * math.pi / 2) * 84, min(1.0, u * 3))
    draw_clouds(img, t, cam, P)
    draw_birds(img, t, tint)
    draw_land(img, cam, P)

    for i, wx in enumerate(LAMPS):
        if -10 < wx - ci < W + 10:
            draw_lamp(img, wx - ci, tint, nf, i + 0.5, lights)
    for i, wx in enumerate(HOUSES):
        if -40 < wx - ci < W + 10:
            draw_house(img, wx - ci, t, tint, nf, i + 1.0, lights)
    if TENT_X - ci < W + 10:
        draw_tent(img, TENT_X - ci, tint)
    if FIRE_X - ci < W + 40:
        draw_fire(img, FIRE_X - ci, t, tint, lights)
    draw_ground(img, ci, P, tint)

    walking = t < DECEL_T + DECEL_LEN - 0.05
    draw_char(img, CHAR_X, GROUND, t, cam / 5.0, walking, tint, nf > 0.35, lights)
    draw_fireflies(img, t, ci, nf, lights)
    for x, y, r, c, s_ in lights:
        glow(img, x, y, r, c, s_)
    draw_hud(img, phi, smooth(T_SCENE0 + 0.3, T_SCENE0 + 0.8, t) * (1 - smooth(20.4, 20.9, t)))
    return img


def title_frame(t):
    P = THEMES['night']
    img = np.empty((H, W, 3), np.float32)
    draw_sky(img, P['sky'])
    draw_stars(img, t, 1.0)
    draw_moon(img, 268, 30)
    draw_land(img, -400 + t * 6, P)
    draw_ground(img, -400 + int(t * 6), P, P['tint'][0] / 255)

    for i, ch in enumerate(TITLE_CN):                  # 大标题逐字弹跳落下
        u = (t - CN_DROP_T[i]) / 0.45
        if u <= 0:
            continue
        dy = -(1 - ease_out_bounce(min(u, 1.0))) * 44
        m = cjk(ch, 2)
        styled(img, m, 96 + i * 32 + (32 - m.shape[1]) // 2, 22 + dy,
               grad(m.shape[0], '#fff3b0', '#ff8a2a'), outline='#1a1c2c', shadow='#0b0c1e')
    if t >= EN_TYPE_T0:                                # 英文副标题打字机
        n = min(len(TITLE_EN), int((t - EN_TYPE_T0) / EN_TYPE_DT) + 1)
        m = text5(TITLE_EN[:n], 2)
        x0 = (W - text5(TITLE_EN, 2).shape[1]) // 2
        styled(img, m, x0, 64, hx('#9fd4ff'), outline='#1a1c2c')
        if t < 2.3 and (t * 4) % 1 < 0.6:
            rect(img, x0 + m.shape[1] + 3, 64, 6, 14, hx('#9fd4ff'))
    if t >= SUB_T:
        m = cjk(SUBTITLE)
        styled(img, m, (W - m.shape[1]) // 2, 88, hx('#e8e8f8'), outline='#1a1c2c',
               alpha=smooth(SUB_T, SUB_T + 0.35, t))
    if t >= START_T:
        on = (t * 12) % 1 < 0.5 if t >= PRESS_T else (t * 3) % 1 < 0.65
        if on:
            col = '#ffd23f' if t >= PRESS_T else '#f4f4f4'
            m = text5("PRESS START")
            x0 = (W - m.shape[1]) // 2 + 4
            styled(img, m, x0, 116, hx(col), outline='#1a1c2c')
            for r in range(5):                         # ▶ 小箭头
                rect(img, x0 - 9, 117 + r, 3 - abs(r - 2) + 1, 1, hx(col))
    return img


def end_frame(t):
    img = np.empty((H, W, 3), np.float32)
    draw_sky(img, pal('#05060f', '#090c22', '#121838', '#1e2550'))
    draw_stars(img, t, 1.0)
    m = text5("THE END", 3)
    styled(img, m, (W - m.shape[1]) // 2, 30, grad(m.shape[0], '#fff3b0', '#ff8a2a'),
           outline='#1a1c2c', shadow='#000000')
    m = cjk("旅途未完，下次见")
    styled(img, m, (W - m.shape[1]) // 2, 66, hx('#cfe8ff'), outline='#1a1c2c')
    for i, (s, c) in enumerate((("EVERY PIXEL AND NOTE IS MADE OF CODE", '#8a9ac8'),
                                ("PYTHON + NUMPY + FFMPEG", '#6a7ab0'))):
        m = text5(s)
        blit(img, m, (W - m.shape[1]) // 2, 94 + i * 12, hx(c))
    rect(img, 120, 160, 80, 1, hx('#3a5a4a'))          # 小小的草地平台
    rect(img, 122, 161, 76, 2, hx('#24342e'))
    lights = []
    draw_char(img, 153, 160, t, 0, False, np.ones(3, np.float32) * 0.85, True, lights)
    for i in range(7):
        x = 160 + 60 * math.sin(t * 0.5 + i * 0.9)
        y = 140 + 12 * math.sin(t * 1.3 + i * 2.0)
        if math.sin(t * 2.5 + i) > -0.2:
            px(img, x, y, hx('#e8ff8a'))
            lights.append((x, y, 6, hx('#c8ff6a'), 0.5))
    for x, y, r, c, s_ in lights:
        glow(img, x, y, r, c, s_)
    return img


def render(t):
    if t < TITLE_OUT[0]:
        img = title_frame(t)
    elif t < TITLE_OUT[1]:
        img = wipe(title_frame(t), scene_frame(t), smooth(*TITLE_OUT, t))
    elif t < END_IN[0]:
        img = scene_frame(t)
    elif t < END_IN[1]:
        img = wipe(scene_frame(t), end_frame(t), smooth(*END_IN, t))
    else:
        img = end_frame(t)
    if t > FADE_OUT[0]:
        img = wipe(img, np.zeros_like(img), smooth(*FADE_OUT, t))
    return np.clip(img, 0, 255).astype(np.uint8)


def frame_bytes(i):
    return render(i / FPS).tobytes()


# ================================================================== 主程序
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--out', default=os.path.join(here, 'out', 'pixel_voyage.mp4'))
    ap.add_argument('--stills', type=float, nargs='*', help='只导出这些秒数的静帧 PNG')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.stills:
        for s in args.stills:
            p = os.path.join(os.path.dirname(args.out), f'still_{s:05.2f}.png')
            Image.fromarray(render(s)).resize((W * 3, H * 3), Image.NEAREST).save(p)
            print(p)
        return

    t0 = time.time()
    wav = os.path.splitext(args.out)[0] + '.wav'
    chiptune.render(wav, dict(
        scene_t0=T_SCENE0, cn_drops=CN_DROP_T,
        en_types=[EN_TYPE_T0 + i * EN_TYPE_DT for i, c in enumerate(TITLE_EN) if c != ' '],
        sub_t=SUB_T, press_t=PRESS_T, shoot_t=SHOOT_T, end_t=END_IN[0]), DURATION)
    print(f'♪ 配乐合成完成 {time.time() - t0:.1f}s')

    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-i', wav,
           '-vf', f'scale={W * SCALE}:{H * SCALE}:flags=neighbor',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-tune', 'animation', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(args.workers) as pool:
        for k, buf in enumerate(pool.imap(frame_bytes, range(NFRAMES), chunksize=4)):
            proc.stdin.write(buf)
            if k % 30 == 0:
                print(f'\r▶ 渲染 {k}/{NFRAMES} 帧', end='', flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit('ffmpeg 编码失败')
    print(f'\r✔ 完成：{args.out}  （共 {NFRAMES} 帧，用时 {time.time() - t0:.1f}s）')


if __name__ == '__main__':
    main()
