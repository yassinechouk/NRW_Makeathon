"""
hud.py — Display. Clean drawing primitives for OpenCV.

OpenCV can only draw right-angled rectangles and aliased text.
Everything here serves to obtain a sharp rendering: rounded corners, translucent panels,
halo around the contour, anti-aliasing everywhere.

The approach: a single piece of information dominates the screen — the TYPE. The rest
(confidence, background state, fps) remains discreet, at the bottom, and does not move.
"""

from __future__ import annotations

import cv2
import numpy as np

# palette, in BGR. Chosen distinct including for red-green color blindness:
# green / amber / blue are separated by luminance as much as by hue.
TYPE_COLORS = [
    (128, 214, 126),   # 1 — green
    (86, 180, 250),    # 2 — amber
    (235, 163, 95),    # 3 — blue
    (200, 130, 235),   # 4 — purple
    (120, 235, 235),   # 5 — yellow
]
GREY = (128, 124, 120)
INK = (250, 248, 245)
DIM = (168, 164, 160)
PANEL = (32, 30, 28)

F = cv2.FONT_HERSHEY_DUPLEX
FS = cv2.FONT_HERSHEY_SIMPLEX


def type_color(index):
    return TYPE_COLORS[index % len(TYPE_COLORS)] if index is not None else GREY


# ------------------------------------------------------------- primitives
def rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    r = int(min(r, w // 2, h // 2))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1, cv2.LINE_AA)
        for cx, cy, a0 in ((x + r, y + r, 180), (x + w - r, y + r, 270),
                           (x + w - r, y + h - r, 0), (x + r, y + h - r, 90)):
            cv2.ellipse(img, (cx, cy), (r, r), a0, 0, 90, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
        for cx, cy, a0 in ((x + r, y + r, 180), (x + w - r, y + r, 270),
                           (x + w - r, y + h - r, 0), (x + r, y + h - r, 90)):
            cv2.ellipse(img, (cx, cy), (r, r), a0, 0, 90, color, thickness, cv2.LINE_AA)


def panel(img, x, y, w, h, r=14, color=PANEL, alpha=0.82):
    """Translucent panel: text remains readable without hiding the video."""
    x, y = max(0, x), max(0, y)
    w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = img[y:y + h, x:x + w]
    layer = np.zeros_like(roi)
    rounded_rect(layer, 0, 0, w - 1, h - 1, r, color, -1)
    m = np.zeros((h, w), np.uint8)
    rounded_rect(m, 0, 0, w - 1, h - 1, r, 255, -1)
    a = (m.astype(np.float32) / 255.0 * alpha)[..., None]
    roi[:] = (roi * (1 - a) + layer * a).astype(np.uint8)


def text(img, s, org, scale, color, thick=1, font=F, center=False, right=False):
    (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
    x, y = org
    if center:
        x -= tw // 2
    if right:
        x -= tw
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)
    return tw, th


def glow_contour(img, cnt, color, core=2, spread=13, strength=0.45):
    """Halo contour. The halo detaches the part from the background without hiding its edges."""
    layer = np.zeros_like(img)
    cv2.drawContours(layer, [cnt], -1, color, spread, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), spread / 2.2)
    cv2.add(img, (layer * strength).astype(np.uint8), img)
    cv2.drawContours(img, [cnt], -1, color, core, cv2.LINE_AA)


def corner_brackets(img, box, color, frac=0.22, thick=3, pad=14):
    """Viewfinder corner brackets: marks the zone without heavily framing it."""
    x, y, w, h = box
    x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
    L = int(min(w, h) * frac)
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1),
                             (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (px, py), (px + dx * L, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, thick, cv2.LINE_AA)


# ------------------------------------------------------------- le badge
def type_badge(img, label, color, confidence, anim=1.0, subtitle=None):
    """The main banner. This is what must be readable from across the room."""
    H, W = img.shape[:2]
    # unit = 1.0 at 720p. We scale on the SMALLEST side: otherwise a portrait
    # frame gives a banner that devours the screen.
    u = min(W, H) / 720.0
    scale = u
    big = 2.3 * u
    big *= 0.90 + 0.10 * anim                      # small spring effect on appearance

    pad_x, pad_y = int(40 * u), int(20 * u)
    max_bw = int(W * 0.94)
    # text shrinks to fit: "CALIBRATION" must not overflow
    # from the banner like "TYPE 2" would
    for _ in range(40):
        (tw, th), _ = cv2.getTextSize(label, F, big, max(2, int(big * 2)))
        if tw + 2 * pad_x <= max_bw or big <= 0.5:
            break
        big *= 0.93
    (tw, th), _ = cv2.getTextSize(label, F, big, max(2, int(big * 2)))
    bw = min(max(tw + 2 * pad_x, int(W * 0.28)), max_bw)
    bh = th + 2 * pad_y + int(30 * u)
    bx, by = (W - bw) // 2, int(H * 0.030)

    panel(img, bx, by, bw, bh, r=int(14 * u), alpha=0.80 * anim + 0.05)
    # type color border: the visual identity of the part
    rounded_rect(img, bx, by, bw - 1, bh - 1, int(14 * u), color, 2)
    # color band on the left, more readable than a simple frame
    cv2.rectangle(img, (bx + 3, by + int(9 * u)),
                  (bx + int(8 * u), by + bh - int(9 * u)), color, -1, cv2.LINE_AA)

    ty = by + pad_y + th
    text(img, label, (W // 2, ty), big, color, max(2, int(big * 2)), center=True)

    if subtitle:
        text(img, subtitle, (W // 2, ty + int(25 * u)), 0.55 * u, DIM, 1,
             font=FS, center=True)

    # confidence gauge, thin, under the text
    gx, gw = bx + pad_x, bw - 2 * pad_x
    gy = by + bh - int(12 * u)
    gh = max(3, int(5 * u))
    rounded_rect(img, gx, gy, gw, gh, gh // 2, (70, 66, 62), -1)
    fill = int(gw * float(np.clip(confidence, 0, 1)) * anim)
    if fill > gh:
        rounded_rect(img, gx, gy, fill, gh, gh // 2, color, -1)
    return by + bh


def status_bar(img, items):
    """Discreet status bar bottom left: (text, color)."""
    H, W = img.shape[:2]
    s = 0.46 * max(max(W, H) / 1280.0, 0.8)
    widths = [cv2.getTextSize(t, FS, s, 1)[0][0] for t, _ in items]
    u = max(max(W, H) / 1280.0, 0.8)
    gap = int(26 * u)
    bw = sum(widths) + gap * (len(items) - 1) + 2 * gap
    bh = int(40 * u)
    bx, by = int(0.02 * W), H - bh - int(0.025 * H)
    panel(img, bx, by, bw, bh, r=bh // 2, alpha=0.72)
    x = bx + gap
    for (t, col), tw in zip(items, widths):
        cv2.circle(img, (x - int(gap * 0.42), by + bh // 2), max(3, int(3.5 * s / 0.46)),
                   col, -1, cv2.LINE_AA)
        text(img, t, (x, by + bh // 2 + int(5 * s / 0.46)), s, INK, 1, font=FS)
        x += tw + gap
