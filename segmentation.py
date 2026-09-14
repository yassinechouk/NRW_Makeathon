"""
segmentation.py — extraction de la pièce, robuste aux effets de lumière.

Le problème réel : sur une table brillante, une nappe de reflet spéculaire est
*plus claire* que la pièce, et si la lampe est chaude elle est aussi *beige* que
la pièce. Aucun seuillage sur l'intensité ou la couleur ne peut decide.

Ce qui tranche, mesuré sur les vraies photos :

                            pièce     reflet chaud    séparation
    chromaticité b*          9.10         8.51           1.07x   <- inutile
    netteté de bord        137.9         25.9            5.3x    <- décisif
    texture (grain sable)    0.045        0.0085         5.3x    <- décisif >=720p

Un reflet est une nappe *lisse* avec un bord *flou* : c'est une propriété
géométrique de la lumière, pas de sa couleur, donc elle survit à tout changement
d'éclairage. Une pièce a un bord franc et a grainy surface.

D'où l'architecture, en trois temps :

  A. PROPOSER   plusieurs masques candidats issus de seuillages indépendants
                (chromaticité, clarté, rétinex, texture, fusion, soustraction de
                fond). Aucun n'est fiable seul ; l'un d'eux est presque toujours bon.
  B. ARBITRER   noter chaque candidat sur des indices qu'un reflet ne peut pas
                imiter (netteté de bord, grain, plausibilité géométrique) et
                garder le meilleur.
  C. AFFINER    re-seuiller localement autour du gagnant. Un Otsu global est
                biaisé par le reflet ; un Otsu limité au voisinage de la pièce
                ne l'est pas.
"""

from __future__ import annotations

import cv2
import numpy as np

WORK_SIDE = 720


# ------------------------------------------------------------------ cue maps
class Cues:
    """All cue maps, calculated once per image."""

    def __init__(self, bgr, background=None):
        h, w = bgr.shape[:2]
        k = WORK_SIDE / max(h, w)
        self.k = k if k < 1.0 else 1.0
        if k < 1.0:
            bgr = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            if background is not None:
                background = cv2.resize(background, (bgr.shape[1], bgr.shape[0]),
                                        interpolation=cv2.INTER_AREA)
        self.bgr = bgr
        self.h, self.w = bgr.shape[:2]
        side = max(self.h, self.w)

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.L = lab[:, :, 0]
        bstar = lab[:, :, 2]

        # chromaticity recentered on the image median = implicit white balance
        # thus insensitive to the *color* of the lamp
        self.warm = bstar - np.median(bstar)

        # retinex: L divided by its very blurry version. A smooth sheet of
        # light disappears (ratio ~1), an object remains (ratio >> 1).
        # The large blur is calculated on a thumbnail: a Gaussian blur of
        # sigma 200 px costs 160 ms in full resolution, 3 ms here.
        self.ratio = self.L / (_big_blur(self.L, 0.28 * side) + 1e-3)

        # texture: local coefficient of variation. Ratio sigma/mu, thus
        # invariant to a multiplicative change in lighting.
        g = cv2.GaussianBlur(self.L, (0, 0), 0.7)
        kk = 5
        mu = cv2.boxFilter(g, -1, (kk, kk))
        mu2 = cv2.boxFilter(g * g, -1, (kk, kk))
        self.tex = np.sqrt(np.maximum(mu2 - mu * mu, 0)) / (mu + 1e-3)

        # equalized local contrast: when a specular reflection washes out half
        # the image, a global threshold no longer has an operating point,
        # whereas a contrast recalculated by tiles finds the part again.
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(
            np.clip(self.L, 0, 255).astype(np.uint8)).astype(np.float32)

        # gradient, for edge sharpness
        gs = cv2.GaussianBlur(self.L, (0, 0), 1.0)
        gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, 3)
        self.grad = np.hypot(gx, gy)
        self.grad_ref = float(np.percentile(self.grad, 99)) + 1e-3

        self._half = None
        self.bgdiff = None
        if background is not None:
            d = cv2.absdiff(cv2.GaussianBlur(bgr, (5, 5), 0),
                            cv2.GaussianBlur(background, (5, 5), 0))
            self.bgdiff = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32)


class _HalfCues:
    """Half-resolution copy of cue maps: scoring a candidate
    does not need full detail, and this divides the cost by ~4."""

    def __init__(self, c):
        self.h, self.w = c.h // 2, c.w // 2
        rs = lambda x: cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.L, self.warm, self.tex, self.grad = rs(c.L), rs(c.warm), rs(c.tex), rs(c.grad)
        self.grad_ref = c.grad_ref


def half(c: Cues):
    if c._half is None:
        c._half = _HalfCues(c)
    return c._half


def _big_blur(x, sigma, factor=8):
    """Large radius Gaussian blur, calculated on a thumbnail then upscaled."""
    h, w = x.shape[:2]
    sm = cv2.resize(x, (max(w // factor, 8), max(h // factor, 8)),
                    interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), max(sigma / factor, 0.8))
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def _otsu(x):
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros(x.shape, np.uint8)
    u8 = np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def _hysteresis(x, k_low=0.72):
    """Hysteresis thresholding: the remedy for a part cut in two by an
    shadow.

    Un seuil unique doit choisir entre garder la moitie sombre de la piece (et
    ramasser du fond) ou la perdre. L'hysteresis ne choisit pas : le seuil HAUT
    donne des germes surs, le seuil BAS donne la region candidate, et on ne garde
    que les morceaux de la region candidate qui contiennent un germe. La partie
    ombree rejoint donc la piece parce qu'elle lui est CONNEXE, sans qu'une zone
    de fond de meme luminosite soit admise.
    """
    hi = _otsu(x)
    lo, span = float(x.min()), max(float(x.max()) - float(x.min()), 1e-6)
    u8 = np.clip((x - lo) / span * 255, 0, 255).astype(np.uint8)
    t = float(cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    lo_val = lo + (t * k_low) / 255.0 * span
    low = (x > lo_val).astype(np.uint8) * 255
    n, lbl, _st, _ce = cv2.connectedComponentsWithStats(low, 8)
    if n <= 1:
        return hi
    keep = np.zeros_like(low)
    seeds = hi > 0
    # a label is kept if it contains at least one seed pixel
    ids = np.unique(lbl[seeds])
    for i in ids:
        if i:
            keep[lbl == i] = 255
    return keep


def _clean(m):
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return m


def _norm01(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


# ------------------------------------------------------- A. propositions
def propose(c: Cues):
    """Multiple candidate binary masks, from independent cues."""
    hyps = {}
    hyps["warm"] = _otsu(c.warm)
    hyps["L"] = _otsu(c.L)
    hyps["ratio"] = _otsu(c.ratio)
    hyps["warm_and_L"] = ((_otsu(c.warm) > 0) & (_otsu(c.L) > 0)).astype(np.uint8) * 255
    hyps["tex"] = _otsu(cv2.GaussianBlur(c.tex, (0, 0), 3))
    hyps["clahe"] = _otsu(c.clahe)
    hyps["hyst_L"] = _hysteresis(c.L)
    hyps["hyst_clahe"] = _hysteresis(c.clahe)
    hyps["hyst_warm"] = _hysteresis(c.warm)
    hyps["clahe_and_warm"] = (((_otsu(c.clahe) > 0) & (_otsu(c.warm) > 0))
                              .astype(np.uint8) * 255)
    # fusion: a point is "part" if it is simultaneously warm, contrasted, and grainy
    fused = (_norm01(c.warm) + _norm01(c.ratio) + _norm01(cv2.GaussianBlur(c.tex, (0, 0), 3))) / 3
    hyps["fused"] = _otsu(fused)
    if c.bgdiff is not None:
        hyps["bgdiff"] = (c.bgdiff > 25).astype(np.uint8) * 255
    return {k: _clean(v) for k, v in hyps.items()}


def _components(mask, h, w, min_frac=0.002, max_frac=0.60, allow_border=False):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (min_frac * h * w <= area <= max_frac * h * w):
            continue
        touches = (x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2)
        if touches and not allow_border:
            continue                      # pièce coupée : non mesurable
        m = ((lbl == i) * 255).astype(np.uint8)
        ff = m.copy()
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
        filled = m | cv2.bitwise_not(ff)
        # part of the contour that touches the frame edge: touching a corner
        # is not the same as losing half the part
        b = np.zeros((h, w), np.uint8)
        b[:2, :] = b[-2:, :] = b[:, :2] = b[:, -2:] = 1
        edge = (cv2.dilate(filled, np.ones((3, 3), np.uint8)) - cv2.erode(
            filled, np.ones((3, 3), np.uint8))) > 0
        bf = float(np.count_nonzero(edge & (b > 0)) / max(np.count_nonzero(edge), 1))
        out.append((filled, bf))

    # A shadow crossing the part cuts it into two components. Each,
    # scored alone, gives a partial silhouette - and a bracket amputated of its
    # head resembles a cylinder. We thus also PROPOSE their union, bridged by
    # a closure proportional to the object size. It's an additional
    # candidate, not a substitution: arbitration decides.
    if len(out) >= 2:
        union = np.zeros((h, w), np.uint8)
        for m_, _ in out:
            union |= m_
        k = int(np.clip(0.030 * max(h, w), 5, 41)) | 1
        bridged = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        n2, lbl2, st2, _ = cv2.connectedComponentsWithStats(bridged, 8)
        for i in range(1, n2):
            x, y, bw, bh, area = st2[i]
            if not (min_frac * h * w <= area <= max_frac * h * w):
                continue
            comp = ((lbl2 == i) * 255).astype(np.uint8)
            ff2 = comp.copy()
            cv2.floodFill(ff2, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
            comp = comp | cv2.bitwise_not(ff2)
            b = np.zeros((h, w), np.uint8)
            b[:2, :] = b[-2:, :] = b[:, :2] = b[:, -2:] = 1
            edge = (cv2.dilate(comp, np.ones((3, 3), np.uint8)) - cv2.erode(
                comp, np.ones((3, 3), np.uint8))) > 0
            bf2 = float(np.count_nonzero(edge & (b > 0)) / max(np.count_nonzero(edge), 1))
            out.append((comp, bf2))
    return out


# ------------------------------------------------------- B. arbitrage
def score(c: Cues, mask, border_frac=0.0, area_range=None):
    """Plausibility score of a candidate. The chosen cues are those that a
    sheet of light cannot imitate."""
    hc = half(c)
    mask = cv2.resize(mask, (hc.w, hc.h), interpolation=cv2.INTER_NEAREST)
    c = hc
    area = int(np.count_nonzero(mask))
    if area < 40:
        return 0.0, {}

    er = cv2.erode(mask, np.ones((5, 5), np.uint8))
    di = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    band = (di > 0) & (er == 0)
    ring = (cv2.dilate(mask, np.ones((21, 21), np.uint8)) > 0) & (di == 0)
    inside = er > 0
    if inside.sum() < 40 or ring.sum() < 40 or band.sum() < 15:
        return 0.0, {}

    # 1. edge sharpness — THE discriminant. A reflection has a blurry edge.
    sharp = float(c.grad[band].mean()) / c.grad_ref
    s_sharp = float(np.clip(sharp / 0.35, 0, 1))

    # 2. surface grain, relative to the local background (illumination invariant).
    #    Reliable only if the part is large enough in pixels. The ratio is
    #    capped: beyond ~4x it is an artifact (shadow edge), not
    #    a grainy surface.
    t_in = float(np.median(c.tex[inside]))
    t_bg = float(np.median(c.tex[ring]))
    texr = min(t_in / (t_bg + 1e-4), 6.0)
    s_tex = float(np.clip((texr - 1.0) / 1.5, 0, 1))
    w_tex = float(np.clip(np.sqrt(area) / 150.0, 0.25, 1.0))

    # 3. chromaticity contrast with the local background
    s_warm = float(np.clip((c.warm[inside].mean() - c.warm[ring].mean()) / 8.0, 0, 1))

    # 4. lightness contrast with the local background
    s_lum = float(np.clip((c.L[inside].mean() - c.L[ring].mean()) / 50.0, 0, 1))

    # 5. geometric plausibility: neither dust nor half-image
    frac = area / (c.h * c.w)
    s_geo = float(np.clip(min(frac / 0.012, 1.0), 0, 1) * np.clip((0.55 - frac) / 0.15, 0, 1))

    # 5b. expected area, learned from enrolled poses. Without it, the score
    #     saturates: a 3% blob and the real 13% part get the same
    #     score on all other criteria.
    s_area = 1.0
    if area_range is not None:
        lo, hi = area_range
        f = max(frac, 1e-6)
        if f < lo:
            s_area = float(np.exp(-((np.log(lo / f)) ** 2) / (2 * 0.45 ** 2)))
        elif f > hi:
            s_area = float(np.exp(-((np.log(f / hi)) ** 2) / (2 * 0.45 ** 2)))

    # 6. thickness: a shadow fringe or reflection edge is a band of
    #    a few pixels. A part has a thickness of the same order as its size.
    #    This is what prevents a thin artifact from winning on local cues.
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    thick = float(dt.max()) / max(np.sqrt(area), 1e-6)
    s_thick = float(np.clip((thick - 0.035) / 0.055, 0, 1))

    total = (0.38 * s_sharp
             + 0.20 * w_tex * s_tex
             + 0.14 * s_warm
             + 0.10 * s_lum
             + 0.08 * s_geo
             + 0.10 * s_thick)
    # edge sharpness, geometry, and thickness are eliminatory: a candidate
    # scoring zero on any of the three cannot be caught up by the others
    total *= (0.35 + 0.65 * s_sharp) * (0.30 + 0.70 * s_geo) * (0.15 + 0.85 * s_thick)
    total *= (0.20 + 0.80 * s_area)      # l'aire attendue est eliminatoire
    # A part touching the edge remains valid - otherwise perfect masks are thrown
    # away just because the part is large in the frame. The penalty
    # is PROPORTIONAL to the portion of the contour actually cut: touching
    # an edge must not cause a loss against a mediocre inner candidate.
    total *= (1.0 - 0.55 * float(np.clip(border_frac / 0.25, 0, 1)))
    return float(total), dict(sharp=sharp, texr=texr, s_sharp=s_sharp, s_tex=s_tex,
                              s_warm=s_warm, s_lum=s_lum, s_geo=s_geo, frac=frac,
                              thick=thick, s_thick=s_thick, s_area=s_area, border_frac=float(border_frac),
                              truncated=bool(border_frac > 0.04))


# ------------------------------------------------------- C. affinage
def refine(c: Cues, mask):
    """Local re-thresholding. A global Otsu is pulled by the reflection; restricted to
    the neighborhood of the part, it finds the true edge."""
    x, y, bw, bh = cv2.boundingRect(mask)
    pad = int(0.35 * max(bw, bh)) + 10
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(c.w, x + bw + pad), min(c.h, y + bh + pad)

    roi_warm = c.warm[y0:y1, x0:x1]
    roi_L = c.L[y0:y1, x0:x1]
    mw = _otsu(roi_warm) > 0
    ml = _otsu(roi_L) > 0
    local = ((mw & ml) * 255).astype(np.uint8)
    if np.count_nonzero(local) < 0.25 * np.count_nonzero(mask[y0:y1, x0:x1]):
        local = (mw * 255).astype(np.uint8)     # repli si le ET est trop strict

    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = _clean(local)
    # only keep what covers the original candidate
    n, lbl, _st, _ce = cv2.connectedComponentsWithStats(out, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        comp = lbl == i
        if np.count_nonzero(comp & (mask > 0)) > 0.10 * np.count_nonzero(comp):
            keep[comp] = 255
    if np.count_nonzero(keep) < 0.3 * np.count_nonzero(mask):
        return mask
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ff = keep.copy()
    cv2.floodFill(ff, np.zeros((c.h + 2, c.w + 2), np.uint8), (0, 0), 255)
    return keep | cv2.bitwise_not(ff)


# ------------------------------------------------------- pipeline complet
def segment_cues(bgr, background=None, roi=None, debug=False, do_refine=True,
            min_score=0.18, min_area_frac=0.012, max_area_frac=0.45,
            allow_border=True, area_range=None):
    """(mask, contour) in the coordinate system of the input image, or (None, None)."""
    full_h, full_w = bgr.shape[:2]
    ox = oy = 0
    if roi is not None:
        x, y, w, h = roi
        bgr = bgr[y:y + h, x:x + w]
        if background is not None:
            background = background[y:y + h, x:x + w]
        ox, oy = x, y
    src_h, src_w = bgr.shape[:2]

    c = Cues(bgr, background)
    best, best_score, best_dbg, best_src, best_touch = None, -1.0, {}, None, False

    seen = []
    for name, hyp in propose(c).items():
        for comp, bfrac in _components(hyp, c.h, c.w, min_area_frac,
                                       max_area_frac, allow_border):
            # cheap dedup: bounding box + area, before any heavy calculation
            x, y, bw, bh = cv2.boundingRect(comp)
            a = int(np.count_nonzero(comp))
            key = (x // 8, y // 8, bw // 8, bh // 8, a // max(a // 20, 1))
            dup = any(abs(k[0] - key[0]) <= 1 and abs(k[1] - key[1]) <= 1
                      and abs(k[2] - key[2]) <= 1 and abs(k[3] - key[3]) <= 1
                      for k in seen)
            if dup:
                continue
            seen.append(key)
            sc, dbg = score(c, comp, bfrac, area_range)
            if sc > best_score:
                best_score, best, best_dbg, best_src = sc, comp, dbg, name
                best_touch = bfrac

    if best is None or best_score < min_score:
        return (None, None, dict(reason="no plausible candidate",
                                 best_score=best_score)) if debug else (None, None)

    if do_refine:
        r = refine(c, best)
        sr, dr = score(c, r, best_touch, area_range)
        if sr >= best_score * 0.92:          # on garde l'affinage s'il ne dégrade pas
            best, best_dbg = r, dr

    mask = best
    if c.k != 1.0:
        mask = cv2.resize(mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return (None, None, {}) if debug else (None, None)
    cnt = max(cnts, key=cv2.contourArea)

    if roi is not None:
        out = np.zeros((full_h, full_w), np.uint8)
        out[oy:oy + src_h, ox:ox + src_w] = mask
        cnt = cnt + np.array([[ox, oy]])
        mask = out

    if debug:
        best_dbg["source"] = best_src
        best_dbg["score"] = best_score
        return mask, cnt, best_dbg
    return mask, cnt


# =================================================== soustraction de fond
# illumination invariant
def segment_bg(bgr, background, thr=0.16, chroma_thr=6.0, min_frac=0.002,
               reject_border=True, debug=False):
    """Segmentation by comparison to an image of the empty scene.

    Une soustraction naive (|frame - fond|) s'effondre des que l'eclairage a
    change depuis la calibration : toute la table devient "objet".

    L'astuce : on regarde le RAPPORT frame/fond, pas la difference. Un
    changement d'eclairage multiplie la scene par un champ LISSE ; un objet pose
    la modifie LOCALEMENT. En divisant le rapport par sa propre version tres
    floue, la composante lisse (donc tout l'eclairage, reflets compris)
    disparait, et il ne reste que ce qui a vraiment change.

        R      = L_frame / L_fond
        R_norm = R / flou(R)          -> ~1 partout sauf sur l'objet

    Une ombre portee est ecartee separement : elle assombrit sans changer la
    chromaticite, un objet change les deux.
    """
    h, w = bgr.shape[:2]
    k = WORK_SIDE / max(h, w)
    if k < 1.0:
        f = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        b = cv2.resize(background, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        k, f, b = 1.0, bgr, background
    H, W = f.shape[:2]

    lf = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lf = cv2.GaussianBlur(lf[:, :, 0], (0, 0), 1.2) + 1.0
    Lb = cv2.GaussianBlur(lb[:, :, 0], (0, 0), 1.2) + 1.0

    R = np.log(Lf / Lb)
    da = lf[:, :, 1] - lb[:, :, 1]
    db = lf[:, :, 2] - lb[:, :, 2]

    # The lighting field must be estimated WITHOUT the object: otherwise the strong signal
    # of the part leaks into the blur and creates a halo that drowns the contrast.
    # 1st pass: where is the object roughly?
    rough = (np.abs(R) > 0.55) | (np.hypot(da, db) > 14.0)
    rough = cv2.dilate(rough.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2) > 0
    if rough.mean() > 0.75:          # garde-fou : tout ne peut pas etre objet
        rough[:] = False

    def smooth_field(x):
        """Smooth field estimated by normalized convolution, object excluded."""
        wgt = (~rough).astype(np.float32)
        xs = x * wgt
        sm = max(H, W) / 8.0
        num = cv2.resize(cv2.GaussianBlur(cv2.resize(xs, (W // 8 + 1, H // 8 + 1),
                                                     interpolation=cv2.INTER_AREA),
                                          (0, 0), sm / 8.0), (W, H))
        den = cv2.resize(cv2.GaussianBlur(cv2.resize(wgt, (W // 8 + 1, H // 8 + 1),
                                                     interpolation=cv2.INTER_AREA),
                                          (0, 0), sm / 8.0), (W, H))
        return num / np.maximum(den, 1e-3)

    mag = np.abs(R - smooth_field(R))
    dchroma = np.hypot(da - smooth_field(da), db - smooth_field(db))

    m = ((mag > thr) | (dchroma > chroma_thr)).astype(np.uint8) * 255
    m = _clean(m)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)

    n_, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best, ba = None, 0
    for i in range(1, n_):
        x, y, bw, bh, area = stats[i]
        if area < min_frac * H * W:
            continue
        if reject_border and (x <= 2 or y <= 2 or x + bw >= W - 2 or y + bh >= H - 2):
            continue
        if area > ba:
            ba, best = area, i
    if best is None:
        return (None, None, {}) if debug else (None, None)

    mask = ((lbl == best) * 255).astype(np.uint8)
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 255)
    mask = mask | cv2.bitwise_not(ff)
    if k != 1.0:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return (None, None, {}) if debug else (None, None)
    cnt = max(cnts, key=cv2.contourArea)
    return (mask, cnt, {}) if debug else (mask, cnt)


# =================================================== point d'entree unique
def segment(bgr, background=None, roi=None, debug=False, do_refine=True,
            area_range=None):
    """Part segmentation. This is the function to call.

    Strategy: if a background model exists, use it (most precise and
    fastest); if it's stale - lighting changed since calibration,
    the result becomes implausible - we automatically fallback to the
    multi-cue path, which depends on no calibration.

    roi = (x, y, w, h) : work zone. Everything outside the zone is ignored
    BEFORE any analysis. This is the most profitable setting when the scene
    contains bright distractors (a floor, a box, a workbench):
    they can no longer compete.
    """
    full_h, full_w = bgr.shape[:2]
    ox = oy = 0
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        x, y = max(0, x), max(0, y)
        w, h = min(w, full_w - x), min(h, full_h - y)
        if w < 20 or h < 20:
            return (None, None, {}) if debug else (None, None)
        bgr = bgr[y:y + h, x:x + w]
        if background is not None:
            background = background[y:y + h, x:x + w]
        ox, oy = x, y

    out = _segment_core(bgr, background, debug, do_refine, area_range)
    if roi is None or out[0] is None:
        return out

    # put the mask and contour back into the full image coordinate system
    mask = np.zeros((full_h, full_w), np.uint8)
    mask[oy:oy + bgr.shape[0], ox:ox + bgr.shape[1]] = out[0]
    cnt = out[1] + np.array([[ox, oy]])
    return (mask, cnt, out[2]) if debug else (mask, cnt)


def _segment_core(bgr, background, debug, do_refine, area_range=None):
    if background is not None:
        res = segment_bg(bgr, background, debug=False)
        if res[0] is not None:
            c = Cues(bgr)
            m = cv2.resize(res[0], (c.w, c.h), interpolation=cv2.INTER_NEAREST)
            sc, dbg = score(c, m)
            if sc >= 0.22:                       # the background still holds
                if debug:
                    dbg.update(source="background", score=sc)
                    return res[0], res[1], dbg
                return res[0], res[1]
    return segment_cues(bgr, background=None, roi=None, debug=debug,
                        do_refine=do_refine)


def background_is_stale(bgr, background, frac=0.45):
    """True if the background model no longer describes the scene (changed lighting,
    camera moved). Used to trigger a recalibration."""
    h, w = bgr.shape[:2]
    k = 240 / max(h, w)
    f = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    b = cv2.resize(background, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_AREA)
    d = cv2.absdiff(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    return float((d > 30).mean()) > frac


# ============================================ candidats multiples
def segment_candidates(bgr, background=None, roi=None, area_range=None,
                       top_k=8, min_score=0.10, do_refine=True):
    """Returns the top K CANDIDATE masks instead of just one.

    Why: the plausibility score cannot tell if a mask represents
    the ENTIRE object. A bracket amputated of its head by a shadow remains a
    perfectly "plausible" candidate - sharp edge, correct thickness, area in
    range - and thus resembles a cylinder. No local criterion can
    decide.

    What can decide is the MODEL: a piece doesn't resemble any
    reference, the whole part does. So we let the recognition arbitrate
    (see PartClassifier.classify_candidates). This function just
    proposes, keeping the plausibility score as a safeguard.

    Returns [(mask, contour, plausibility)], from most plausible to least.
    """
    full_h, full_w = bgr.shape[:2]
    ox = oy = 0
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        x, y = max(0, x), max(0, y)
        w, h = min(w, full_w - x), min(h, full_h - y)
        if w < 20 or h < 20:
            return []
        bgr = bgr[y:y + h, x:x + w]
        if background is not None:
            background = background[y:y + h, x:x + w]
        ox, oy = x, y

    c = Cues(bgr, background)
    cands, seen = [], []
    for name, hyp in propose(c).items():
        for comp, bfrac in _components(hyp, c.h, c.w, 0.012, 0.45, True):
            x, y, bw, bh = cv2.boundingRect(comp)
            a = int(np.count_nonzero(comp))
            key = (x // 8, y // 8, bw // 8, bh // 8)
            if any(abs(k[0] - key[0]) <= 1 and abs(k[1] - key[1]) <= 1
                   and abs(k[2] - key[2]) <= 1 and abs(k[3] - key[3]) <= 1 for k in seen):
                continue
            seen.append(key)
            sc, _ = score(c, comp, bfrac, area_range)
            if sc >= min_score:
                cands.append((sc, comp))
    if background is not None:
        out = segment_bg(bgr, background, debug=False)
        if out[0] is not None:
            sc, _ = score(c, cv2.resize(out[0], (c.w, c.h),
                                        interpolation=cv2.INTER_NEAREST))
            cands.append((max(sc, 0.35), cv2.resize(
                out[0], (c.w, c.h), interpolation=cv2.INTER_NEAREST)))

    cands.sort(key=lambda t: -t[0])
    cands = cands[:top_k]
    if do_refine and cands:
        best_sc, best_m = cands[0]
        r = refine(c, best_m)
        sr, _ = score(c, r, 0.0, area_range)
        if sr >= best_sc * 0.92 and np.count_nonzero(r) > 50:
            cands.insert(0, (sr, r))
            cands = cands[:top_k]

    out = []
    for sc, m in cands:
        mm = m
        if c.k != 1.0:
            mm = cv2.resize(m, (bgr.shape[1], bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if roi is not None:
            full = np.zeros((full_h, full_w), np.uint8)
            full[oy:oy + bgr.shape[0], ox:ox + bgr.shape[1]] = mm
            mm, cnt = full, cnt + np.array([[ox, oy]])
        out.append((mm, cnt, float(sc)))
    return out
