"""
segmentation.py — extraction de la pièce, robuste aux effets de lumière.

Le problème réel : sur une table brillante, une nappe de reflet spéculaire est
*plus claire* que la pièce, et si la lampe est chaude elle est aussi *beige* que
la pièce. Aucun seuillage sur l'intensité ou la couleur ne peut trancher.

Ce qui tranche, mesuré sur les vraies photos :

                            pièce     reflet chaud    séparation
    chromaticité b*          9.10         8.51           1.07x   <- inutile
    netteté de bord        137.9         25.9            5.3x    <- décisif
    texture (grain sable)    0.045        0.0085         5.3x    <- décisif >=720p

Un reflet est une nappe *lisse* avec un bord *flou* : c'est une propriété
géométrique de la lumière, pas de sa couleur, donc elle survit à tout changement
d'éclairage. Une pièce a un bord franc et une surface granuleuse.

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
    """Toutes les cartes d'indices, calculées une fois par image."""

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

        # chromaticité recentrée sur la médiane de l'image = balance des blancs
        # implicite, donc insensible à la *couleur* de la lampe
        self.warm = bstar - np.median(bstar)

        # rétinex : L divisé par sa version très floue. Une nappe de lumière
        # lisse disparaît (ratio ~1), un objet reste (ratio >> 1).
        # Le flou large se calcule sur une miniature : un flou gaussien de
        # sigma 200 px coute 160 ms en pleine resolution, 3 ms ici.
        self.ratio = self.L / (_big_blur(self.L, 0.28 * side) + 1e-3)

        # texture : coefficient de variation local. Rapport sigma/mu, donc
        # invariant à un changement multiplicatif d'éclairage.
        g = cv2.GaussianBlur(self.L, (0, 0), 0.7)
        kk = 5
        mu = cv2.boxFilter(g, -1, (kk, kk))
        mu2 = cv2.boxFilter(g * g, -1, (kk, kk))
        self.tex = np.sqrt(np.maximum(mu2 - mu * mu, 0)) / (mu + 1e-3)

        # contraste local egalise : quand une nappe speculaire delave une moitie
        # de l'image, un seuillage global n'a plus de point de fonctionnement,
        # alors qu'un contraste recalcule par tuiles retrouve la piece.
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(
            np.clip(self.L, 0, 255).astype(np.uint8)).astype(np.float32)

        # gradient, pour la netteté de bord
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
    """Copie demi-resolution des cartes d'indices : la notation d'un candidat
    n'a pas besoin du plein detail, et cela divise le cout par ~4."""

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
    """Flou gaussien de grand rayon, calcule sur une miniature puis re-agrandi."""
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
    """Seuillage par hysteresis : le remede a une piece coupee en deux par une
    ombre.

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
    # un label est conserve s'il contient au moins un pixel germe
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
    """Plusieurs masques binaires candidats, issus d'indices indépendants."""
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
    # fusion : un point est "pièce" s'il est à la fois chaud, contrasté et granuleux
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
        # part du contour qui s'appuie sur le bord du cadre : effleurer un coin
        # n'est pas la meme chose que perdre la moitie de la piece
        b = np.zeros((h, w), np.uint8)
        b[:2, :] = b[-2:, :] = b[:, :2] = b[:, -2:] = 1
        edge = (cv2.dilate(filled, np.ones((3, 3), np.uint8)) - cv2.erode(
            filled, np.ones((3, 3), np.uint8))) > 0
        bf = float(np.count_nonzero(edge & (b > 0)) / max(np.count_nonzero(edge), 1))
        out.append((filled, bf))

    # Une ombre qui traverse la piece la coupe en deux composantes. Chacune,
    # notee seule, donne une silhouette partielle - et une equerre amputee de sa
    # tete ressemble a un cylindre. On PROPOSE donc aussi leur union, pontee par
    # une fermeture proportionnelle a la taille de l'objet. C'est un candidat de
    # plus, pas une substitution : l'arbitrage tranche.
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
    """Note de plausibilité d'un candidat. Les indices choisis sont ceux qu'une
    nappe de lumière ne peut pas imiter."""
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

    # 1. netteté de bord — LE discriminant. Un reflet a un bord flou.
    sharp = float(c.grad[band].mean()) / c.grad_ref
    s_sharp = float(np.clip(sharp / 0.35, 0, 1))

    # 2. grain de la surface, rapporté au fond local (invariant à l'éclairage).
    #    Fiable seulement si la pièce est assez grande en pixels. Le rapport est
    #    plafonné : au-delà de ~4x on est sur un artefact (bord d'ombre), pas sur
    #    une surface granuleuse.
    t_in = float(np.median(c.tex[inside]))
    t_bg = float(np.median(c.tex[ring]))
    texr = min(t_in / (t_bg + 1e-4), 6.0)
    s_tex = float(np.clip((texr - 1.0) / 1.5, 0, 1))
    w_tex = float(np.clip(np.sqrt(area) / 150.0, 0.25, 1.0))

    # 3. contraste de chromaticité avec le fond local
    s_warm = float(np.clip((c.warm[inside].mean() - c.warm[ring].mean()) / 8.0, 0, 1))

    # 4. contraste de clarté avec le fond local
    s_lum = float(np.clip((c.L[inside].mean() - c.L[ring].mean()) / 50.0, 0, 1))

    # 5. plausibilité géométrique : ni poussière, ni demi-image
    frac = area / (c.h * c.w)
    s_geo = float(np.clip(min(frac / 0.012, 1.0), 0, 1) * np.clip((0.55 - frac) / 0.15, 0, 1))

    # 5b. aire attendue, apprise sur les poses enrolees. Sans elle, le score
    #     sature : un blob de 3 % et la vraie piece a 13 % obtiennent la meme
    #     note sur tous les autres criteres.
    s_area = 1.0
    if area_range is not None:
        lo, hi = area_range
        f = max(frac, 1e-6)
        if f < lo:
            s_area = float(np.exp(-((np.log(lo / f)) ** 2) / (2 * 0.45 ** 2)))
        elif f > hi:
            s_area = float(np.exp(-((np.log(f / hi)) ** 2) / (2 * 0.45 ** 2)))

    # 6. épaisseur : un liseré d'ombre ou un bord de reflet est une bande de
    #    quelques pixels. Une pièce a une épaisseur du même ordre que sa taille.
    #    C'est ce qui empêche un artefact fin de gagner sur des indices locaux.
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    thick = float(dt.max()) / max(np.sqrt(area), 1e-6)
    s_thick = float(np.clip((thick - 0.035) / 0.055, 0, 1))

    total = (0.38 * s_sharp
             + 0.20 * w_tex * s_tex
             + 0.14 * s_warm
             + 0.10 * s_lum
             + 0.08 * s_geo
             + 0.10 * s_thick)
    # netteté de bord, géométrie et épaisseur sont éliminatoires : un candidat
    # nul sur l'une des trois ne peut pas être rattrapé par les autres
    total *= (0.35 + 0.65 * s_sharp) * (0.30 + 0.70 * s_geo) * (0.15 + 0.85 * s_thick)
    total *= (0.20 + 0.80 * s_area)      # l'aire attendue est eliminatoire
    # Une piece qui touche le bord reste recevable - sinon on jette des masques
    # parfaits juste parce que la piece est grande dans le cadre. La penalite
    # est PROPORTIONNELLE a la portion de contour reellement coupee : effleurer
    # un bord ne doit pas faire perdre contre un candidat interieur mediocre.
    total *= (1.0 - 0.55 * float(np.clip(border_frac / 0.25, 0, 1)))
    return float(total), dict(sharp=sharp, texr=texr, s_sharp=s_sharp, s_tex=s_tex,
                              s_warm=s_warm, s_lum=s_lum, s_geo=s_geo, frac=frac,
                              thick=thick, s_thick=s_thick, s_area=s_area, border_frac=float(border_frac),
                              truncated=bool(border_frac > 0.04))


# ------------------------------------------------------- C. affinage
def refine(c: Cues, mask):
    """Re-seuillage local. Un Otsu global est tiré par le reflet ; restreint au
    voisinage de la pièce, il retrouve le vrai bord."""
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
    # ne garder que ce qui recouvre le candidat d'origine
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
    """(mask, contour) dans le repère de l'image d'entrée, ou (None, None)."""
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
            # dedup bon marche : bounding box + aire, avant tout calcul lourd
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
# invariante a l'eclairage
def segment_bg(bgr, background, thr=0.16, chroma_thr=6.0, min_frac=0.002,
               reject_border=True, debug=False):
    """Segmentation par comparaison a une image de la scene vide.

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

    # Le champ d'eclairage doit etre estime SANS l'objet : sinon le signal fort
    # de la piece fuit dans le flou et fabrique un halo qui noie le contraste.
    # 1er passage : ou est grossierement l'objet ?
    rough = (np.abs(R) > 0.55) | (np.hypot(da, db) > 14.0)
    rough = cv2.dilate(rough.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2) > 0
    if rough.mean() > 0.75:          # garde-fou : tout ne peut pas etre objet
        rough[:] = False

    def smooth_field(x):
        """Champ lisse estime par convolution normalisee, objet exclu."""
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
    """Segmentation de la piece. C'est la fonction a appeler.

    Strategie : si un modele de fond existe, on l'utilise (le plus precis et le
    plus rapide) ; s'il est perime - l'eclairage a change depuis la calibration,
    le resultat devient implausible - on bascule automatiquement sur la voie
    multi-indices, qui ne depend d'aucune calibration.

    roi = (x, y, w, h) : zone de travail. Tout ce qui est hors zone est ignore
    AVANT toute analyse. C'est le reglage le plus rentable quand la scene
    contient des distracteurs clairs (un sol, un carton, un plan de travail) :
    ils ne peuvent plus concourir.
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

    # on remet le masque et le contour dans le repere de l'image complete
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
            if sc >= 0.22:                       # le fond tient encore
                if debug:
                    dbg.update(source="background", score=sc)
                    return res[0], res[1], dbg
                return res[0], res[1]
    return segment_cues(bgr, background=None, roi=None, debug=debug,
                        do_refine=do_refine)


def background_is_stale(bgr, background, frac=0.45):
    """True si le modele de fond ne decrit plus la scene (eclairage change,
    camera bougee). Sert a declencher une recalibration."""
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
    """Renvoie les K meilleurs masques CANDIDATS au lieu d'un seul.

    Pourquoi : le score de plausibilite ne sait pas dire si un masque represente
    l'objet ENTIER. Une equerre amputee de sa tete par une ombre reste un
    candidat parfaitement "plausible" - bord net, epaisseur correcte, aire dans
    la plage - et ressemble alors a un cylindre. Aucun critere local ne peut
    trancher.

    Ce qui peut trancher, c'est le MODELE : un morceau ne ressemble a aucune
    reference, la piece entiere si. On laisse donc la reconnaissance arbitrer
    (voir PartClassifier.classify_candidates). Cette fonction se contente de
    proposer, en conservant la note de plausibilite comme garde-fou.

    Renvoie [(mask, contour, plausibilite)], du plus plausible au moins.
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
