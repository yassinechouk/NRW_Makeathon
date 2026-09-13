"""
part_classifier.py — One-shot silhouette classifier for industrial parts (noyaux de robinet).

Designed for: ONE reference image per type, single part on a contrasted background,
top view, real-time on CPU. No training, no GPU, no dataset.

Pipeline
--------
  frame -> segment()        : LAB (L AND b) dual-Otsu, or background subtraction
        -> ShapeSignature   : scale/translation normalised silhouette + descriptors
        -> match()          : rotation found analytically (FFT circular correlation of
                              the radial signature), verified by silhouette IoU,
                              fused with invariant moment/geometry descriptors
        -> (label, confidence, per-class scores)

Everything is invariant to translation, in-plane rotation, scale and (optionally)
mirroring, which is what you need when a part is dropped on a table in any pose.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# ----------------------------------------------------------------------------- config
CANVAS = 200          # size of the normalised silhouette canvas (px)
TARGET_SQRT_AREA = 52 # sqrt(area) the part is rescaled to inside the canvas.
                      # Sized so that even an elongated part (max radius up to
                      # ~1.9 x sqrt(area)) fits without being clipped - clipping
                      # a silhouette silently destroys the overlap score.
NRAYS = 256           # angular resolution of the radial signature
NHARM = 24            # Fourier harmonics kept as a rotation-invariant descriptor
CHAMFER_TAU = 1.8     # px du canevas : echelle de la similarite de contour.
                      # Plus petit = plus severe sur les ecarts de forme. Regle
                      # a 1.8 par balayage : c'est la valeur qui supprime la
                      # confusion type1/type4 (1 erreur -> 0 sur 160 essais).


# ============================================================== 1. SEGMENTATION
# La segmentation vit dans segmentation.py : c'est la partie difficile (les
# effets de lumiere), et elle merite son propre module et son propre banc d'essai.
from segmentation import segment, segment_bg, segment_cues, background_is_stale  # noqa: E402,F401


# ======================================================== 2. SHAPE SIGNATURE
@dataclass
class ShapeSignature:
    canvas: np.ndarray      # CANVAS x CANVAS uint8 normalised silhouette
    radial: np.ndarray      # NRAYS radial signature (mean-normalised)
    fdesc: np.ndarray       # NHARM rotation-invariant Fourier magnitudes
    geom: np.ndarray        # 6 geometric ratios
    hu: np.ndarray          # 7 log-Hu moments
    area_px: float          # raw pixel area (scale cue, only if camera is fixed)
    area_frac: float = -1.0 # aire rapportee a celle du cadre analyse. C'est
                            # CETTE grandeur qui se compare d'une prise de vue a
                            # l'autre, pas les pixels.

    def descriptor(self):
        return np.concatenate([self.fdesc, self.geom, self.hu])


def _normalise_mask(mask, cnt):
    """Translate to centroid + rescale by sqrt(area) into a fixed canvas."""
    M = cv2.moments(cnt)
    area = M["m00"]
    if area <= 1:
        return None, 0.0
    cx, cy = M["m10"] / area, M["m01"] / area
    s = TARGET_SQRT_AREA / np.sqrt(area)
    # guard against clipping for unusually elongated parts
    rmax = np.max(np.hypot(cnt[:, 0, 0] - cx, cnt[:, 0, 1] - cy))
    s = min(s, (CANVAS / 2 - 2) / max(rmax, 1e-6))
    A = np.float32([[s, 0, CANVAS / 2 - s * cx],
                    [0, s, CANVAS / 2 - s * cy]])
    norm = cv2.warpAffine(mask, A, (CANVAS, CANVAS), flags=cv2.INTER_NEAREST)
    return norm, area


def _radial_signature(norm):
    """Distance centroid->outer boundary as a function of angle (NRAYS samples).

    A rotation of the part = a circular shift of this vector, which is what lets
    us recover the angle analytically instead of brute-forcing it.
    """
    c = CANVAS / 2.0
    ys, xs = np.nonzero(norm)
    if len(xs) == 0:
        return np.zeros(NRAYS, np.float32)
    dx, dy = xs - c, ys - c
    ang = np.arctan2(dy, dx)
    rad = np.hypot(dx, dy)
    idx = ((ang + np.pi) / (2 * np.pi) * NRAYS).astype(np.int32) % NRAYS
    sig = np.zeros(NRAYS, np.float32)
    np.maximum.at(sig, idx, rad)           # outer boundary per angular bin
    # fill empty bins by circular interpolation
    z = sig == 0
    if z.any() and not z.all():
        good = np.nonzero(~z)[0]
        sig[z] = np.interp(np.nonzero(z)[0], good, sig[good], period=NRAYS)
    m = sig.mean()
    return sig / m if m > 0 else sig


def _geometry(cnt, norm):
    A = cv2.contourArea(cnt)
    P = max(cv2.arcLength(cnt, True), 1e-6)
    hull = cv2.convexHull(cnt)
    HA = max(cv2.contourArea(hull), 1e-6)
    HP = max(cv2.arcLength(hull, True), 1e-6)
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    w, h = max(w, h), max(min(w, h), 1e-6)
    ecc = 0.0
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        MA, ma = max(MA, ma), max(min(MA, ma), 1e-6)
        ecc = np.sqrt(max(0.0, 1 - (ma / MA) ** 2))
    return np.array([
        w / h,                       # elongation
        A / (w * h),                 # extent (fill of the oriented box)
        A / HA,                      # solidity (concavity)
        4 * np.pi * A / P ** 2,      # circularity
        P / HP,                      # boundary roughness
        ecc,                         # eccentricity
    ], np.float32)


def signature_from_mask(mask, cnt, frame_area=None) -> "ShapeSignature | None":
    norm, area = _normalise_mask(mask, cnt)
    if norm is None:
        return None
    if frame_area is None:
        frame_area = mask.shape[0] * mask.shape[1]
    radial = _radial_signature(norm)
    F = np.abs(np.fft.rfft(radial))[1:1 + NHARM]
    F = F / max(F[0], 1e-6)                       # rotation- & scale-invariant
    hu = cv2.HuMoments(cv2.moments(norm)).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return ShapeSignature(norm, radial, F.astype(np.float32),
                          _geometry(cnt, norm), hu.astype(np.float32), float(area),
                          float(area) / max(float(frame_area), 1.0))


# ================================================================ 3. MATCHING
def _best_shifts(a, b, k=3):
    """Top-k circular shifts aligning signature `b` onto `a` (FFT correlation)."""
    A = np.fft.rfft(a - a.mean())
    B = np.fft.rfft(b - b.mean())
    corr = np.fft.irfft(A * np.conj(B), n=len(a))
    order = np.argsort(corr)[::-1]
    picked = []
    for s in order:
        if all(min(abs(s - p), len(a) - abs(s - p)) > len(a) // 16 for p in picked):
            picked.append(int(s))
        if len(picked) == k:
            break
    return picked


def _iou(m1, m2):
    i = np.count_nonzero(m1 & m2)
    u = np.count_nonzero(m1 | m2)
    return i / u if u else 0.0


_ROT_CACHE: dict = {}
_K3 = np.ones((3, 3), np.uint8)


def _rotate(norm, deg):
    key = round(float(deg) % 360, 2)
    R = _ROT_CACHE.get(key)
    if R is None:
        R = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), key, 1.0)
        _ROT_CACHE[key] = R
    return cv2.warpAffine(norm, R, (CANVAS, CANVAS), flags=cv2.INTER_NEAREST)


def _edges(mask_bool):
    """Bord de la silhouette par gradient morphologique : plus rapide et plus
    stable qu'un Canny sur une image deja binaire."""
    m = mask_bool.astype(np.uint8)
    return (cv2.dilate(m, _K3) - cv2.erode(m, _K3)) > 0


def _dt(edges_bool):
    """Distance a l'ensemble des bords, en pixels du canevas."""
    return cv2.distanceTransform((~edges_bool).astype(np.uint8), cv2.DIST_L2, 3)


def chamfer_similarity(query_mask, ref_mask, dt_query=None, tau=CHAMFER_TAU):
    """Similarite fondee sur la distance moyenne symetrique entre les DEUX
    CONTOURS, une fois les formes alignees.

    Pourquoi ce terme existe : l'IoU d'aire est presque aveugle a un petit
    appendice. Entre le type 1 et le type 4, qui ne different que par un ergot
    en pied, l'IoU ne separe que de 0.16 ; le Chamfer separe de 0.53 (mesure sur
    20 vues degradees). C'est ce qui rend deux sosies distinguables.
    """
    eq = _edges(query_mask)
    er = _edges(ref_mask)
    if eq.sum() < 5 or er.sum() < 5:
        return 0.0
    if dt_query is None:
        dt_query = _dt(eq)
    d = 0.5 * (float(dt_query[er].mean()) + float(_dt(er)[eq].mean()))
    return float(np.exp(-d / tau))


def _mirror_radial(radial):
    """Radial signature of the horizontally flipped shape."""
    return np.roll(radial[::-1], 1 + NRAYS // 2).copy()


def align_iou(query: ShapeSignature, ref: ShapeSignature, allow_mirror=True,
              fallback_below=0.60, fast_only=False):
    """Best silhouette overlap after recovering the in-plane rotation.

    Fast path: the rotation is read off the circular cross-correlation of the two
    radial signatures (top-3 peaks), then refined locally.
    Safety net: if the best overlap stays poor the angle may have been missed, so
    a coarse exhaustive sweep is run before concluding it is a different part.
    """
    qb = (query.canvas > 0)
    best, best_deg, best_mir, best_aligned = 0.0, 0.0, False, None
    variants = [(ref.canvas, ref.radial, False)]
    if allow_mirror:
        variants.append((cv2.flip(ref.canvas, 1), _mirror_radial(ref.radial), True))

    def try_angles(rcanvas, mir, angles):
        nonlocal best, best_deg, best_mir, best_aligned
        for deg in angles:
            rot = _rotate(rcanvas, -deg) > 0
            v = _iou(qb, rot)
            if v > best:
                best, best_deg, best_mir, best_aligned = v, float(deg % 360), mir, rot

    for rcanvas, rradial, mir in variants:
        for s in _best_shifts(query.radial, rradial, k=3):
            deg0 = s * 360.0 / NRAYS
            try_angles(rcanvas, mir, np.arange(deg0 - 6, deg0 + 6.1, 1.5))

    if best < fallback_below and not fast_only:
        for rcanvas, _rr, mir in variants:
            try_angles(rcanvas, mir, np.arange(0, 360, 6.0))
        # refine around whatever the sweep found
        rcanvas = variants[1][0] if (best_mir and allow_mirror) else variants[0][0]
        try_angles(rcanvas, best_mir, np.arange(best_deg - 6, best_deg + 6.1, 1.5))

    return best, best_deg, best_mir, best_aligned


fallback_all_below = 0.60


class PartClassifier:
    """Holds the reference signatures and classifies a new frame."""

    # Trois evidences independantes, fusionnees. Le terme Chamfer porte le poids
    # principal : c'est lui qui separe deux pieces qui ne different que par un
    # detail local, la ou l'IoU d'aire est presque aveugle.
    W_IOU = 0.32          # recouvrement d'aire  - la forme d'ensemble
    W_CHAMFER = 0.43      # distance des contours - les details locaux
    W_DESC = 0.25         # descripteurs invariants - moments et harmoniques
    # seuil absolu d'acceptation sur le score fusionne
    MIN_SCORE = 0.50
    # ecart minimal avec le second pour trancher
    MIN_MARGIN = 0.05
    # temperature du softmax de confiance (voir _confidence)
    TEMPERATURE = 0.10
    # MONDE FERME : la piece devant la camera est forcement l'un des types
    # enroles. Dans ce cas repondre UNKNOWN n'apporte rien - il faut trancher et
    # montrer a quel point c'est serre. UNKNOWN n'est alors emis que si aucune
    # piece n'a ete segmentee. Mettre a True pour retrouver le rejet (utile si
    # des pieces hors catalogue peuvent se presenter).
    OPEN_SET = False
    # Nombre de references qui passent l'alignement complet. Au-dela, le cout
    # croit lineairement avec le nombre de poses enrolees (20 refs = 198 ms).
    # Les descripteurs invariants, eux, se comparent en microsecondes : ils
    # servent de pre-filtre. Le meilleur representant de CHAQUE type est
    # toujours conserve, pour qu'aucun type ne soit ecarte sans avoir ete
    # aligne - sinon le pre-filtre pourrait decider a la place de l'appariement.
    PREFILTER_K = 4
    # Nombre de decoupages qui passent l'appariement COMPLET. Les autres sont
    # ecartes par le pre-tri sur descripteurs, sans alignement.
    MAX_CANDIDATES = 3

    def __init__(self, allow_mirror=True):
        # labels et refs sont paralleles ; un meme label peut apparaitre
        # PLUSIEURS fois : une entree par pose stable de la piece.
        self.labels: list[str] = []
        self.refs: list[ShapeSignature] = []
        self.allow_mirror = allow_mirror
        self._scales = None

    @property
    def classes(self):
        """Types distincts, dans l'ordre d'enrolement (une pose = pas un type)."""
        out = []
        for l in self.labels:
            if l not in out:
                out.append(l)
        return out

    def area_frac_range(self, widen=1.7):
        """Plage d'aire plausible, apprise sur les poses enrolees.

        C'est l'information qui manquait au score de segmentation : netteté,
        épaisseur et géométrie saturent toutes a 1.0 sur des candidats absurdes,
        et c'est l'AIRE qui separe un blob de 3 % de la vraie piece a 13 %.
        """
        v = sorted(r.area_frac for r in self.refs if getattr(r, "area_frac", -1) > 0)
        if len(v) < 3:
            return None
        # percentiles plutot que min/max : une seule reference cadree autrement
        # ne doit pas ouvrir la plage au point de la rendre inutile
        lo = float(np.percentile(v, 10)) / widen
        hi = float(np.percentile(v, 90)) * widen
        return (lo, hi)

    def poses(self, label):
        return [i for i, l in enumerate(self.labels) if l == label]

    # ---- enrolment -------------------------------------------------------
    def add_reference(self, label, bgr, background=None, roi=None):
        mask, cnt = segment(bgr, background=background, roi=roi)
        if mask is None:
            raise ValueError(f"no part found in the reference image for '{label}'")
        sig = signature_from_mask(mask, cnt)
        self.labels.append(label)
        self.refs.append(sig)
        self._fit_scales()
        return sig

    def _fit_scales(self):
        """Per-feature scaling from the spread across the enrolled types, so that
        every descriptor contributes on a comparable footing."""
        if len(self.refs) < 2:
            self._scales = None
            return
        D = np.stack([r.descriptor() for r in self.refs])
        s = D.std(axis=0)
        s[s < 1e-6] = 1e-6
        self._scales = s

    # ---- inference -------------------------------------------------------
    def classify_signature(self, sig: ShapeSignature):
        scores = {}
        details = {}
        dq = sig.descriptor()

        # pre-filtre par descripteurs invariants : quasi gratuit, il elague les
        # poses sans rapport avant les alignements, qui sont le vrai cout
        ddist = np.array([self._desc_distance(dq, r) for r in self.refs])
        keep = set(np.argsort(ddist)[:self.PREFILTER_K].tolist())
        for cls in self.classes:                      # filet de securite
            idx = self.poses(cls)
            keep.add(int(min(idx, key=lambda i: ddist[i])))

        # pass 1: cheap angle recovery, sur les seules references retenues
        fast = [align_iou(sig, r, self.allow_mirror, fast_only=True)
                if i in keep else (0.0, 0.0, False, None)
                for i, r in enumerate(self.refs)]
        best_fast = max((f[0] for f in fast), default=0.0)
        dt_query = _dt(_edges(sig.canvas > 0))
        # pass 2: pay for the exhaustive angle sweep only where it can change the
        # ranking - i.e. the plausible candidates, or nothing matched at all
        for i, (label, ref) in enumerate(zip(self.labels, self.refs)):
            if i not in keep:
                continue
            if fast[i][0] >= best_fast - 0.12 or best_fast < fallback_all_below:
                iou, deg, mir, aligned = align_iou(sig, ref, self.allow_mirror)
            else:
                iou, deg, mir, aligned = fast[i]
            cham = (chamfer_similarity(sig.canvas > 0, aligned, dt_query)
                    if aligned is not None else 0.0)
            d = self._desc_distance(dq, ref)
            dsim = float(np.exp(-d))                     # 1.0 = identical
            fused = self.W_IOU * iou + self.W_CHAMFER * cham + self.W_DESC * dsim
            # plusieurs poses par type : on retient la MEILLEURE. Une piece ne
            # peut etre posee que d'une facon a la fois ; les autres poses du
            # meme type ne sont pas des concurrentes, ce sont des alternatives.
            if fused > scores.get(label, -1.0):
                scores[label] = float(fused)
                details[label] = dict(iou=float(iou), chamfer_sim=float(cham),
                                      desc_sim=dsim, angle=float(deg),
                                      mirrored=bool(mir), pose=i)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, second = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0.0))
        margin = best[1] - second[1]
        ok = best[1] >= self.MIN_SCORE and margin >= self.MIN_MARGIN
        conf = self._confidence(ranked)
        # en monde ferme on tranche toujours ; `accepted` reste calcule et sert
        # a signaler une decision peu sure au lieu de la masquer
        label = best[0] if (ok or not self.OPEN_SET) else "UNKNOWN"
        return dict(label=label,
                    raw_label=best[0], confidence=float(np.clip(conf, 0, 1)),
                    accepted=bool(ok), margin=float(margin),
                    scores=scores, details=details,
                    angle=details[best[0]]["angle"],
                    mirrored=details[best[0]]["mirrored"])

    def _desc_distance(self, dq, ref):
        """Distance sur les descripteurs invariants, normalisee par la dispersion
        observee entre les references enrolees."""
        diff = dq - ref.descriptor()
        if self._scales is not None:
            diff = diff / self._scales
        return float(np.linalg.norm(diff) / np.sqrt(len(dq)))

    def _confidence(self, ranked):
        """Confiance = probabilite du gagnant, moderee par la qualite absolue.

        Un softmax sur les scores repond a la vraie question ("a quel point le
        gagnant domine-t-il ses concurrents ?") au lieu de ne regarder que
        l'ecart avec le second. Le facteur de qualite empeche un appariement
        mediocre mais isole d'afficher 95 % : dominer trois mauvais candidats
        ne prouve rien.
        """
        vals = np.array([v for _, v in ranked], dtype=np.float64)
        if len(vals) == 1:
            return float(np.clip(vals[0], 0, 1))
        e = np.exp((vals - vals.max()) / self.TEMPERATURE)
        p = float(e[0] / e.sum())
        quality = float(np.clip(vals[0] / 0.72, 0, 1))
        return float(np.clip(p * quality, 0, 1))

    def prune_poses(self, max_similarity=0.88, verbose=True):
        """Supprime les poses redondantes d'un meme type.

        Enroler dix vues de la meme face n'apporte rien et coute cher : chaque
        pose est un alignement de plus a chaque image. On garde un representant
        par groupe de poses tres semblables.
        """
        keep_idx = []
        for cls in self.classes:
            kept = []
            for i in self.poses(cls):
                red = False
                for j in kept:
                    iou, _, _, al = align_iou(self.refs[i], self.refs[j], self.allow_mirror)
                    ch = (chamfer_similarity(self.refs[i].canvas > 0, al)
                          if al is not None else 0.0)
                    if 0.42 * iou + 0.58 * ch > max_similarity:
                        red = True
                        break
                if not red:
                    kept.append(i)
            keep_idx += kept
        keep_idx.sort()
        before = len(self.labels)
        self.labels = [self.labels[i] for i in keep_idx]
        self.refs = [self.refs[i] for i in keep_idx]
        self._fit_scales()
        if verbose:
            print(f"  poses : {before} -> {len(self.labels)}  "
                  + ", ".join(f"{c}={len(self.poses(c))}" for c in self.classes))
        return self

    def classify_candidates(self, candidates, frame_area=None):
        """Choisit CONJOINTEMENT le masque et le type.

        En monde ferme, le modele sait a quoi ressemblent les pieces : c'est
        l'arbitre naturel entre plusieurs decoupages possibles. Un masque
        partiel n'est explique par aucune reference et perd donc contre le
        masque complet, la ou aucun critere local ne pouvait les separer.

        La plausibilite de segmentation reste au produit, avec un exposant
        faible : elle empeche un decoupage absurde de gagner sur un appariement
        de hasard, sans dominer la decision.
        """
        # Pre-tri des candidats sur les descripteurs invariants : c'est
        # quasi gratuit (pas d'alignement) et cela evite de payer l'appariement
        # complet pour des decoupages manifestement absurdes.
        scored = []
        for mask, cnt, seg_score in candidates:
            sig = signature_from_mask(mask, cnt, frame_area)
            if sig is None:
                continue
            dq = sig.descriptor()
            cheap = max(np.exp(-self._desc_distance(dq, r_)) for r_ in self.refs)
            scored.append((cheap * (max(seg_score, 1e-3) ** 0.35), mask, cnt, sig, seg_score))
        scored.sort(key=lambda t: -t[0])
        scored = scored[:self.MAX_CANDIDATES]

        best = None
        for _, mask, cnt, sig, seg_score in scored:
            r = self.classify_signature(sig)
            combined = (max(seg_score, 1e-3) ** 0.35) * max(r["scores"].values())
            if best is None or combined > best[0]:
                best = (combined, r, mask, cnt, sig, seg_score)
        if best is None:
            return None, None, None
        _, r, mask, cnt, _sig, seg_score = best
        r["seg_score"] = seg_score
        r["n_candidates"] = len(candidates)
        return r, mask, cnt

    def classify(self, bgr, background=None, roi=None):
        mask, cnt = segment(bgr, background=background, roi=roi)
        if mask is None:
            return None, None, None
        sig = signature_from_mask(mask, cnt)
        return self.classify_signature(sig), mask, cnt

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            labels=np.array(self.labels),
            canvases=np.stack([r.canvas for r in self.refs]),
            radials=np.stack([r.radial for r in self.refs]),
            fdescs=np.stack([r.fdesc for r in self.refs]),
            geoms=np.stack([r.geom for r in self.refs]),
            hus=np.stack([r.hu for r in self.refs]),
            areas=np.array([r.area_px for r in self.refs]),
            area_fracs=np.array([getattr(r, "area_frac", -1.0) for r in self.refs]),
            allow_mirror=np.array([self.allow_mirror]),
        )

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        c = cls(allow_mirror=bool(z["allow_mirror"][0]))
        c.labels = [str(x) for x in z["labels"]]
        af = z["area_fracs"] if "area_fracs" in z.files else None
        c.refs = [ShapeSignature(z["canvases"][i], z["radials"][i], z["fdescs"][i],
                                 z["geoms"][i], z["hus"][i], float(z["areas"][i]),
                                 float(af[i]) if af is not None else -1.0)
                  for i in range(len(c.labels))]
        c._fit_scales()
        return c
