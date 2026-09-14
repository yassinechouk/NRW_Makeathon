"""
live.py — real-time identification of the part in front of the camera.

    python live.py --model model.npz --camera 2

The screen announces one main thing largely: the TYPE. A beep signals each
stabilized decision, so you don't have to look at the screen continuously.

Once the decision is made, it is LOCKED: no more recalculation as long as
the scene has not changed. The result therefore does not flicker, and the loop remains
fluid despite an intentionally heavy analysis cycle.

The background model is the heart of the system, not an option: with a fixed camera,
a reflection on the table is STATIC, so it belongs to the background and
cancels out. This is what increases precision from 13/30 to 27/30 under degraded lighting
(see README). The app manages it alone: calibration at startup,
continuous refresh when the scene is empty, recalibration if the background
becomes stale.

Keys
    z   define the work zone (drag a rectangle, ENTER to confirm)
    c   clear the work zone
    d   show / hide the score details
    b   recalibrate the background now (empty scene)
    a   enable / disable automatic refresh
    n   ignore the background (multi-cue mode, less precise)
    m   mute / unmute the beep
    r   force a new measurement (unlocks)
    s   save a capture
    +/- increase / decrease the acceptance threshold
    q   quit
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import wave
from collections import deque, Counter

import cv2
import numpy as np

import hud
import segmentation as S
from part_classifier import PartClassifier, signature_from_mask

# The complete cycle (multi-segmentation arbitration) costs ~0.5 s: demanding 6
# concordant frames would mean a 3s wait. We therefore adapt the requirement to the
# confidence, which has proven reliable: above 0.70, 29/29 correct on real
# captures. Two concordant frames are enough then; otherwise we ask for four.
VOTE_WINDOW = 6
VOTE_MIN = 4
VOTE_MIN_SURE = 2
CONF_SURE = 0.70
# Once the decision is made, it is LOCKED: we no longer recalculate as long
# as the scene has not changed. Two reasons. First, the display: a valid
# result must not flicker or question itself. Second, the
# cost: multi-candidate arbitration is heavy (the model decides between several
# possible segmentations), which is only tenable because it only runs when
# the part changes.
CHANGE_FRAC = 0.020        # part de la zone qui doit changer pour deverrouiller
CHANGE_FRAMES = 3          # images consecutives, pour ignorer un scintillement
CALIB_FRAMES = 12
EMPTY_BEFORE_REFRESH = 25
BG_ALPHA = 0.05
ANIM_SECONDS = 0.28


# ------------------------------------------------------------------ le bip
class Beeper:
    """Bip court et non bloquant. Sur Linux on passe par aplay/paplay ; a defaut
    on se rabat sur la cloche du terminal."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.path = None
        self.player = None
        for exe in ("paplay", "aplay"):
            try:
                subprocess.run([exe, "--version"], capture_output=True, timeout=2)
                self.player = exe
                break
            except Exception:
                continue
        if self.player:
            self.path = os.path.join("/tmp", "partvision_beep.wav")
            self._write_wav(self.path)

    @staticmethod
    def _write_wav(path, freq=880.0, ms=110, rate=44100):
        n = int(rate * ms / 1000)
        t = np.arange(n) / rate
        env = np.minimum(1.0, np.minimum(t * 120, (t[-1] - t) * 60))  # attaque/chute douces
        x = (np.sin(2 * np.pi * freq * t) * env * 0.35 * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(x.tobytes())

    def beep(self):
        if not self.enabled:
            return
        if self.player and self.path:
            try:
                subprocess.Popen([self.player, "-q", self.path]
                                 if self.player == "aplay" else [self.player, self.path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                self.player = None
        sys.stdout.write("\a")
        sys.stdout.flush()


# ------------------------------------------------------------------ camera
def open_camera(index, width, height):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"impossible d'ouvrir la camera {index}")
    return cap


def display_name(label):
    """'type2_equerre' -> 'TYPE 2'. Un nom sans chiffre est repris tel quel."""
    m = re.match(r"type[_\- ]?(\d+)", label, re.I)
    return f"TYPE {m.group(1)}" if m else label.replace("_", " ").upper()


# -------------------------------------------------------------- fond adaptatif
class Background:
    def __init__(self):
        self.model = None
        self.acc = []
        self.empty_streak = 0
        self.auto = True
        self.calibrating = True

    def feed_calibration(self, frame):
        self.acc.append(frame.astype(np.float32))
        if len(self.acc) >= CALIB_FRAMES:
            self.model = np.mean(self.acc, axis=0).astype(np.uint8)
            self.acc = []
            self.calibrating = False
            return True
        return False

    def start_calibration(self):
        self.acc = []
        self.calibrating = True
        self.model = None

    def observe(self, frame, part_found):
        if self.model is None or not self.auto:
            return
        if part_found:
            self.empty_streak = 0
            return
        self.empty_streak += 1
        if self.empty_streak >= EMPTY_BEFORE_REFRESH:
            # slow fusion: the background tracks lighting drift without ever
            # absorbing a briefly placed part
            self.model = cv2.addWeighted(self.model, 1 - BG_ALPHA, frame, BG_ALPHA, 0)


# ------------------------------------------------------- verrouillage
class Decision:
    """Machine a trois etats : VIDE -> MESURE -> VERROUILLE.

    En VERROUILLE on ne segmente plus du tout : on se contente de comparer
    l'image a celle prise au moment du verrouillage. C'est une soustraction sur
    une vignette, quelques dixiemes de milliseconde, contre ~150 ms pour le
    cycle complet. Le resultat affiche est donc stable ET la boucle reste fluide.
    """

    VIDE, MESURE, VERROUILLE = "vide", "mesure", "verrouille"

    def __init__(self):
        self.state = self.VIDE
        self.label = None
        self.res = None
        self.cnt = None
        self.ref = None              # vignette de reference au verrouillage
        self.change_streak = 0
        self.locked_at = 0.0

    @staticmethod
    def _thumb(frame, zone):
        if zone is not None:
            x, y, w, h = [int(v) for v in zone]
            frame = frame[y:y + h, x:x + w]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (96, 72), interpolation=cv2.INTER_AREA).astype(np.int16)

    def lock(self, label, res, cnt, frame, zone):
        self.state = self.VERROUILLE
        self.label, self.res, self.cnt = label, res, cnt
        self.ref = self._thumb(frame, zone)
        self.change_streak = 0
        self.locked_at = time.time()

    def scene_changed(self, frame, zone):
        """Vrai si la scene a bouge depuis le verrouillage. Volontairement
        tolerant : il faut un changement net et SOUTENU, sinon un passage
        d'ombre ou le bruit du capteur deverrouillerait sans arret."""
        t = self._thumb(frame, zone)
        d = np.abs(t - self.ref)
        frac = float(np.count_nonzero(d > 22)) / d.size
        if frac > CHANGE_FRAC:
            self.change_streak += 1
        else:
            self.change_streak = max(0, self.change_streak - 1)
        return self.change_streak >= CHANGE_FRAMES

    def unlock(self):
        self.state = self.VIDE
        self.label = self.res = self.cnt = self.ref = None
        self.change_streak = 0


# ------------------------------------------------------------------ rendu
def render(frame, clf, res, cnt, stable, bg, fps, anim, show_details, stale, muted,
           zone=None, locked=False):
    vis = frame.copy()
    H, W = vis.shape[:2]

    color = hud.GREY
    if res is not None and res["accepted"]:
        color = hud.type_color(clf.classes.index(res["raw_label"]))

    if cnt is not None:
        hud.glow_contour(vis, cnt, color)
        hud.corner_brackets(vis, cv2.boundingRect(cnt), color)

    if bg.calibrating:
        badge_bottom = hud.type_badge(vis, "CALIBRATION", (86, 180, 250),
                                      len(bg.acc) / CALIB_FRAMES, anim=1.0,
                                      subtitle="laissez la scene vide")
    elif stable is None:
        badge_bottom = hud.type_badge(vis, "EN ATTENTE", hud.GREY, 0.0, anim=1.0,
                                      subtitle="posez une piece")
    elif stable == "UNKNOWN":
        badge_bottom = hud.type_badge(vis, "INDETERMINE", (90, 90, 225),
                                      res["confidence"] if res else 0.0, anim,
                                      subtitle="forme non reconnue")
    else:
        conf = res["confidence"] if res else 0.0
        c = hud.type_color(clf.classes.index(stable))
        # Intentionally sober display: the banner gives the TYPE in its
        # color and percentage, nothing else. The thin gauge under the title
        # is enough to see at a glance if the measurement is clear or close.
        badge_bottom = hud.type_badge(vis, display_name(stable), c, conf, anim,
                                      subtitle=f"confiance {conf:.0%}")

    if bg.model is None:
        bgtxt, bgcol = ("sans fond", (86, 180, 250))
    elif stale:
        bgtxt, bgcol = ("fond perime - b", (90, 90, 225))
    else:
        bgtxt, bgcol = (f"fond actif{' +auto' if bg.auto else ''}", (128, 214, 126))
    items = [(bgtxt, bgcol), (f"{fps:.0f} fps", hud.DIM)]
    if muted:
        items.append(("bip coupe", hud.DIM))
    hud.status_bar(vis, items)

    if show_details and res is not None:
        pw, ph = max(int(W * 0.23), 210), int(34 + 34 * len(res["scores"]))
        px, py = W - pw - int(0.02 * W), badge_bottom + int(0.02 * H)
        hud.panel(vis, px, py, pw, ph, alpha=0.78)
        hud.text(vis, "scores", (px + 16, py + 24), 0.44, hud.DIM, 1, font=hud.FS)
        y = py + 34
        for lab, sc in sorted(res["scores"].items(), key=lambda kv: -kv[1]):
            y += 34
            c = hud.type_color(clf.classes.index(lab))
            hud.rounded_rect(vis, px + 16, y - 13, max(4, int(sc * (pw - 32))), 16, 8, c, -1)
            hud.text(vis, f"{display_name(lab)}  {sc:.2f}", (px + 22, y), 0.42,
                     (20, 20, 20), 1, font=hud.FS)
    return vis


# ------------------------------------------------------------------ boucle
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.npz")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-bg", action="store_true")
    ap.add_argument("--no-beep", action="store_true")
    ap.add_argument("--details", action="store_true", help="show scores")
    ap.add_argument("--open-set", action="store_true",
                    help="allow UNKNOWN response (if parts outside "
                         "catalog can appear)")
    ap.add_argument("--zone", default="zone.json",
                    help="file where the work zone is saved")
    a = ap.parse_args()

    clf = PartClassifier.load(a.model)
    if a.open_set:
        clf.OPEN_SET = True
    area_range = clf.area_frac_range()
    print("types loaded:", ", ".join(f"{display_name(c)} x{len(clf.poses(c))} poses"
                                       for c in clf.classes))
    if area_range:
        print(f"expected part area: {area_range[0]*100:.1f}% a "
              f"{area_range[1]*100:.1f}% of the analyzed frame")
    print("mode:", "open world (UNKNOWN possible)" if clf.OPEN_SET
          else "closed world (always a type, confidence says if sure)")
    print("Leave the scene EMPTY for a few seconds: background calibration.")
    cap = open_camera(a.camera, a.width, a.height)
    beeper = Beeper(enabled=not a.no_beep)

    zone = None
    if os.path.exists(a.zone):
        try:
            zone = tuple(json.load(open(a.zone)))
            print(f"work zone loaded: {zone}")
        except Exception:
            zone = None

    bg = Background()
    if a.no_bg:
        bg.calibrating = False
    votes = deque(maxlen=VOTE_WINDOW)
    dec = Decision()
    stable, prev_stable, res, cnt, stale = None, None, None, None, False
    show_details = a.details
    fps, t_prev, t_change, i = 0.0, time.time(), 0.0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1

        if bg.calibrating:
            bg.feed_calibration(frame)
            res, cnt, stable = None, None, None
        elif dec.state == Decision.VERROUILLE:
            # nothing to recalculate: we only monitor if the scene has changed
            res, cnt, stable = dec.res, dec.cnt, dec.label
            if dec.scene_changed(frame, zone):
                print("scene modified -> new measurement")
                dec.unlock()
                votes.clear()
                stable = prev_stable = None
                res, cnt = None, None
        else:
            fa = (zone[2] * zone[3]) if zone is not None else frame.shape[0] * frame.shape[1]
            cands = S.segment_candidates(frame, background=bg.model, roi=zone,
                                         area_range=area_range)
            if not cands:
                votes.append(None)
                res, cnt = None, None
                bg.observe(frame, False)
            else:
                bg.observe(frame, True)
                # the model arbitrates between possible segmentations
                res, mask, cnt = clf.classify_candidates(cands, fa)
                votes.append(res["label"] if res else None)

            c = Counter(v for v in votes if v is not None)
            if c:
                top, n_top = c.most_common(1)[0]
                need = (VOTE_MIN_SURE if (res and res["confidence"] >= CONF_SURE)
                        else VOTE_MIN)
                if n_top >= need:
                    stable = top
            else:
                stable = None

            if stable != prev_stable:
                t_change = time.time()
                if stable not in (None, "UNKNOWN"):
                    beeper.beep()
                    if res:
                        print(f"  {display_name(stable)}   confiance {res['confidence']:.2f}"
                              f"   ({res.get('n_candidates', 0)} segmentations examined)")
                prev_stable = stable

            # stabilized and usable decision -> we freeze it
            if stable not in (None, "UNKNOWN") and res is not None and cnt is not None:
                dec.lock(stable, res, cnt, frame, zone)

            if bg.model is not None and i % 30 == 0:
                stale = S.background_is_stale(frame, bg.model)
                if stale and bg.auto and all(v is None for v in votes):
                    print("stale background + empty scene -> recalibration")
                    bg.start_calibration()

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        anim = float(np.clip((now - t_change) / ANIM_SECONDS, 0, 1))

        cv2.imshow("part identification",
                   render(frame, clf, res, cnt, stable, bg, fps, anim,
                          show_details, stale, not beeper.enabled, zone,
                          dec.state == Decision.VERROUILLE))

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('z'):
            # the zone eliminates bright distractors outside the station (floor, box,
            # workbench) BEFORE any analysis: this is the most profitable
            # setting when the scene is not entirely controlled
            r = cv2.selectROI("work zone - ENTER to validate", frame,
                              showCrosshair=False)
            cv2.destroyWindow("work zone - ENTER to validate")
            if r[2] > 20 and r[3] > 20:
                zone = tuple(int(v) for v in r)
                json.dump(zone, open(a.zone, "w"))
                print(f"work zone: {zone}  (leave a margin: a part"
                      f" touching the zone edge is ignored)")
                bg.start_calibration()
        elif k == ord('c'):
            zone = None
            if os.path.exists(a.zone):
                os.remove(a.zone)
            print("zone cleared")
        elif k == ord('d'):
            show_details = not show_details
        elif k == ord('m'):
            beeper.enabled = not beeper.enabled
        elif k == ord('b'):
            print("recalibration: leave scene empty")
            bg.start_calibration()
        elif k == ord('a'):
            bg.auto = not bg.auto
        elif k == ord('n'):
            bg.model = None
            bg.calibrating = False
        elif k == ord('r'):
            votes.clear()
            dec.unlock()
            stable = prev_stable = None
        elif k in (ord('+'), ord('=')):
            clf.MIN_SCORE = min(0.95, clf.MIN_SCORE + 0.02)
        elif k in (ord('-'), ord('_')):
            clf.MIN_SCORE = max(0.20, clf.MIN_SCORE - 0.02)
        elif k == ord('s'):
            os.makedirs("snaps", exist_ok=True)
            ts = time.strftime("%H%M%S")
            cv2.imwrite(f"snaps/{ts}.png", frame)
            if bg.model is not None:
                cv2.imwrite(f"snaps/{ts}_bg.png", bg.model)
            print(f"snap -> snaps/{ts}.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
