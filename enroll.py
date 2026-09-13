"""
enroll.py — build the reference model (one image per type).

Two ways to use it:

  1) from image files (what you already have):
        python enroll.py --images refs/ --out model.npz
     every file in refs/ becomes a class, named after the file:
        refs/type1.jpg  ->  "type1"
        refs/type2.jpg  ->  "type2"
        refs/type3.jpg  ->  "type3"

  2) live from the camera (recommended - same optics as inference):
        python enroll.py --camera 0 --live --out model.npz
     put one part in the frame, press the number key of its type (1..9),
     press 's' to save the model, 'q' to quit.

A control image is written next to the model so you can check the silhouette
that was actually extracted - always look at it before trusting the model.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

import segmentation as S
from part_classifier import PartClassifier, segment, signature_from_mask, CANVAS


def open_camera(index, width=1280, height=720):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # fixed exposure / white balance gives far more repeatable colours
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index}")
    return cap


def contact_sheet(clf, path):
    """One strip showing every enrolled silhouette — your sanity check."""
    tiles = []
    for label, ref in zip(clf.labels, clf.refs):
        t = cv2.cvtColor(ref.canvas, cv2.COLOR_GRAY2BGR)
        t = cv2.resize(t, (CANVAS * 2, CANVAS * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        tiles.append(t)
    if tiles:
        cv2.imwrite(path, np.hstack(tiles))


def from_images(folder, out, background=None):
    clf = PartClassifier()
    files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                   if os.path.splitext(f)[1].lower() in
                   (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    if not files:
        raise SystemExit(f"no images in {folder}")
    bg = cv2.imread(background) if background else None
    files = [f for f in files if not os.path.splitext(os.path.basename(f))[0].endswith("_bg")]
    for f in files:
        label = os.path.splitext(os.path.basename(f))[0]
        # "type2_equerre@plat.png" = une pose SUPPLEMENTAIRE du type2_equerre.
        # Une piece 3D n'a pas la meme silhouette selon la face sur laquelle
        # elle repose : chaque pose stable merite sa propre reference.
        pose = None
        if "@" in label:
            label, pose = label.split("@", 1)
        img = cv2.imread(f)
        if img is None:
            print(f"  !! illisible : {f}")
            continue
        # un fond propre a ce type prime sur le fond global : refs/type1_bg.png
        stem = os.path.splitext(os.path.basename(f))[0]
        own = os.path.join(os.path.dirname(f), stem + "_bg" + os.path.splitext(f)[1])
        b = cv2.imread(own) if os.path.exists(own) else bg
        if b is not None:
            print(f"    (fond: {'propre' if b is not bg else 'global'})")
        sig = clf.add_reference(label, img, background=b)
        tag = f"{label} (pose {pose})" if pose else label
        print(f"  + {tag:24s} aire={sig.area_px:8.0f} px   "
              f"elong={sig.geom[0]:.2f} solidite={sig.geom[2]:.2f} circ={sig.geom[3]:.2f}")
    clf.save(out)
    contact_sheet(clf, os.path.splitext(out)[0] + "_refs.png")
    print(f"\nmodel -> {out}\ncheck the silhouettes -> {os.path.splitext(out)[0]}_refs.png")
    separation_report(clf)


def separation_report(clf):
    """Deux types sont-ils separables ? On compare chaque paire de TYPES en
    prenant le PIRE cas sur leurs poses : c'est la pose la plus ressemblante qui
    fera l'erreur, pas la moyenne."""
    from part_classifier import align_iou, chamfer_similarity
    cls = clf.classes
    if len(cls) < 2:
        return
    print("\nsimilitude entre types enroles (plus bas = mieux separes) :")
    for a in range(len(cls)):
        for b in range(a + 1, len(cls)):
            worst_iou, worst_ch = 0.0, 0.0
            for i in clf.poses(cls[a]):
                for j in clf.poses(cls[b]):
                    iou, _, _, aligned = align_iou(clf.refs[i], clf.refs[j],
                                                   clf.allow_mirror)
                    ch = (chamfer_similarity(clf.refs[i].canvas > 0, aligned)
                          if aligned is not None else 0.0)
                    if iou + ch > worst_iou + worst_ch:
                        worst_iou, worst_ch = iou, ch
            # une paire n'est vraiment confondable que si les DEUX mesures la
            # rapprochent : l'aire ET le trace du contour
            flag = "  <-- TROP SEMBLABLES" if (worst_iou > 0.80 and worst_ch > 0.70) \
                else ("  <-- proches" if worst_iou > 0.75 else "")
            print(f"  {cls[a][:16]:16s} vs {cls[b][:16]:16s}  "
                  f"aire={worst_iou:.3f}  contour={worst_ch:.3f}{flag}")
    n = {c: len(clf.poses(c)) for c in cls}
    print("  poses par type : " + ", ".join(f"{k}={v}" for k, v in n.items()))
    if any(v == 1 for v in n.values()):
        print("  ! un type avec UNE seule pose ne sera reconnu que pose comme sa")
        print("    reference. Si la piece tient sur plusieurs faces, enrole chaque face.")


def live(camera, out, width, height):
    clf = PartClassifier()
    cap = open_camera(camera, width, height)
    bg = None
    print("ETAPE 1 : scene VIDE, appuyez sur b pour calibrer le fond.")
    print("ETAPE 2 : posez une piece, appuyez sur son chiffre (1..9).")
    print("          RETOURNEZ-LA sur une autre face stable et rappuyez sur le")
    print("          MEME chiffre : c'est une pose de plus pour ce type.")
    print("ETAPE 3 : s pour sauvegarder.   d = annuler le dernier   q = quitter")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, cnt = S.segment(frame, background=bg)
        vis = frame.copy()
        if cnt is not None:
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(vis, f"{cv2.contourArea(cnt):.0f} px", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "aucune piece detectee", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        rec = ", ".join(f"{c}x{len(clf.poses(c))}" for c in clf.classes) or "-"
        cv2.putText(vis, "enroles: " + rec, (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis, "fond: " + ("calibre" if bg is not None else "AUCUN - appuyez sur b"),
                    (20, height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("enroll", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('b'):
            # moyenne de quelques images : moins de bruit capteur dans le fond
            acc = [frame.astype(np.float32)]
            for _ in range(11):
                o, f2 = cap.read()
                if o:
                    acc.append(f2.astype(np.float32))
            bg = np.mean(acc, axis=0).astype(np.uint8)
            print("fond calibre (la scene doit etre vide !)")
        if k == ord('d') and clf.labels:
            print("annule :", clf.labels.pop())
            clf.refs.pop()
            clf._fit_scales()
        if ord('1') <= k <= ord('9'):
            if cnt is None:
                print("!! rien a enroler")
                continue
            label = f"type{chr(k)}"
            sig = signature_from_mask(mask, cnt, frame.shape[0] * frame.shape[1])
            # AJOUT, jamais remplacement : reappuyer sur la meme touche avec la
            # piece posee autrement enregistre une pose de plus pour ce type.
            clf.labels.append(label)
            clf.refs.append(sig)
            clf._fit_scales()
            print(f"enrole {label} — pose {len(clf.poses(label))}  aire={sig.area_px:.0f}")
        if k == ord('s'):
            if not clf.labels:
                print("!! nothing enrolled")
                continue
            clf.save(out)
            contact_sheet(clf, os.path.splitext(out)[0] + "_refs.png")
            print(f"sauvegarde -> {out}")
            separation_report(clf)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="folder with one reference image per type")
    ap.add_argument("--background", help="optional photo of the empty scene")
    ap.add_argument("--live", action="store_true", help="enroll from the camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="model.npz")
    a = ap.parse_args()
    if a.live:
        live(a.camera, a.out, a.width, a.height)
    elif a.images:
        from_images(a.images, a.out, a.background)
    else:
        ap.error("give --images FOLDER or --live")
