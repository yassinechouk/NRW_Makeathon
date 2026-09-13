"""
check_camera.py — find your Microsoft HD camera and check the framing.

    python check_camera.py            # scan indices 0..5 and report
    python check_camera.py --show 1   # live preview of camera 1, 'q' to quit

Use the preview to set up the scene before enrolling:
  * the part fully inside the frame, NOT touching the edges (parts touching a
    border are ignored on purpose — a partially visible part cannot be measured)
  * the part filling roughly 1/4 to 1/2 of the frame height
  * a plain, dark, matte background, no strong glare
  * camera perpendicular to the table (top view), same height for enrolment and
    for inference
"""
import argparse
import sys

import cv2


def backend():
    return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def v4l2_name(index):
    """Linux: the device label, so you can tell the built-in webcam from the
    external one without guessing."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name") as fh:
            return fh.read().strip()
    except OSError:
        return None


def scan(n=8):
    print(f"platform: {sys.platform}   backend: "
          f"{'DSHOW' if sys.platform.startswith('win') else 'default(V4L2/AVF)'}")
    # OpenCV prints its own probe failures on stderr; they are expected here
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass

    found = []
    for i in range(n):
        name = v4l2_name(i)
        # UVC devices expose an extra "metadata" node that is not an image source
        if name and "metadata" in name.lower():
            print(f"  index {i}: (metadata node - {name}) skipped")
            continue

        cap = cv2.VideoCapture(i, backend())
        if not cap.isOpened():
            cap.release()
            continue

        # a real capture node returns two consistent, non-uniform frames
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        good = (ok1 and ok2 and f1 is not None and f2 is not None
                and f1.shape == f2.shape and float(f1.std()) > 1.0)
        if not good:
            print(f"  index {i}: opens but delivers no usable image - skipped"
                  + (f"  ({name})" if name else ""))
            cap.release()
            continue

        h, w = f1.shape[:2]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.read()
        w2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        label = f"  camera {i}: OK  {w}x{h} -> {w2}x{h2}"
        if name:
            label += f"   [{name}]"
        print(label)
        found.append(i)
        cap.release()

    if not found:
        print("  no camera found.")
        print("  Linux: check `ls /dev/video*` and that your user is in the 'video' group.")
        print("  Windows: check Settings > Privacy > Camera, and close Teams/Zoom.")
    else:
        print(f"\n  usable indices: {found}")
        print("  Pick the EXTERNAL one (its name is the model, not 'Integrated Camera'),")
        print(f"  then:  python check_camera.py --show {found[-1]}")
    return found


def show(index):
    cap = cv2.VideoCapture(index, backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {index}")
    from part_classifier import segment
    while True:
        ok, f = cap.read()
        if not ok:
            break
        mask, cnt = segment(f)
        if cnt is not None:
            cv2.drawContours(f, [cnt], -1, (0, 255, 0), 3)
            a = cv2.contourArea(cnt)
            frac = a / (f.shape[0] * f.shape[1])
            msg = f"area {a:.0f} px ({frac*100:.1f}% of frame)"
            col = (0, 255, 0) if 0.02 < frac < 0.35 else (0, 180, 255)
            cv2.putText(f, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            if frac <= 0.02:
                cv2.putText(f, "move the camera CLOSER", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
        else:
            cv2.putText(f, "no part detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(f"camera {index} - q to quit", f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=None)
    a = ap.parse_args()
    if a.show is None:
        scan()
    else:
        show(a.show)
