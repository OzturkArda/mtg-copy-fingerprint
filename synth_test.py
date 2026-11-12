#!/usr/bin/env python3
"""
synth_test.py - ground-truth test for copy_fingerprint.py.

Renders one card "printing" (a synthetic design, or a real card scan with --scan),
cuts several physical copies from it with known centering offsets and known marks,
photographs them with random pose, perspective, lighting, shadow, lens distortion,
blur, noise and JPEG compression, then checks what copy_fingerprint.py recovers.

  python synth_test.py OUTPUT_DIR [trials]
  python synth_test.py OUTPUT_DIR 8 --scan bolt.png              # real card art
  python synth_test.py OUTPUT_DIR 8 --scan angel.png --bg 40,38,36  # white border, dark table
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy_fingerprint as cd  # noqa: E402

PPM = 30      # render resolution, px per mm
BLEED = 2.0   # printed area beyond the nominal cut, mm


def mm(v):
    return int(round(v * PPM))


def make_print(seed=7):
    """One printing: black border, frame, art, text. Canvas is the card plus bleed."""
    rng = np.random.default_rng(seed)
    wc, hc = mm(63 + 2 * BLEED), mm(88 + 2 * BLEED)
    img = np.full((hc, wc, 3), 16, np.uint8)

    def box(x0, y0, x1, y1):
        return slice(mm(y0 + BLEED), mm(y1 + BLEED)), slice(mm(x0 + BLEED), mm(x1 + BLEED))

    def tex(sl, base, amp, grain):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        lo = rng.normal(0, 1, (max(2, h // grain), max(2, w // grain), 3)).astype(np.float32)
        t = cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(np.array(base, np.float32) + amp * t, 0, 255)

    frame = box(3, 3, 60, 85)
    img[frame] = tex(frame, (60, 110, 70), 14, 12)
    art = box(5.5, 12, 57.5, 48)
    a = tex(art, (120, 120, 120), 45, 160) + tex(art, (0, 0, 0), 22, 36) + tex(art, (0, 0, 0), 10, 8)
    a = np.clip(a, 0, 255).astype(np.uint8)
    for _ in range(40):
        p = (int(rng.integers(0, a.shape[1])), int(rng.integers(0, a.shape[0])))
        cv2.circle(a, p, int(rng.integers(8, 90)), tuple(int(v) for v in rng.integers(30, 230, 3)), -1, cv2.LINE_AA)
    img[art] = cv2.GaussianBlur(a, (0, 0), 1.2)
    for b in (box(4.5, 4.5, 58.5, 10.5), box(4.5, 49.5, 58.5, 55), box(5.5, 56.5, 57.5, 81)):
        img[b] = tex(b, (190, 208, 204), 5, 10).astype(np.uint8)

    def text(s, x, y, scale, thick):
        cv2.putText(img, s, (mm(x + BLEED), mm(y + BLEED)), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick, cv2.LINE_AA)

    words = "when this creature enters target player draws two cards then discards a card unless you pay".split()
    text("Synthetic Test Golem", 6, 9, 3.0, 6)
    text("Artifact Creature - Golem", 6, 53.6, 2.3, 5)
    for i in range(7):
        text(" ".join(rng.choice(words, 6)), 6.5, 60.5 + 3.1 * i, 1.9, 4)
    return img


def print_from_scan(path):
    """A real card scan (e.g. a Scryfall PNG) as a printing: scaled to PPM, rounded corners and
    BLEED filled with the border colour."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgr = img[..., :3].astype(np.float32)
    alpha = img[..., 3:].astype(np.float32) / 255 if img.shape[2] == 4 else np.ones_like(bgr[..., :1])
    h, w = bgr.shape[:2]
    e = max(2, h // 150)                                       # sample the border just inside the edge
    ring = np.concatenate([bgr[e:2 * e, w // 4:3 * w // 4].reshape(-1, 3),
                           bgr[h - 2 * e:h - e, w // 4:3 * w // 4].reshape(-1, 3)])
    border = np.median(ring, axis=0)
    card = cv2.resize(bgr * alpha + border * (1 - alpha), (mm(63), mm(88)), interpolation=cv2.INTER_CUBIC)
    b = mm(BLEED)
    pr = cv2.copyMakeBorder(card, b, b, b, b, cv2.BORDER_CONSTANT, value=[float(v) for v in border])
    return np.clip(pr, 0, 255).astype(np.uint8)


def rand_marks(rng, n):
    """Wear on one physical copy, in card mm."""
    marks = []
    for _ in range(n):
        kind = rng.choice(["speck", "scratch", "edge"])
        if kind == "speck":
            marks.append(("speck", rng.uniform(4, 59), rng.uniform(4, 84), rng.uniform(0.15, 0.35)))
        elif kind == "scratch":
            x, y, ang, ln = rng.uniform(8, 55), rng.uniform(8, 80), rng.uniform(0, np.pi), rng.uniform(3, 9)
            marks.append(("scratch", x, y, x + ln * np.cos(ang), y + ln * np.sin(ang)))
        else:
            side, pos = int(rng.integers(0, 4)), rng.uniform(0.15, 0.85)
            marks.append(("edge", side, pos, rng.uniform(1.0, 1.6), rng.uniform(1.5, 3.5)))
    return marks


def make_copy(pr, ox=0.0, oy=0.0, rot=0.0, marks=()):
    """Cut one physical copy: print shifted (ox, oy) mm right/down and rotated relative to the cut."""
    wc, hc = mm(63), mm(88)
    th = np.radians(rot)
    rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.array([wc / 2, hc / 2])
    t = c + np.array([ox, oy]) * PPM - rm @ (c + BLEED * PPM)
    card = cv2.warpAffine(pr, np.column_stack([rm, t]), (wc, hc), flags=cv2.INTER_CUBIC, borderValue=(16, 16, 16))
    for m in marks:
        lay = card.copy()
        if m[0] == "speck":
            cv2.circle(lay, (mm(m[1]), mm(m[2])), max(mm(m[3]), 2), (225, 225, 225), -1, cv2.LINE_AA)
            card = cv2.addWeighted(lay, 0.9, card, 0.1, 0)
        elif m[0] == "scratch":
            cv2.line(lay, (mm(m[1]), mm(m[2])), (mm(m[3]), mm(m[4])), (215, 215, 215), 4, cv2.LINE_AA)
            card = cv2.addWeighted(lay, 0.55, card, 0.45, 0)
        else:
            _, side, pos, depth, ln = m
            if side in (0, 2):
                x, y = pos * 63, (0 if side == 0 else 88)
                ax = (mm(ln / 2), mm(depth))
            else:
                x, y = (0 if side == 3 else 63), pos * 88
                ax = (mm(depth), mm(ln / 2))
            cv2.ellipse(lay, (mm(x), mm(y)), ax, 0, 0, 360, (220, 220, 220), -1, cv2.LINE_AA)
            card = cv2.addWeighted(lay, 0.85, card, 0.15, 0)
    alpha = np.zeros((hc, wc), np.uint8)
    r = mm(3.0)
    cv2.rectangle(alpha, (r, 0), (wc - 1 - r, hc - 1), 255, -1)
    cv2.rectangle(alpha, (0, r), (wc - 1, hc - 1 - r), 255, -1)
    for cx, cy in ((r, r), (wc - 1 - r, r), (r, hc - 1 - r), (wc - 1 - r, hc - 1 - r)):
        cv2.circle(alpha, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return card, alpha


def shoot(card, alpha, rng, flip=False, out_wh=(3000, 4000), bg_bgr=(176, 196, 205)):
    """Photograph a copy: random pose and perspective, shadow, lighting, blur, noise, JPEG."""
    wo, ho = out_wh
    bg = np.full((ho, wo, 3), bg_bgr, np.float32)
    bg += cv2.resize(rng.normal(0, 5, (ho // 40, wo // 40)).astype(np.float32), (wo, ho), interpolation=cv2.INTER_CUBIC)[..., None]
    hc, wc = card.shape[:2]
    sc = rng.uniform(0.62, 0.72) * ho / hc
    ang = np.radians(rng.uniform(-6, 6) + (180 if flip else 0))
    rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = np.array([[0, 0], [wc, 0], [wc, hc], [0, hc]], np.float32)
    dst = (src - [wc / 2, hc / 2]) * sc @ rm.T + [wo / 2, ho / 2] + rng.uniform(-80, 80, 2)
    dst = (dst + rng.normal(0, 0.006 * hc * sc, (4, 2))).astype(np.float32)   # perspective
    hm = cv2.getPerspectiveTransform(src, dst)
    wcard = cv2.warpPerspective(card, hm, (wo, ho), flags=cv2.INTER_LINEAR).astype(np.float32)
    wa = cv2.warpPerspective(alpha, hm, (wo, ho), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    sx, sy = rng.uniform(-6, 6, 2)
    shadow = cv2.GaussianBlur(cv2.warpAffine(wa, np.float32([[1, 0, sx], [0, 1, sy]]), (wo, ho)), (0, 0), 4) * 0.35
    img = bg * (1 - shadow[..., None])
    img = img * (1 - wa[..., None]) + wcard * wa[..., None]
    yy, xx = np.mgrid[0:ho, 0:wo].astype(np.float32)
    gx, gy = rng.uniform(-0.12, 0.12, 2)
    light = (1 + gx * (xx / wo - 0.5) + gy * (yy / ho - 0.5)) * rng.uniform(0.9, 1.05)
    img = img * light[..., None] * rng.uniform(0.97, 1.03, 3)
    # Residual lens distortion (0.8% barrel at the frame corner), so the card's position in the
    # frame changes its apparent shape slightly from shot to shot.
    r2 = ((xx - wo / 2) ** 2 + (yy - ho / 2) ** 2) / ((wo / 2) ** 2 + (ho / 2) ** 2)
    f = 1 + 0.008 * r2
    img = cv2.remap(img, (wo / 2 + (xx - wo / 2) * f).astype(np.float32),
                    (ho / 2 + (yy - ho / 2) * f).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.7, 1.4))
    img = np.clip(img + 2.5 * rng.standard_normal(img.shape, dtype=np.float32), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def run(outdir, trials=8, seed=1, scan=None, bg_bgr=(176, 196, 205)):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pr = print_from_scan(scan) if scan else make_print()
    n = [0]

    def photo(copy, flip=False):
        n[0] += 1
        p = os.path.join(outdir, f"shot_{n[0]:03d}.jpg")
        cv2.imwrite(p, shoot(*copy, rng, flip=flip, bg_bgr=bg_bgr))
        return p

    def base():
        return rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.15, 0.15)

    print("== 1. Same copy, two separate photos (expect: no difference) ==")
    floor, false_marks = [], 0
    for i in range(trials):
        ox, oy, rot = base()
        copy = make_copy(pr, ox, oy, rot, rand_marks(rng, 4 if i % 2 else 0))
        r = cd.compare(photo(copy), photo(copy, flip=(i % 4 == 3)))
        floor.append(r["centering"]["max_displacement_mm"])
        false_marks += len(r["marks"])
        print(f"  disp {floor[-1]:.3f} mm  scale {r['centering']['scale_mismatch_pct']:+.2f}%  marks {len(r['marks'])}  -> {r['verdict']}"
              + "".join(f"\n    warning: {w}" for w in r["warnings"]))
    print(f"  noise floor: median {np.median(floor):.3f} mm, max {np.max(floor):.3f} mm; false marks: {false_marks}")

    print("== 2. Two mint copies, centering differs by a known amount ==")
    for delta in (0.10, 0.15, 0.20, 0.30, 0.50, 1.00):
        errs, hits = [], 0
        for _ in range(trials):
            ox, oy, rot = base()
            ang = rng.uniform(0, 2 * np.pi)
            dx, dy = delta * np.cos(ang), delta * np.sin(ang)
            r = cd.compare(photo(make_copy(pr, ox, oy, rot)), photo(make_copy(pr, ox + dx, oy + dy, rot)))
            c = r["centering"]
            errs.append(np.hypot(c["dx_mm"] - dx, c["dy_mm"] - dy))
            hits += r["centering_differs"]
        print(f"  true {delta:.2f} mm: detected {hits}/{trials}, vector error median {np.median(errs):.3f} max {np.max(errs):.3f} mm")

    print("== 3. Same centering, tilt differs by 0.3 deg ==")
    errs, hits = [], 0
    for _ in range(trials):
        ox, oy, rot = base()
        r = cd.compare(photo(make_copy(pr, ox, oy, rot)), photo(make_copy(pr, ox, oy, rot + 0.3)))
        errs.append(abs(r["centering"]["rotation_deg"] - 0.3))
        hits += r["centering_differs"]
    print(f"  detected {hits}/{trials}, rotation error median {np.median(errs):.3f} max {np.max(errs):.3f} deg")

    print("== 4. Identical centering, one copy has wear (expect: marks found, located) ==")
    found = total = extra = 0
    sample = None
    for i in range(trials):
        ox, oy, rot = base()
        marks = rand_marks(rng, 4)
        pa, pb = photo(make_copy(pr, ox, oy, rot, marks)), photo(make_copy(pr, ox, oy, rot))
        r = cd.compare(pa, pb)
        pts = np.array([[m["x_mm"], m["y_mm"]] for m in r["marks"]]).reshape(-1, 2)
        used = set()
        for m in marks:
            if m[0] == "speck":
                seg = [(m[1], m[2])]
            elif m[0] == "scratch":
                seg = [(m[1] + (m[3] - m[1]) * t, m[2] + (m[4] - m[2]) * t) for t in np.linspace(0, 1, 9)]
            else:
                seg = [((m[2] * 63, 0), (63, m[2] * 88), (m[2] * 63, 88), (0, m[2] * 88))[m[1]]]
            total += 1
            near = [j for j in range(len(pts)) if min(np.hypot(pts[j][0] - x, pts[j][1] - y) for x, y in seg) < 2.5]
            found += bool(near)
            used.update(near)
        extra += len(pts) - len(used)
        for j in set(range(len(pts))) - used:
            print(f"    unexplained mark at {pts[j]} {r['marks'][j]}  pair: {pa} {pb}\n    planted: {marks}")
        if sample is None and len(r["marks"]) >= 3:
            sample = (pa, pb)
        print(f"  planted {len(marks)}, reported {len(r['marks'])}, disp {r['centering']['max_displacement_mm']:.3f} mm -> {r['verdict']}")
    print(f"  planted marks found: {found}/{total}; reported marks with no planted cause: {extra}")
    if sample:
        print(f"  sample pair: {sample[0]} {sample[1]}")


    print("== 5. Failure paths ==")
    other = make_print(seed=99)   # any design other than pr
    for label, fn in (("different card design", lambda: cd.compare(photo(make_copy(pr)), photo(make_copy(other)))),
                      ("photo with no card", lambda: cd.compare(photo(make_copy(pr)), _blank(outdir, bg_bgr)))):
        try:
            fn()
            print(f"  {label}: NOT REJECTED")
        except cd.CardError as e:
            print(f"  {label}: rejected ({e})")


def _blank(outdir, bg_bgr):
    p = os.path.join(outdir, "blank.jpg")
    cv2.imwrite(p, np.full((4000, 3000, 3), bg_bgr, np.uint8))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ground-truth test for copy_fingerprint.py on simulated photos.")
    ap.add_argument("outdir", help="where the simulated photos are written")
    ap.add_argument("trials", nargs="?", type=int, default=8, help="pairs per test row (default 8)")
    ap.add_argument("--scan", help="use this card scan (PNG/JPEG, e.g. from Scryfall) as the printing")
    ap.add_argument("--bg", default="176,196,205", help="background colour as B,G,R (default 176,196,205, light beige)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    run(args.outdir, args.trials, args.seed, args.scan, tuple(int(v) for v in args.bg.split(",")))
