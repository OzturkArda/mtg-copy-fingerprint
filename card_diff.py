#!/usr/bin/env python3
"""
card_diff.py - tell two physical copies of the same card printing apart from photos.

Copies of one printing carry no unique ID, so this compares the two things that
do differ between physical copies:

  1. Cut registration - where the printed image sits relative to the cut edge
     (centering and tilt). Set at the factory, stays with the copy.
  2. Surface marks    - wear, scratches, print specks, whitening.

Method
  - Find each card's cut outline and rectify it to a 63 x 88 mm canvas.
  - Register print-to-print with feature matching. Both images are already
    aligned on the cut, so the residual transform IS the centering difference.
  - Align on the print, cancel lighting differences, and map what is left.

Usage
  python card_diff.py a.jpg b.jpg --out report.png
  python card_diff.py a.jpg b.jpg --json

Exit code: 0 = no difference detected, 1 = difference detected, 2 = error.

Photos: one unsleeved card per photo, shot from straight above on a plain
background that contrasts with the card border, no glare, card filling most of
the frame. Fronts and backs both work; run the script once per side.

Calibrate first: run it on two photos of the SAME card. The "print displacement"
it reports is your setup's noise floor, and --tol-mm must sit above it.
"""
import argparse
import json
import sys

import cv2
import numpy as np

CARD_W, CARD_H, CORNER_R = 63.0, 88.0, 3.0  # mm
PX = 20                                      # rectified pixels per mm
W, H = int(CARD_W * PX), int(CARD_H * PX)


class CardError(Exception):
    pass


# --------------------------------------------------------------------------
# 1. Find the cut outline and rectify
# --------------------------------------------------------------------------

def _bg_distance(img):
    """Per-pixel Lab distance from the background colour (taken from the photo's outer frame)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.5), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(4, int(0.02 * min(h, w)))
    frame = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    return np.linalg.norm(lab - np.median(frame, axis=0), axis=2).astype(np.float32)


def _order(q):
    """Corners clockwise from top-left, rolled so the card is portrait."""
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    q = np.roll(q, -int(np.argmin(q.sum(1))), axis=0)
    if np.linalg.norm(q[1] - q[0]) > np.linalg.norm(q[3] - q[0]):
        q = np.roll(q, -1, axis=0)
    return q


def _rough_quad(dist):
    h, w = dist.shape
    s = 1000.0 / max(h, w)
    small = cv2.resize(dist, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    d8 = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise CardError("no card found: nothing stands out from the background")
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * mask.size:
        raise CardError("no card found: the card must contrast with the background and fill more of the frame")
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        quad = cv2.approxPolyDP(hull, eps * peri, True)
        if len(quad) == 4:
            return quad.reshape(4, 2).astype(np.float64) / s
    raise CardError("card outline is not a quadrilateral: is the whole card in frame?")


def _edge_points(dist, p0, p1, centre, win, n=160):
    """Sub-pixel edge points along side p0->p1: the sharpest card->background step along the normal.
    Skips the ends of the side, where the corners are rounded."""
    d = p1 - p0
    length = np.linalg.norm(d)
    d = d / length
    nrm = np.array([-d[1], d[0]])
    if np.dot(nrm, (p0 + p1) / 2 - centre) < 0:
        nrm = -nrm                                            # point outward
    base = p0 + np.linspace(0.15, 0.85, n)[:, None] * length * d
    t = np.arange(-win, win + 0.5, 0.5)
    xs = (base[:, 0:1] + t[None] * nrm[0]).astype(np.float32)
    ys = (base[:, 1:2] + t[None] * nrm[1]).astype(np.float32)
    prof = cv2.remap(dist, xs, ys, cv2.INTER_LINEAR)
    g = -np.gradient(prof, axis=1)                            # drop in bg-distance going outward
    k = np.argmax(g[:, 1:-1], axis=1) + 1
    r = np.arange(n)
    y0, y1, y2 = g[r, k - 1], g[r, k], g[r, k + 1]
    den = y0 - 2 * y1 + y2
    sub = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    tt = t[k] + 0.5 * np.clip(sub, -1, 1)
    return base + tt[:, None] * nrm


def _fit_line(pts):
    m = pts.mean(0)
    vx, vy, x0, y0 = cv2.fitLine((pts - m).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return m + np.array([x0, y0], float), np.array([vx, vy], float)


def _intersect(l1, l2):
    (p, u), (q, v) = l1, l2
    a = np.linalg.solve(np.column_stack([u, -v]), q - p)
    return p + a[0] * u


def find_card(img):
    """Four corners of the cut outline (where the straight edges intersect), sub-pixel."""
    dist = _bg_distance(img)
    quad = _order(_rough_quad(dist))
    short = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
    for win in (short * 0.04, short * 0.012):                 # coarse pass, then fine
        centre = quad.mean(0)
        lines = [_fit_line(_edge_points(dist, quad[i], quad[(i + 1) % 4], centre, win)) for i in range(4)]
        quad = np.array([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    return quad


def rectify(img):
    """Warp the card to a W x H canvas aligned on its cut edges. Returns (image, photo px/mm)."""
    quad = find_card(img)
    long_px = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    short_px = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    if abs(long_px / short_px / (CARD_H / CARD_W) - 1) > 0.08:
        raise CardError("outline does not have a 63x88 mm card's proportions: shoot from straight above, whole card in frame")
    ppm = long_px / CARD_H
    if ppm > PX:                                              # anti-alias before downsampling
        img = cv2.GaussianBlur(img, (0, 0), 0.5 * np.sqrt((ppm / PX) ** 2 - 1))
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    hm = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, hm, (W, H), flags=cv2.INTER_LINEAR), ppm


# --------------------------------------------------------------------------
# 2. Print-to-print registration -> centering difference
# --------------------------------------------------------------------------

def card_mask(inset_mm):
    """Rounded-rectangle card shape, inset from the cut edge."""
    m = np.zeros((H, W), np.uint8)
    i = int(round(inset_mm * PX))
    r = max(int(round((CORNER_R - inset_mm) * PX)), 0)
    x0, y0, x1, y1 = i, i, W - 1 - i, H - 1 - i
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def register_print(a, b):
    """Similarity transform taking B's print onto A's print. Returns (2x3 matrix, inliers, rms px)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ga = clahe.apply(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
    gb = clahe.apply(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    mask = card_mask(2.0)                                     # keep features off the cut edge
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(ga, mask)
    kb, db = sift.detectAndCompute(gb, mask)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        raise CardError("too little printed detail to register (blank or badly blurred photo?)")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        raise CardError("the two photos do not show the same card design")
    pb = np.float32([kb[m.queryIdx].pt for m in good])
    pa = np.float32([ka[m.trainIdx].pt for m in good])
    mat, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                           maxIters=5000, confidence=0.999, refineIters=20)
    if mat is None or int(inl.sum()) < 30:
        raise CardError("the two photos do not show the same card design")
    keep = inl.ravel() > 0
    res = pb[keep] @ mat[:, :2].T + mat[:, 2] - pa[keep]
    return mat, int(keep.sum()), float(np.sqrt((res ** 2).sum(1).mean()))


def centering_difference(mat):
    """How far B's print sits from A's print, in cut-edge coordinates (mm, +x right, +y down).
    Scale is a rectification artefact (both cards are 63x88), so it is reported but not counted."""
    inv = cv2.invertAffineTransform(mat)
    scale = float(np.hypot(inv[0, 0], inv[1, 0]))
    rot = float(np.arctan2(inv[1, 0], inv[0, 0]))
    c = np.array([W / 2, H / 2])
    shift = inv[:, :2] @ c + inv[:, 2] - c
    rm = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    disp = (corners - c) @ rm.T + c + shift - corners         # rigid part only
    return {
        "dx_mm": float(shift[0] / PX),
        "dy_mm": float(shift[1] / PX),
        "rotation_deg": float(np.degrees(rot)),
        "max_displacement_mm": float(max(np.linalg.norm(disp, axis=1).max(), np.linalg.norm(shift)) / PX),
        "scale_mismatch_pct": (scale - 1) * 100,
    }


# --------------------------------------------------------------------------
# 3. Surface marks
# --------------------------------------------------------------------------

def _glare(img):
    """Large blown-out areas. Small saturated spots are kept: they can be real whitening."""
    sat = (img.max(axis=2) >= 250).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    big = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= PX * PX])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PX + 1, PX + 1))
    return cv2.dilate(big.astype(np.uint8) * 255, k)


def _match_sharpness(ga, gb, valid):
    """Blur the sharper image down to the softer one, so focus differences are not read as marks."""
    def sharp(g):
        return float(cv2.Laplacian(g, cv2.CV_32F)[valid > 0].var())

    def blur_to(g, target):
        out = g
        for s in np.arange(0.3, 3.01, 0.15):
            out = cv2.GaussianBlur(g, (0, 0), float(s))
            if sharp(out) <= target:
                break
        return out

    sa, sb = sharp(ga), sharp(gb)
    return (ga, blur_to(gb, sa)) if sb > sa else (blur_to(ga, sb), gb)


def find_marks(a, b, valid, min_area_mm2, k_sigma):
    """Localised differences between two print-aligned images. Returns (marks, difference map)."""
    ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    ga, gb = _match_sharpness(ga, gb, valid)

    # Local gain/offset: make B's brightness and contrast match A's in every ~3 mm neighbourhood.
    # (Computed at 1/4 scale: the statistics are smooth, so nothing is lost and it is ~10x faster.)
    f, sig = 4, 3.0 * PX / 4

    def smooth(x):
        lo = cv2.resize(x, (W // f, H // f), interpolation=cv2.INTER_AREA)
        return cv2.resize(cv2.GaussianBlur(lo, (0, 0), sig), (W, H), interpolation=cv2.INTER_LINEAR)

    def match_b(v):
        wsum = smooth(v) + 1e-6

        def stats(g):
            mu = smooth(g * v) / wsum
            sd = np.sqrt(np.maximum(smooth((g - mu) ** 2 * v) / wsum, 0))
            return mu, sd

        (mua, sda), (mub, sdb) = stats(ga), stats(gb)
        return (gb - mub) * np.clip((sda + 2) / (sdb + 2), 0.5, 2.0) + mua

    # Two passes: a large mark would skew its own neighbourhood's statistics and cast false
    # marks around itself, so pass 2 leaves out whatever pass 1 found to differ.
    v = (valid > 0).astype(np.float32)
    resid = np.abs(ga - match_b(v))
    rough = resid > max(k_sigma * 1.4826 * float(np.median(resid[valid > 0])), 10.0)
    v[cv2.dilate(rough.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0] = 0
    gb = match_b(v)

    # A pixel only counts if it falls outside the other image's local range, which
    # forgives a pixel of misregistration along printed edges.
    k = np.ones((5, 5), np.uint8)

    def outside(x, y):
        return np.maximum(np.maximum(x - cv2.dilate(y, k), cv2.erode(y, k) - x), 0)

    d = np.maximum(outside(ga, gb), outside(gb, ga))
    d[valid == 0] = 0
    noise = 1.4826 * float(np.median(np.abs(ga - gb)[valid > 0]))
    thr = max(k_sigma * noise, 10.0)

    hot = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(hot, 8)
    marks = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA] / PX ** 2
        if area < min_area_mm2:
            continue
        sel = lab == i
        marks.append({
            "x_mm": round(float(cen[i][0] / PX), 1),
            "y_mm": round(float(cen[i][1] / PX), 1),
            "length_mm": round(float(max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) / PX), 2),
            "area_mm2": round(float(area), 3),
            "strength": round(float(d[sel].max()), 1),
            "brighter_in": "A" if float((ga - gb)[sel].mean()) > 0 else "B",
        })
    marks.sort(key=lambda m: -m["area_mm2"] * m["strength"])
    return marks, d / thr


# --------------------------------------------------------------------------
# 4. Compare, report
# --------------------------------------------------------------------------

def compare(path_a, path_b, tol_mm=0.15, min_mark_mm2=0.03, k_sigma=6.0):
    imgs = []
    for p in (path_a, path_b):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise CardError(f"cannot read {p} (convert HEIC to JPEG or PNG first)")
        imgs.append(img)
    (a, ppm_a), (b, ppm_b) = rectify(imgs[0]), rectify(imgs[1])

    mat, inliers, rms = register_print(a, b)
    if mat[0, 0] < 0:                                         # B was shot upside down
        # Exact half-turn about the card centre: x -> W - x, y -> H - y. (cv2.rotate maps
        # x -> W - 1 - x, which would shift the cut frame by one pixel = 0.05 mm per axis.)
        b = cv2.warpAffine(b, np.float32([[-1, 0, W], [0, -1, H]]), (W, H),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        mat, inliers, rms = register_print(a, b)
    cen = centering_difference(mat)

    b_al = cv2.warpAffine(b, mat, (W, H), flags=cv2.INTER_LINEAR)
    shape = card_mask(0.5)
    valid = shape & cv2.warpAffine(shape, mat, (W, H), flags=cv2.INTER_NEAREST)
    glare = _glare(a) | cv2.warpAffine(_glare(b), mat, (W, H), flags=cv2.INTER_NEAREST)
    valid = valid & ~glare
    marks, dmap = find_marks(a, b_al, valid, min_mark_mm2, k_sigma)
    differing = float((dmap > 1)[valid > 0].mean())
    if differing > 0.03:                                      # two copies of one printing never differ this much
        raise CardError(f"the photos differ over {differing:.0%} of the card: different card designs, "
                        "or one photo is badly blurred or glared")

    centering_differs = cen["max_displacement_mm"] > tol_mm
    reasons = []
    if centering_differs:
        reasons.append("centering")
    if marks:
        reasons.append("surface marks")

    warnings = []
    if min(ppm_a, ppm_b) < PX:
        warnings.append(f"low resolution ({min(ppm_a, ppm_b):.0f} px/mm, want {PX}+): get closer to the card")
    if abs(cen["scale_mismatch_pct"]) > 0.5:
        warnings.append("cut edges were located inconsistently (shadow, curl or low contrast): centering result is unreliable")
    glare_pct = 100.0 * float((glare > 0).sum()) / float((shape > 0).sum())
    if glare_pct > 2:
        warnings.append(f"{glare_pct:.0f}% of the card is hidden by glare and was not compared")

    return {
        "verdict": "DIFFERENCE DETECTED" if reasons else "NO DIFFERENCE DETECTED",
        "reasons": reasons,
        "centering": {k: round(v, 3) for k, v in cen.items()},
        "centering_differs": centering_differs,
        "tolerance_mm": tol_mm,
        "marks": marks,
        "registration": {"inliers": inliers, "rms_px": round(rms, 2)},
        "photo_px_per_mm": [round(ppm_a, 1), round(ppm_b, 1)],
        "warnings": warnings,
        "_images": (a, b_al, dmap),
    }


def text_report(r):
    c = r["centering"]
    out = [r["verdict"] + (": " + " + ".join(r["reasons"]) if r["reasons"] else
                           " (same copy, or two copies these photos cannot separate)"),
           "",
           f"Centering   print displacement {c['max_displacement_mm']:.2f} mm "
           f"(tolerance {r['tolerance_mm']:.2f}) -> {'differs' if r['centering_differs'] else 'matches'}",
           f"            B's print vs A's: {c['dx_mm']:+.2f} mm right, {c['dy_mm']:+.2f} mm down, "
           f"{c['rotation_deg']:+.2f} deg",
           f"Marks       {len(r['marks'])} found"]
    for i, m in enumerate(r["marks"][:10], 1):
        out.append(f"            #{i} at ({m['x_mm']:.1f}, {m['y_mm']:.1f}) mm from top-left, "
                   f"{m['length_mm']:.1f} mm long, brighter in {m['brighter_in']}")
    if len(r["marks"]) > 10:
        out.append(f"            ... and {len(r['marks']) - 10} more")
    out.append(f"Quality     {r['registration']['inliers']} matched print features, "
               f"photos at {r['photo_px_per_mm'][0]:.0f} and {r['photo_px_per_mm'][1]:.0f} px/mm")
    out += [f"WARNING     {w}" for w in r["warnings"]]
    return "\n".join(out)


def save_report(r, path):
    """Three panels: A, B aligned to A, and A with the difference map and numbered marks."""
    a, b_al, dmap = r["_images"]
    heat = cv2.applyColorMap((np.clip(dmap / 3, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(dmap, 0, 1)[..., None]
    over = (cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.6 * (1 - alpha)
            + heat * alpha).astype(np.uint8)
    for i, m in enumerate(r["marks"][:30], 1):
        p = (int(m["x_mm"] * PX), int(m["y_mm"] * PX))
        rad = int(max(m["length_mm"] * PX * 0.8, 24))
        cv2.circle(over, p, rad, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(over, str(i), (p[0] + rad + 4, p[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 255), 3, cv2.LINE_AA)
    gap, head = 24, 150
    canvas = np.full((H + head, 3 * W + 4 * gap, 3), 24, np.uint8)
    for i, (im, label) in enumerate(((a, "A"), (b_al, "B (aligned to A's print)"), (over, "Differences"))):
        x = gap + i * (W + gap)
        canvas[head:, x:x + W] = im
        cv2.putText(canvas, label, (x, head - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
    c = r["centering"]
    title = r["verdict"] + (": " + " + ".join(r["reasons"]) if r["reasons"] else "")
    cv2.putText(canvas, title, (gap, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                (80, 200, 255) if r["reasons"] else (140, 220, 140), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"print displacement {c['max_displacement_mm']:.2f} mm (tol {r['tolerance_mm']:.2f}), "
                        f"{len(r['marks'])} marks", (gap + 2 * (W + gap) - 60, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.imwrite(path, cv2.resize(canvas, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser(description="Tell two physical copies of the same card printing apart from photos.")
    ap.add_argument("photo_a")
    ap.add_argument("photo_b")
    ap.add_argument("--out", help="write a visual diff image to this path (png/jpg)")
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    ap.add_argument("--tol-mm", type=float, default=0.15,
                    help="print displacement above this counts as a centering difference (default 0.15)")
    ap.add_argument("--min-mark-mm2", type=float, default=0.03, help="ignore marks smaller than this area")
    ap.add_argument("--mark-sigma", type=float, default=6.0, help="mark threshold in noise sigmas")
    args = ap.parse_args()
    try:
        r = compare(args.photo_a, args.photo_b, args.tol_mm, args.min_mark_mm2, args.mark_sigma)
    except CardError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.out:
        save_report(r, args.out)
    r.pop("_images")
    print(json.dumps(r, indent=2) if args.json else text_report(r))
    return 1 if r["reasons"] else 0


if __name__ == "__main__":
    sys.exit(main())
