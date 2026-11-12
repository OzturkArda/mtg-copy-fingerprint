import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import copy_fingerprint as cf
import synth_test as st

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# load_image
# --------------------------------------------------------------------------

def test_load_image_reads_jpeg(tmp_path):
    p = tmp_path / "card.jpg"
    cv2.imwrite(str(p), np.full((40, 30, 3), (10, 120, 230), np.uint8))
    img = cf.load_image(str(p))
    assert img.shape == (40, 30, 3)
    assert np.allclose(img[20, 15], (10, 120, 230), atol=3)


def test_load_image_reads_heic_as_bgr(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    from PIL import Image
    pillow_heif.register_heif_opener()
    p = tmp_path / "card.heic"
    Image.fromarray(np.full((64, 48, 3), (200, 30, 40), np.uint8)).save(p)   # RGB
    img = cf.load_image(str(p))
    assert img.shape == (64, 48, 3)
    assert np.allclose(img[30, 20], (40, 30, 200), atol=4)                    # BGR


def test_load_image_missing_file(tmp_path):
    with pytest.raises(cf.CardError, match="no such file"):
        cf.load_image(str(tmp_path / "nope.jpg"))


def test_load_image_not_an_image(tmp_path):
    p = tmp_path / "notes.jpg"
    p.write_text("not a photo")
    with pytest.raises(cf.CardError, match="cannot read"):
        cf.load_image(str(p))


# --------------------------------------------------------------------------
# _clipped_highlights
# --------------------------------------------------------------------------

def test_clipped_highlights_masks_detail_clipped_in_one_photo_only():
    a = np.full((60, 60, 3), 120, np.uint8)
    b = a.copy()
    a[10:20, 10:20] = 255          # same light detail: clipped in A, bright in B
    b[10:20, 10:20] = 230
    a[40:44, 40:44] = 255          # whitening on a mid-tone area: B stays at 120
    valid = np.full((60, 60), 255, np.uint8)
    m = cf._clipped_highlights(a, b, valid)
    assert m[15, 15] == 255
    assert m[42, 42] == 0
    assert m[50, 5] == 0


# --------------------------------------------------------------------------
# _match_colour
# --------------------------------------------------------------------------

def test_match_colour_undoes_exposure_and_white_balance():
    rng = np.random.default_rng(0)
    a = rng.integers(20, 230, (120, 90, 3)).astype(np.uint8)
    b = np.clip(a.astype(np.float32) * (0.85, 0.92, 1.04) + (3, -2, 5), 0, 255).astype(np.uint8)
    b[50:54, 40:44] = 250                                     # a real mark must not pull the fit
    valid = np.full(a.shape[:2], 255, np.uint8)
    out = cf._match_colour(a, b, valid).astype(int)
    diff = np.abs(out - a.astype(int))
    diff[50:54, 40:44] = 0
    assert diff.max() <= 2


# --------------------------------------------------------------------------
# register_print
# --------------------------------------------------------------------------

def test_register_print_rejects_collapsed_transform(shots, monkeypatch):
    a, _ = cf.rectify(cf.load_image(shots["mint_a"]))
    b, _ = cf.rectify(cf.load_image(shots["mint_b"]))
    assert cf.register_print(a, b)[1] >= 30

    def collapsed(src, dst, **kw):                            # every B point onto one A point
        return np.float64([[0, 0, 200], [0, 0, 900]]), np.ones((len(src), 1), np.uint8)
    monkeypatch.setattr(cf.cv2, "estimateAffinePartial2D", collapsed)
    with pytest.raises(cf.CardError, match="same card design"):
        cf.register_print(a, b)


# --------------------------------------------------------------------------
# End to end on synthetic photos with known ground truth
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shots(tmp_path_factory):
    out = tmp_path_factory.mktemp("shots")
    rng = np.random.default_rng(11)
    pr = st.make_print()
    n = [0]

    def photo(copy, flip=False):
        n[0] += 1
        p = out / f"shot_{n[0]:02d}.jpg"
        cv2.imwrite(str(p), st.shoot(*copy, rng, flip=flip))
        return str(p)

    marks = [("speck", 30.0, 30.0, 0.3), ("scratch", 15.0, 25.0, 22.0, 29.0), ("edge", 1, 0.4, 1.3, 2.5)]
    mint = st.make_copy(pr, 0.2, -0.3, 0.05)
    return {
        "mint_a": photo(mint),
        "mint_b": photo(mint),
        "mint_flipped": photo(mint, flip=True),
        "shifted": photo(st.make_copy(pr, 0.2 + 0.3, -0.3, 0.05)),
        "worn": photo(st.make_copy(pr, 0.2, -0.3, 0.05, marks)),
        "marks": marks,
        "other_design": photo(st.make_copy(st.make_print(seed=99))),
    }


def test_same_copy_shows_no_difference(shots):
    r = cf.compare(shots["mint_a"], shots["mint_b"])
    assert r["reasons"] == []
    assert r["centering"]["max_displacement_mm"] < 0.05


def test_same_copy_upside_down_shows_no_difference(shots):
    r = cf.compare(shots["mint_a"], shots["mint_flipped"])
    assert r["reasons"] == []
    assert r["centering"]["max_displacement_mm"] < 0.05


def test_centering_offset_is_measured(shots):
    r = cf.compare(shots["mint_a"], shots["shifted"])
    assert r["reasons"] == ["centering"]
    assert abs(r["centering"]["dx_mm"] - 0.3) < 0.03
    assert abs(r["centering"]["dy_mm"]) < 0.03


def test_wear_marks_are_located(shots):
    r = cf.compare(shots["worn"], shots["mint_a"])
    assert r["reasons"] == ["surface marks"]
    pts = [(m["x_mm"], m["y_mm"]) for m in r["marks"]]
    planted = [(30.0, 30.0), (18.5, 27.0), (63.0, 0.4 * 88)]
    for x, y in planted:
        assert min(np.hypot(px - x, py - y) for px, py in pts) < 2.5, f"no mark near ({x}, {y}): {pts}"


def test_different_design_is_rejected(shots):
    with pytest.raises(cf.CardError):
        cf.compare(shots["mint_a"], shots["other_design"])


def test_photo_without_card_is_rejected(shots, tmp_path):
    blank = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank), np.full((4000, 3000, 3), (176, 196, 205), np.uint8))
    with pytest.raises(cf.CardError, match="no card found"):
        cf.compare(shots["mint_a"], str(blank))


@pytest.mark.parametrize("pair,code", [(("mint_a", "mint_b"), 0), (("mint_a", "shifted"), 1)])
def test_cli_exit_codes(shots, tmp_path, pair, code):
    report = tmp_path / "report.png"
    res = subprocess.run([sys.executable, str(REPO / "copy_fingerprint.py"), shots[pair[0]], shots[pair[1]],
                          "--out", str(report)], capture_output=True, text=True)
    assert res.returncode == code, res.stderr
    assert report.exists()


def test_cli_error_exit_code(tmp_path):
    res = subprocess.run([sys.executable, str(REPO / "copy_fingerprint.py"), str(tmp_path / "a.jpg"),
                          str(tmp_path / "b.jpg")], capture_output=True, text=True)
    assert res.returncode == 2
    assert "no such file" in res.stderr
