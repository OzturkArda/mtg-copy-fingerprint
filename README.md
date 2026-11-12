# card-diff

Tell two physical copies of the same trading-card printing apart from photos.

Copies of one printing carry no unique ID, so `card_diff.py` compares what physically differs:

1. **Cut registration**: where the print sits relative to the cut edge (centering and tilt).
2. **Surface marks**: wear, scratches, print specks, whitening.

## Usage

```
pip install -r requirements.txt
python card_diff.py a.jpg b.jpg --out report.png
python card_diff.py a.jpg b.jpg --json
```

Exit code: `0` no difference detected, `1` difference detected, `2` error.

## Photos

One unsleeved card per photo, shot from straight above on a plain background that contrasts
with the card border, no glare, card filling most of the frame. Fronts and backs both work.

Calibrate first: run it on two photos of the **same** card. The print displacement it reports
is your setup's noise floor, and `--tol-mm` must sit above it.

## How it works

- Finds each card's cut outline and rectifies it to a 63 x 88 mm canvas.
- Registers print-to-print with SIFT. Both images are already aligned on the cut, so the
  residual transform is the centering difference.
- Aligns on the print, cancels lighting differences, and maps the remaining marks.

## Testing

`synth_test.py` renders synthetic copies with known centering offsets and marks, photographs
them with random pose, perspective, lighting, shadow, lens distortion, blur, noise and JPEG
compression, and checks what the script recovers:

```
python synth_test.py /tmp/card-diff-test 8
```

Results on synthetic photos (8 trials per row, 12 MP shots at about 30 px/mm):

| Test | Result |
|---|---|
| Same copy, two photos | 8/8 no difference; print displacement median 0.009 mm, max 0.011 mm; 0 false marks |
| Centering differs by 0.20 to 1.00 mm | 32/32 detected; offset error at most 0.013 mm |
| Centering differs by 0.15 mm | 7/8 detected (sits on the default tolerance) |
| Centering differs by 0.10 mm | 0/8 flagged (below the default tolerance; lower `--tol-mm` to catch it) |
| Tilt differs by 0.3 degrees | 8/8 detected |
| Same centering, one copy worn | 8/8 detected; 29/32 planted marks located, 0 marks without a planted cause |
| Different card design, photo with no card | both rejected with an error |

**Not validated on real card photos.** Real photos add curl, uneven shadows, glare and dust,
so expect a higher noise floor than these numbers. The 0.15 mm default tolerance is a starting
point, not a measured value: calibrate it as described above.

## Limits

- Two mint copies whose centering matches within your noise floor cannot be separated.
- Wear confined to the outer 0.5 mm of the edge is not compared.
- Foils and sleeved cards are not supported: glare and reflections swamp the comparison.
- Dust on the card or lens shows up as marks. Check the diff image before trusting them.

![Example report](example_report.png)
