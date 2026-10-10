"""Render a simulated photo of a page so the layout mapping in layout.py can be tested.

Unlike make_test_text.py (a flat screenshot), this builds a real page whose geometry we
know exactly - A4, one-inch margins, 0.34 in between lines, 0.11 in per character - and
then tilts it into a photo on a dark desk. So we can check that scanning recovers the
true inches and points at the right characters.

    python make_test_page.py            # -> page_test.jpg
"""
import cv2
import numpy as np

DPI = 200
PAPER = ("a4", 8.268, 11.693)
MARGIN_IN = 1.0       # left margin, and top of the first line's cell
LINE_IN = 0.34        # baseline-to-baseline spacing
PITCH_IN = 0.11       # fixed character pitch, like a typewriter
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 1

LINES = [
    "i think its time to recieve the package.",
    "The dog run fast but teh cat is calm.",
    "There are alot of reasons to be happy",
    "We should of checked the weather.",
]

# (1-based line, the word that is wrong, what it should be, why)
MISTAKES = [
    (1, "i", "I", "capitalization"),
    (1, "its", "it's", "punctuation"),
    (1, "recieve", "receive", "spelling"),
    (2, "run", "runs", "grammar"),
    (2, "teh", "the", "spelling"),
    (3, "alot", "a lot", "spelling"),
    (4, "should of", "should have", "grammar"),
]


def pick_scale():
    """The largest font scale whose widest glyph still fits inside one character cell."""
    limit = 0.75 * PITCH_IN * DPI
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for scale in (0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4):
        widest = max(cv2.getTextSize(c, FONT, scale, THICKNESS)[0][0] for c in alphabet)
        if widest <= limit:
            return scale
    return 0.35


def render_page(scale=None):
    """A clean, flat page at DPI, with every character at its exact fixed-pitch cell."""
    scale = scale or pick_scale()
    width = int(round(PAPER[1] * DPI))
    height = int(round(PAPER[2] * DPI))
    page = np.full((height, width, 3), 255, np.uint8)
    for i, line in enumerate(LINES):
        baseline = int((MARGIN_IN + i * LINE_IN + 0.78 * LINE_IN) * DPI)
        for j, char in enumerate(line):
            x = int((MARGIN_IN + j * PITCH_IN) * DPI)
            cv2.putText(page, char, (x, baseline), FONT, scale, (0, 0, 0), THICKNESS)
    return page, scale


def photo_of(page: np.ndarray, seed: int = 3) -> np.ndarray:
    """Drop the page onto a dark desk at a slight angle, like a webcam shot."""
    height, width = page.shape[:2]
    pad_x, pad_y = 240, 300
    rng = np.random.default_rng(seed)
    desk = np.full((height + pad_y, width + pad_x, 3), 70, np.uint8)
    desk = np.clip(desk.astype(np.int16) + rng.integers(-4, 5, desk.shape), 0, 255).astype(np.uint8)

    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    corners = np.float32(
        [[150, 130], [width + 130, 70], [width + 60, height + 190], [70, height + 250]]
    )
    matrix = cv2.getPerspectiveTransform(source, corners)
    canvas_size = (width + pad_x, height + pad_y)
    page_in_photo = cv2.warpPerspective(page, matrix, canvas_size, borderValue=(70, 70, 70))
    page_area = cv2.warpPerspective(
        np.full((height, width, 3), 255, np.uint8), matrix, canvas_size, borderValue=(0, 0, 0)
    )
    shot = np.where(page_area > 0, page_in_photo, desk)
    return cv2.GaussianBlur(shot, (3, 3), 0)


def true_lines(page: np.ndarray) -> list[dict]:
    """Measure the clean page independently, giving the inches the model should recover.

    A plain row/column projection - deliberately not the code under test.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    ink = gray < 128
    rows = ink.any(axis=1)
    runs, row = [], 0
    while row < len(rows):
        if not rows[row]:
            row += 1
            continue
        end = row
        while end < len(rows) and rows[end]:
            end += 1
        if runs and row - runs[-1][1] < 8:  # the gap between ascenders, not a new line
            runs[-1][1] = end
        else:
            runs.append([row, end])
        row = end

    out = []
    for top, bottom in runs:
        if bottom - top < 10:
            continue
        columns = np.nonzero(ink[top:bottom].any(axis=0))[0]
        out.append(
            {
                "top": top / DPI,
                "bottom": bottom / DPI,
                "left": float(columns.min()) / DPI,
                "right": float(columns.max() + 1) / DPI,
            }
        )
    return out


def main():
    page, scale = render_page()
    shot = photo_of(page)
    cv2.imwrite("page_test.jpg", shot, [cv2.IMWRITE_JPEG_QUALITY, 92])
    bands = true_lines(page)
    print(f"wrote page_test.jpg ({shot.shape[1]}x{shot.shape[0]}, font scale {scale})")
    print(f"page {PAPER[0]}, {PAPER[1]} x {PAPER[2]} in at {DPI} dpi, "
          f"margin {MARGIN_IN} in, line spacing {LINE_IN} in, pitch {PITCH_IN} in")
    print(f"measured {len(bands)} line(s) on the clean page:")
    for i, band in enumerate(bands, 1):
        print(f"  {i}. y {band['top']:.3f}-{band['bottom']:.3f} in, "
              f"x {band['left']:.3f}-{band['right']:.3f} in")


if __name__ == "__main__":
    main()
