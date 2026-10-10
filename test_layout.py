"""Check the layout mapping model against a page whose true geometry we know.

Runs the real pipeline - scan the photo, flatten the sheet, project the ink, fit the
grid - and asserts the inches it reports match the page that was printed, and that the
predicted points actually land on ink.

    python make_test_page.py    # writes page_test.jpg
    python test_layout.py       # asserts the model recovers the geometry
"""
import sys

import cv2
import numpy as np

import layout
import make_test_page as fixture

# How close we must land, in inches. A character cell is 0.11 in wide, so these
# tolerances still mean "pointing at the right character".
TOLERANCE_IN = 0.07
INK_WINDOW_IN = 0.05


def check_photo():
    image = cv2.imread("page_test.jpg")
    assert image is not None, "run make_test_page.py first"

    sheet = layout.scan_page(image)
    print(f"sheet: {'detected in the photo' if sheet.detected else 'NOT DETECTED'}, "
          f"{sheet.paper} {sheet.size[0]:.2f} x {sheet.size[1]:.2f} in at {sheet.px_per_in:.0f} px/in")
    assert sheet.detected, "the sheet of paper should be found in this photo"

    # Orientation must survive the tilt: a portrait page stays portrait, text upright.
    true_w, true_h = fixture.PAPER[1], fixture.PAPER[2]
    assert abs(sheet.size[0] - true_w) < 0.01 and abs(sheet.size[1] - true_h) < 0.01, sheet.size

    page, _ = fixture.render_page()
    bands = fixture.true_lines(page)
    assert len(bands) == len(fixture.LINES), f"fixture should have {len(fixture.LINES)} lines"

    fitted = layout.fit_layout(sheet, [len(line) for line in fixture.LINES])
    print(f"layout: {fitted.describe()}")
    assert fitted.matched, f"lines of ink should line up with the transcript: {fitted.notes}"

    gray = cv2.cvtColor(sheet.image, cv2.COLOR_BGR2GRAY)
    ppi = sheet.px_per_in
    window = max(2, int(INK_WINDOW_IN * ppi))
    worst = 0.0

    for number, (line, word, _correction, _kind) in enumerate(fixture.MISTAKES, 1):
        text = fixture.LINES[line - 1]
        char = text.index(word) + 1

        x, y = fitted.position(line, char)
        true_x = fixture.MARGIN_IN + (char - 0.5) * fixture.PITCH_IN
        true_y = (bands[line - 1]["top"] + bands[line - 1]["bottom"]) / 2

        error = max(abs(x - true_x), abs(y - true_y))
        worst = max(worst, error)
        print(f"  {number}. line {line} char {char:3d} \"{word}\": "
              f"({x:.3f}, {y:.3f}) in vs true ({true_x:.3f}, {true_y:.3f}) in, "
              f"off by {error:.3f} in")
        assert error < TOLERANCE_IN, f"'{word}' landed {error:.3f} in from where it is"

        # The point must sit on ink, not in the margin or on the neighbouring word.
        col, row = int(x * ppi), int(y * ppi)
        patch = gray[max(row - window, 0):row + window, max(col - window, 0):col + window]
        assert patch.size and patch.min() < 128, f"no ink at the spot computed for '{word}'"

    print(f"worst miss: {worst:.3f} in (tolerance {TOLERANCE_IN} in)")

    marks = [
        (line, text.index(word) + 1, len(word), word)
        for (line, word, _correction, _kind) in fixture.MISTAKES
        for text in [fixture.LINES[line - 1]]
    ]
    marked = layout.draw_marks(sheet, fitted, marks)
    assert not np.array_equal(marked, sheet.image), "draw_marks should draw something"
    assert (marked[..., 2].astype(int) - marked[..., 0].astype(int) > 100).sum() > 500, \
        "the boxes should be red on the page"
    cv2.imwrite("page_test_marked.jpg", marked)
    print("wrote page_test_marked.jpg")
    return fitted


def check_flat_picture():
    """A flat screenshot has no sheet in it; the model must still answer in inches."""
    image = cv2.imread("text_test.png")
    if image is None:
        print("text_test.png not found - skipping the flat-picture check")
        return
    sheet = layout.scan_page(image)
    fitted = layout.fit_layout(sheet, [len(line) for line in ("x" * 42, "y" * 38)])
    print(f"flat picture: detected={sheet.detected}, {fitted.describe()}")
    assert not sheet.detected, "a screenshot has no sheet of paper to detect"
    x, y = fitted.position(1, 10)
    assert 0 <= x <= sheet.size[0] and 0 <= y <= sheet.size[1], (x, y)
    _, _, x1, y1 = fitted.span(1, 10, 4)
    assert x1 > x and y1 > y
    print(f"  line 1 char 10 -> ({x:.2f}, {y:.2f}) in on the picture")


def main():
    check_photo()
    check_flat_picture()
    print("\nall layout checks passed")


if __name__ == "__main__":
    sys.exit(main())
