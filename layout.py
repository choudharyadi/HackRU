"""Text layout mapping model: turn "line 3, character 10" into inches on the page.

A sheet of paper has a consistent structure - margins, line spacing and character
pitch - so a text position can be converted into an exact (x, y) point on the page.
That is what tells the robot *where* a mistake is, not just that one exists.

Two halves:

* :func:`scan_page` looks at a photo, finds the sheet of paper in it, flattens it to a
  straight top-down view (A4 or Letter), and measures the ink on it. Everything is
  then in inches from the top-left corner of the sheet, whatever the camera angle.
* :class:`PageLayout` fits the page grid to the ink: it finds the real lines of text
  with a row projection, lines them up with the transcript, and answers
  ``layout.position(line, char)`` in inches.

Use it without a page at all (a screenshot, a scan) - ``scan_page`` then treats the
whole picture as the sheet and everything still works, just at the picture's scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Standard paper sizes, portrait, in inches.
PAGE_SIZES = {"a4": (8.268, 11.693), "letter": (8.5, 11.0)}
DEFAULT_DPI = 150

# Tuning for the ink projections. The fractions are relative to the picture, so the
# same numbers work for a phone photo and for a flat scan.
INK_ROW_FRACTION = 0.005  # a row counts as inked when this share of it is dark
TRIM_FRACTION = 0.02  # ignore this much of the border, where shadows and edges live
MIN_LINE_FRACTION = 0.003  # ignore ink bands shorter than this share of the height
MERGE_GAP_FRACTION = 0.006  # bands nearer than this are one line, not two


# ----------------------------------------------------------------- geo shapes
@dataclass(frozen=True)
class Box:
    """A rectangle on the page, in inches from the top-left corner."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)


@dataclass(frozen=True)
class LineBand:
    """One detected line of ink: its vertical band and horizontal extent, in inches."""

    top: float
    bottom: float
    left: float
    right: float

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass
class Sheet:
    """The page as it lies on the desk, flattened to a straight top-down view."""

    size: tuple[float, float]  # inches (width, height)
    paper: str  # 'a4' or 'letter'
    dpi: float  # pixels per inch in `image`
    detected: bool  # True when a sheet was actually found in the photo
    image: np.ndarray  # the flattened page (BGR)

    @property
    def px_per_in(self) -> float:
        return self.image.shape[1] / self.size[0]


# ------------------------------------------------------------- decoding a photo
def decode(image_bytes) -> np.ndarray:
    """Photo bytes -> BGR array."""
    import cv2

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode that picture")
    return image


def _order_quad(points) -> np.ndarray:
    """Put four corners in the order top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(total)], pts[np.argmin(diff)], pts[np.argmax(total)], pts[np.argmax(diff)]],
        np.float32,
    )


def closest_page_size(aspect: float, paper: str | None = None):
    """Pick the standard paper whose shape is nearest.

    Returns (name, (width, height), error) with the orientation that matches `aspect`,
    so a landscape photo stays landscape and the text stays upright.
    """
    if paper and paper in PAGE_SIZES:
        width, height = PAGE_SIZES[paper]
        if aspect > 1.0:  # seen sideways: keep the sheet's own shape
            return paper, (height, width), abs(np.log(aspect / (height / width)))
        return paper, (width, height), abs(np.log(aspect / (width / height)))

    names = list(PAGE_SIZES)
    best = (names[0], PAGE_SIZES[names[0]], float("inf"))
    for name in names:
        width, height = PAGE_SIZES[name]
        for size, shape in (((width, height), width / height), ((height, width), height / width)):
            error = abs(np.log(aspect / shape))
            if error < best[2]:
                best = (name, size, error)
    return best


def find_sheet(image: np.ndarray) -> np.ndarray | None:
    """The four corners of the sheet of paper in the photo, or None if there is no sheet."""
    import cv2

    height, width = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # Paper is the big bright region. Try both polarities so a dark page on a light
    # desk also counts, and accept a quad at any of a few tolerances.
    masks = [
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]
    best, best_area = None, 0.12 * height * width
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= best_area:
                continue
            for epsilon in (0.02, 0.01, 0.03, 0.05):
                quad = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
                if len(quad) != 4 or not cv2.isContourConvex(quad):
                    continue
                ordered = _order_quad(quad)
                # The desk is the region running off the edge of the picture; the sheet
                # sits inside it, so anything touching the border is the background.
                if _touches_border(ordered, width, height):
                    continue
                best, best_area = ordered, area
                break
    return best


def _touches_border(quad: np.ndarray, width: int, height: int, margin: int = 3) -> bool:
    for x, y in quad:
        if x <= margin or y <= margin or x >= width - 1 - margin or y >= height - 1 - margin:
            return True
    return False


def warp_sheet(image: np.ndarray, quad: np.ndarray, size: tuple[float, float], dpi: float) -> np.ndarray:
    """Flatten the sheet: the four corners in the photo become the corners of the page."""
    import cv2

    out_w = int(round(size[0] * dpi))
    out_h = int(round(size[1] * dpi))
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


def scan_page(image: np.ndarray, dpi: float = DEFAULT_DPI, paper: str | None = None) -> Sheet:
    """Find the sheet of paper in the photo and lay it out flat.

    With no sheet in view (a scan, a screenshot) the whole picture is treated as the
    page, so the maths below still produces inches.
    """
    quad = find_sheet(image)
    if quad is not None:
        across = float(np.linalg.norm(quad[1] - quad[0]))
        down = float(np.linalg.norm(quad[3] - quad[0]))
        name, size, _ = closest_page_size(across / max(down, 1e-6), paper)
        return Sheet(size, name, float(dpi), True, warp_sheet(image, quad, size, dpi))

    height, width = image.shape[:2]
    name, size, _ = closest_page_size(width / max(height, 1), paper)
    return Sheet(size, name, float(width) / size[0], False, image)


# ------------------------------------------------------- measuring the ink
def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean mask of the dark ink on a page, ignoring the border."""
    import cv2

    height, width = gray.shape[:2]
    trim = int(TRIM_FRACTION * min(height, width))
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
    mask[:trim] = False
    mask[height - trim:] = False
    mask[:, :trim] = False
    mask[:, width - trim:] = False
    return mask


def find_lines(sheet: Sheet, mask: np.ndarray | None = None) -> list[LineBand]:
    """Locate the real lines of text by projecting the ink onto the vertical axis.

    Ascenders and descenders can split one line into a few bands, so bands that sit
    close together are merged. All lengths come back in inches, page-relative.
    """
    gray = _to_gray(sheet.image)
    mask = _ink_mask(gray) if mask is None else mask
    height, width = mask.shape
    ppi = width / sheet.size[0]
    rows = mask.sum(axis=1) >= max(2, int(INK_ROW_FRACTION * width))
    min_height = max(2.0, MIN_LINE_FRACTION * height)
    merge_gap = max(1.0, MERGE_GAP_FRACTION * height)

    runs: list[list[int]] = []
    row = 0
    while row < height:
        if not rows[row]:
            row += 1
            continue
        end = row
        while end < height and rows[end]:
            end += 1
        if runs and (row - runs[-1][1]) < merge_gap:
            runs[-1][1] = end  # part of the same line (a gap between ascenders)
        else:
            runs.append([row, end])
        row = end

    bands = []
    for top, bottom in runs:
        if (bottom - top) < min_height:
            continue
        columns = np.nonzero(mask[top:bottom].any(axis=0))[0]
        if columns.size == 0:
            continue
        bands.append(
            LineBand(
                top=top / ppi,
                bottom=bottom / ppi,
                left=float(columns.min()) / ppi,
                right=float(columns.max() + 1) / ppi,
            )
        )
    return bands


def _to_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ink_box(sheet: Sheet) -> Box | None:
    """Bounding box (inches) of all the ink on the page, or None if the page is blank."""
    mask = _ink_mask(_to_gray(sheet.image))
    rows = np.nonzero(mask.any(axis=1))[0]
    columns = np.nonzero(mask.any(axis=0))[0]
    if rows.size == 0 or columns.size == 0:
        return None
    ppi = sheet.px_per_in
    return Box(
        left=float(columns.min()) / ppi,
        top=float(rows.min()) / ppi,
        right=float(columns.max() + 1) / ppi,
        bottom=float(rows.max() + 1) / ppi,
    )


def block_box(bands: list[LineBand]) -> Box | None:
    """The smallest box holding every detected line."""
    if not bands:
        return None
    return Box(
        left=min(b.left for b in bands),
        top=min(b.top for b in bands),
        right=max(b.right for b in bands),
        bottom=max(b.bottom for b in bands),
    )


def nominal_box(size: tuple[float, float], margin: float = 1.0) -> Box:
    """The assumed text area of a page when nothing can be measured: one-inch margins."""
    left = min(margin, size[0] / 4)
    top = min(margin, size[1] / 4)
    return Box(left, top, size[0] - left, size[1] - top)


# ---------------------------------------------------------------- the model
@dataclass
class PageLayout:
    """The page's grid of lines and characters, in inches.

    ``line_lengths`` is how many characters each line of the transcript has, which is
    what turns a character index into a fraction across the line.
    """

    size: tuple[float, float]
    paper: str
    dpi: float
    text: Box  # the block of ink on the page
    line_lengths: list[int]
    rows: dict[int, LineBand] = field(default_factory=dict)  # transcript line -> measured band
    notes: list[str] = field(default_factory=list)

    @property
    def lines(self) -> int:
        return len(self.line_lengths)

    @property
    def matched(self) -> bool:
        """True when the measured lines of ink line up with the transcript."""
        return len(self.rows) > 0 and all(i in self.rows for i, n in enumerate(self.line_lengths) if n)

    @property
    def line_height(self) -> float:
        """Space between lines, in inches."""
        if len(self.rows) > 1:
            centers = [self.rows[i].center_y for i in sorted(self.rows)]
            return float(np.median(np.diff(centers)))
        return self.text.height / max(self.lines, 1)

    @property
    def char_width(self) -> float:
        """Average character pitch, in inches."""
        widths = [
            band.right - band.left
            for i, band in self.rows.items()
            if self.line_lengths[i]
        ]
        if widths:
            return float(np.mean(widths)) / max(max(self.line_lengths), 1)
        return self.text.width / max(max(self.line_lengths, default=1), 1)

    @property
    def chars_per_line(self) -> int:
        return max(self.line_lengths, default=0)

    def _metrics(self, line: int) -> tuple[int, float, float, float, int]:
        """(row index, ink left, ink right, line centre y, characters on that line)."""
        index = min(max(int(line), 1), max(self.lines, 1)) - 1
        band = self.rows.get(index)
        if band is not None:
            left, right, center = band.left, band.right, band.center_y
        else:
            left, right = self.text.left, self.text.right
            center = self.text.top + (index + 0.5) * self.line_height
        return index, left, right, center, max(self.line_lengths[index], 1)

    def position(self, line: int, char: int) -> tuple[float, float]:
        """Centre of character `char` of line `line`, in inches from the sheet's top-left.

        Both are 1-based, matching "line 3, character 10".
        """
        _, left, right, center, count = self._metrics(line)
        index = min(max(int(char), 1), count)
        return (left + (index - 0.5) * (right - left) / count, center)

    def span(self, line: int, char: int, length: int = 1) -> tuple[float, float, float, float]:
        """Box (x0, y0, x1, y1) in inches around characters `char` .. `char+length-1`."""
        _, left, right, center, count = self._metrics(line)
        index = min(max(int(char), 1), count)
        step = (right - left) / count
        width = left + (index - 1) * step
        end = width + max(1, min(int(length), count - index + 1)) * step
        half = max(self.line_height, 0.12) / 2
        return (width, center - half, end, center + half)

    def describe(self) -> str:
        source = "measured from the photo" if self.matched else "averaged grid"
        return (
            f"{self.paper} page {self.size[0]:.2f} x {self.size[1]:.2f} in, "
            f"text block {self.text.width:.2f} x {self.text.height:.2f} in, "
            f"{self.lines} line(s), {self.chars_per_line} char(s) at the longest, "
            f"line spacing {self.line_height:.2f} in, pitch {self.char_width:.3f} in "
            f"({source})"
        )


def fit_layout(sheet: Sheet, line_lengths: list[int], bands: list[LineBand] | None = None) -> PageLayout:
    """Fit the page grid to the writing.

    When the lines of ink on the page match the lines of the transcript one for one,
    each line is placed using its own measured band - so ragged edges and short lines
    still land in the right place. Otherwise the lines are spread evenly over the ink.
    """
    bands = find_lines(sheet) if bands is None else bands
    text = block_box(bands) or ink_box(sheet) or nominal_box(sheet.size)
    lengths = [int(n) for n in line_lengths] or [0]
    filled = [i for i, n in enumerate(lengths) if n > 0]

    notes: list[str] = []
    rows: dict[int, LineBand] = {}
    if bands and len(bands) == len(filled) and filled:
        rows = {i: band for i, band in zip(filled, bands)}
    elif bands:
        notes.append(
            f"saw {len(bands)} line(s) of ink but the transcript has {len(filled)}; "
            "using the average grid"
        )
    else:
        notes.append("no ink found to measure; using the average grid")

    return PageLayout(
        size=sheet.size,
        paper=sheet.paper,
        dpi=sheet.dpi,
        text=text,
        line_lengths=lengths,
        rows=rows,
        notes=notes,
    )


# ------------------------------------------------------------------ drawing
def draw_marks(
    sheet: Sheet,
    layout: PageLayout,
    marks: list[tuple[int, int, int, str]],
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw a numbered box around each mark, at the inches the model computed.

    `marks` is a list of (line, char, length, label).
    """
    import cv2

    out = sheet.image.copy()
    ppi = sheet.px_per_in
    height = out.shape[0]
    for number, (line, char, length, label) in enumerate(marks, 1):
        x0, y0, x1, y1 = layout.span(line, char, length)
        p0 = (int(x0 * ppi), int(max(y0, 0) * ppi))
        p1 = (int(x1 * ppi), int(min(y1, height / ppi) * ppi))
        cv2.rectangle(out, p0, p1, color, 2)
        cv2.putText(
            out,
            f"{number}: {label}",
            (p0[0], max(16, p0[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return out
