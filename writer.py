#!/usr/bin/env python3
"""
arm_writer.py - take the pen to every mistake solve.py found on the sheet.

Two steps:

  1. CALIBRATE at one of the sheet's corners, twice - the pen in two heights:
       P  pen TOUCHING the paper on that corner.  That fixes the page:
          everything the proofreader printed is inches from the sheet's top-left
          corner, so one corner plus the sheet's size pins down every mistake.
          It also fixes the paper's surface, which is where the pen marks.
          The default corner is the TOP-RIGHT; use --corner top-left (etc.) when
          the arm reaches the page better from another corner - the whole arm
          only spans about half a sheet at a time, so pick the corner on the
          side you want to work.
          Right after that the arm taps the page origin (0, 0) - the sheet's
          top-left corner - so one dot tells you if the mapping is accurate.
       H  pen HOVERING over that same corner, at the height you want it to fly
          between marks (lift with W).  That fixes the travel height.  Skip H and
          it travels PEN_TRAVEL above the surface it touched.

  2. TOUR.  Press T: the arm flies over each mistake at the hover height, drops the
     pen to the touched surface, lifts straight back up, and goes on to the next
     correction, in the order they were listed.  N marks one at a time; V hops
     round the four corners of the sheet so you can check the mapping is the
     right way round.

The report comes from solve.py, which saves it after every check, or any file
you point at:

    python solve.py --image page.jpg          # -> last_report.txt
    python writer.py                          # jog, press P, press H, press T
    python writer.py report.txt               # some other listing
    python writer.py report.txt --ref 12,-4.13,0 --hover 0.4 --tour --sim --fast

Keys (hold or tap):
  Arrows   move X / Y (JOG_STEP per tap)
  W / S    move pen up / down (JOG_STEP per tap)
  P        calibrate point 1: pen on the sheet's TOP-RIGHT corner, touching the paper
           - then it taps the page origin (top-left corner) as an accuracy test
  H        calibrate point 2: pen hovering over that corner at travel height
  T        tour every mistake: fly over, down, up, next
  N        mark the next mistake only
  V        hop to the next corner of the sheet (checks the page mapping)
  O / C    open / close the claw (the ONLY thing that moves the claw servo)
  D        down 0.25", smooth straight line in +X, back up
  SPACE    down 0.25", smooth circle;  X = an X;  4 = the digit 4
  Q        quit

Jogging, touring and drawing never touch the claw servo.
"""

from __future__ import annotations

try:
    import serial
except Exception:
    serial = None

try:
    from pynput import keyboard
except Exception:
    keyboard = None

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from math import acos, atan2, cos, fabs, pi, sin, sqrt
from statistics import median
from typing import Optional

# --- serial ---
SERIAL_PORT = '/dev/tty.usbserial-2130'
BAUD_RATE = 9600

# --- arm ---
A, B = 5.75, 7.375            # AL5D link lengths, inches
BASE_PIN, SHOULDER_PIN, ELBOW_PIN = 0, 1, 2
WRIST_PIN, GRIPPER_PIN, WRIST_ROTATE_PIN = 3, 4, 5
TIP_OFFSET = (0.0, 0.0, -2.0)  # pen tip relative to wrist
LEVEL_OFFSET = 0.0

# --- drawing ---
JOG_STEP = 0.06                # inches per jog tap
PEN_TRAVEL = 0.35             # pen-up height when travelling between marks (higher lift)
PEN_DRAW = 0.36                # pen-down press when drawing a mark (a bit more than travel)
LINE_LEN = 3.0                 # straight segment length, inches (in +X)
X_SPAN = 3.0                  # X width of the X, inches
X_HEIGHT = 2.0                # Y height of the X, inches (centered on start)
CIRCLE_R = 0.5               # circle radius, inches (circle sits ahead in +X)
FOUR_H = 2.0                  # height of the digit 4, inches

# --- the sheet ---
# Portrait sizes, mirroring layout.PAGE_SIZES - writer.py stays importable
# without the vision stack, so the numbers live here too.
PAGE_SIZES = {"a4": (8.268, 11.693), "letter": (8.5, 11.0)}
DEFAULT_REPORT = 'last_report.txt'
PRESS = 0.0                    # extra push below the calibrated touch point, inches
MARK_LEN = 0.0                 # mark length in inches; 0 = a dot (down, then straight up)

# How the sheet sits under the arm: assumed square to the arm's own axes, with the
# reference corner at the sheet's TOP-RIGHT.
#   ACROSS_SIGN +1 : arm +X points at the sheet's RIGHT edge  (page x grows with arm x)
#   DOWN_SIGN   +1 : arm +Y points at the sheet's BOTTOM edge (page y grows with arm y)
# Flip a sign if the arm walks the wrong way across the paper.
ACROSS_SIGN = +1
DOWN_SIGN = +1

# --- motion ---
STEP = 0.05                   # inches per command
SPEED = 0.5                   # in/sec, while the pen is down
TRAVEL_SPEED = 1.5            # in/sec, hopping between marks with the pen up
STEP_MS = max(80, int(1000.0 * STEP / SPEED))   # time per step -> constant glide

# --- claw (overridden by pen_calibration.json) ---
CAL_FILE = 'pen_calibration.json'
GRIPPER_OPEN_PULSE = 2500      # claw wide open
GRIPPER_CLOSE_PULSE = 700      # claw clamped

# --- report parsing ---
ITEM_RE = re.compile(r'^\s*(\d+)\.\s+"(.*?)"\s*->\s*"(.*?)"\s*(?:\((.*)\))?\s*$')
POS_RE = re.compile(
    r"position:\s*line\s+(\d+),\s*char\s+(\d+)\s*->\s*([0-9.]+)\s*in from the left,\s*"
    r"([0-9.]+)\s*in down\s*(?:\(([0-9.]+),\s*([0-9.]+)\s*of the page\))?"
)

ser = None
curr_x, curr_y, curr_z = 6.0, 0.0, 4.0
FAST = False                  # --fast: don't wait for the arm between moves
SHEET = None                  # the calibrated sheet, once a report is loaded


def delay(seconds):
    """Wait for the arm - unless we are running --fast (simulation, tests)."""
    if not FAST:
        time.sleep(seconds)


def load_cal():
    global TIP_OFFSET, LEVEL_OFFSET
    global GRIPPER_OPEN_PULSE, GRIPPER_CLOSE_PULSE
    if not os.path.exists(CAL_FILE):
        return
    try:
        with open(CAL_FILE) as f:
            cal = json.load(f)
        LEVEL_OFFSET = cal.get('level_offset', LEVEL_OFFSET)
        GRIPPER_OPEN_PULSE = cal.get('gripper_open', GRIPPER_OPEN_PULSE)
        GRIPPER_CLOSE_PULSE = cal.get('gripper_close', GRIPPER_CLOSE_PULSE)
        if len(cal.get('tip_offset', [])) == 3:
            TIP_OFFSET = tuple(cal['tip_offset'])
    except Exception as e:
        print(f"[cal] {e}")


def getPulse(angle):
    angle = max(min(angle, 180.0), 0.0)
    return int(angle * (2500 - 500) / 180.0 + 500)


def tip_at(x, y, z):
    """Wrist position that puts the pen tip at this world point."""
    return (x - TIP_OFFSET[0], y - TIP_OFFSET[1], z - TIP_OFFSET[2])


def reachable(x, y, z):
    wx, wy, wz = tip_at(x, y, z)
    if wx <= 0:
        return False
    R = sqrt(wx * wx + wy * wy)
    return fabs(A - B) < sqrt(R * R + wz * wz) < (A + B)


def send(x, y, z, t_ms):
    """Send one arm move for the pen tip (no claw channel).
    Returns False if unreachable."""
    global curr_x, curr_y, curr_z
    wx, wy, wz = tip_at(x, y, z)
    if wx <= 0:
        return False
    R = sqrt(wx * wx + wy * wy)
    M = sqrt(R * R + wz * wz)
    if M <= 0 or M > (A + B) or M < fabs(A - B):
        return False
    try:
        a1 = atan2(wz, R)
        a2 = acos(max(min((A * A - B * B + M * M) / (2 * A * M), 1.0), -1.0))
        elbow = acos(max(min((A * A + B * B - M * M) / (2 * A * B), 1.0), -1.0))
        S = (a1 + a2) * (180.0 / pi)
        E = elbow * (180.0 / pi)
        wrist = 270.0 - S - E + LEVEL_OFFSET
        cmd = (f"#{BASE_PIN} P{getPulse(90.0 + atan2(wy, wx) * (180.0 / pi))} "
               f"#{SHOULDER_PIN} P{getPulse(S)} "
               f"#{ELBOW_PIN} P{getPulse(180 - E)} "
               f"#{WRIST_PIN} P{getPulse(wrist)} "
               f"#{WRIST_ROTATE_PIN} P{getPulse(90.0)} T{int(t_ms)}\r")
    except Exception:
        return False
    if ser is not None:
        ser.write(cmd.encode('utf-8'))
    curr_x, curr_y, curr_z = x, y, z
    return True


def move(x, y, z, t_ms, wait=True):
    if not send(x, y, z, t_ms):
        raise RuntimeError(f"point ({x:.2f}, {y:.2f}, {z:.2f}) out of reach")
    if wait:
        delay(t_ms / 1000.0)


def set_gripper(p, label):
    """Open or close the claw - the only place the claw ever moves."""
    if ser is not None:
        ser.write(f"#{GRIPPER_PIN} P{int(p)} T400\r".encode('utf-8'))
    delay(0.4)
    print(f"claw {label}")


def jog(dx, dy, dz):
    """One jog tap: move JOG_STEP inches in the given direction."""
    nx = curr_x + dx * JOG_STEP
    ny = curr_y + dy * JOG_STEP
    nz = curr_z + dz * JOG_STEP
    if send(nx, ny, nz, 300):
        print(f"pos -> X: {curr_x:.2f}  Y: {curr_y:.2f}  Z: {curr_z:.2f}")
    else:
        print("blocked (out of reach)")


def stroke(x0, y0, x1, y1, z_draw):
    """One constant-speed diagonal from (x0, y0) to (x1, y1) at z_draw."""
    length = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    n = max(1, int(round(length / STEP)))
    for i in range(1, n + 1):
        f = i / n
        move(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, z_draw, STEP_MS)


def travel_ms(x, y):
    """How long to hop to (x, y) with the pen up."""
    dist = sqrt((x - curr_x) ** 2 + (y - curr_y) ** 2)
    return int(max(250.0, 1000.0 * dist / TRAVEL_SPEED))


# --------------------------------------------------------------- the report
@dataclass
class Mark:
    """One mistake from the report and where it sits on the sheet, in inches."""

    number: int
    original: str
    correction: str
    kind: str = ""
    line: int = 0
    char: int = 0
    x: Optional[float] = None   # inches from the sheet's left edge
    y: Optional[float] = None   # inches from the sheet's top edge
    frac: Optional[tuple] = None  # (x, y) as a fraction of the page, if printed

    @property
    def placed(self) -> bool:
        return self.x is not None and self.y is not None

    @property
    def label(self) -> str:
        return f'{self.number}. "{self.original}" -> "{self.correction}"'


def parse_report(text):
    """The listing solve.py prints -> the mistakes, each with its inches.

    Stops at the "Corrected:" block, which is the writing itself and may contain
    anything, numbered lines included.
    """
    marks, current = [], None
    for raw in (text or "").splitlines():
        if raw.strip().startswith("Corrected:"):
            break
        item = ITEM_RE.match(raw)
        if item:
            current = Mark(
                number=int(item.group(1)),
                original=item.group(2),
                correction=item.group(3),
                kind=(item.group(4) or "").strip(),
            )
            marks.append(current)
            continue
        pos = POS_RE.search(raw)
        if pos and current is not None:
            current.line, current.char = int(pos.group(1)), int(pos.group(2))
            current.x, current.y = float(pos.group(3)), float(pos.group(4))
            if pos.group(5) and pos.group(6):
                current.frac = (float(pos.group(5)), float(pos.group(6)))
    return marks


def _orient(portrait, guess):
    """The portrait size, turned landscape if the report says the page was wide."""
    w, h = portrait
    if guess and guess[0] > guess[1]:
        return (h, w)
    return (w, h)


def _snap(size):
    """Nearest standard sheet, when the report's own inches are close enough."""
    w, h = size
    best = None
    for name, (pw, ph) in PAGE_SIZES.items():
        for cand in ((pw, ph), (ph, pw)):
            err = max(abs(cand[0] / w - 1.0), abs(cand[1] / h - 1.0))
            if best is None or err < best[1]:
                best = (cand, err)
    return best[0] if best[1] <= 0.06 else (w, h)


def paper_label(size):
    for name, (pw, ph) in PAGE_SIZES.items():
        if abs(size[0] - pw) < 0.05 and abs(size[1] - ph) < 0.05:
            return name
        if abs(size[0] - ph) < 0.05 and abs(size[1] - pw) < 0.05:
            return f"{name} landscape"
    return "measured"


def choose_page(marks, paper="auto"):
    """The sheet the report was measured on: (size in inches, label).

    solve.py prints every position twice - inches and "of the page" - so the two
    together give the page back (x / fraction). The fractions are only printed to
    two decimals, so the median is snapped onto a standard sheet.
    """
    widths = [m.x / m.frac[0] for m in marks
              if m.frac and m.x is not None and m.x >= 0.1 and m.frac[0] > 0.005]
    heights = [m.y / m.frac[1] for m in marks
               if m.frac and m.y is not None and m.y >= 0.1 and m.frac[1] > 0.005]
    guess = (median(widths), median(heights)) if widths and heights else None
    if paper != "auto":
        size = _orient(PAGE_SIZES[paper], guess)
        return size, paper_label(size)
    if guess:
        size = _snap(guess)
        return size, paper_label(size)
    return PAGE_SIZES["a4"], "a4"


# ------------------------------------------------------------- the page map
class SheetMap:
    """Where the sheet lies in the arm's world, calibrated at its top-right corner."""

    CORNERS = ("top-left", "bottom-left", "bottom-right", "top-right")
    # Where the reference corner sits in page inches: everything the report
    # printed is measured from the top-left corner (0, 0).
    REF_AT = {"top-left": (0.0, 0.0), "top-right": (1.0, 0.0),
              "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0)}

    def __init__(self, size, marks, label="a4", ref_corner="top-right"):
        if ref_corner not in self.REF_AT:
            raise ValueError(f"unknown corner {ref_corner!r} - "
                             f"one of {', '.join(self.CORNERS)}")
        self.size = size          # (width, height) in inches, as the report measured it
        self.label = label
        self.marks = marks
        self.ref_corner = ref_corner   # which paper corner the pen touches at P
        self.ref = None           # arm (x, y, z) of that corner, pen touching
        self.hover = None         # arm z of the second point: the pen hovering there
        self.cursor = 0           # next mistake to mark
        self._corner = 0

    # -- calibration -------------------------------------------------------
    @property
    def calibrated(self):
        return self.ref is not None

    def calibrate(self, x, y, z, tap=True):
        """Record the pen, touching the paper, at the sheet's reference corner."""
        self.ref = (x, y, z)
        w, h = self.size
        print(f"[page] {self.label} {w:.2f} x {h:.2f} in, {self.ref_corner} corner -> "
              f"arm X {x:.2f}  Y {y:.2f}  Z {z:.2f} (pen on the paper)")
        top_left = self.corner_xy("top-left")
        print(f"[page] the sheet's top-left corner (the page origin) is at arm X {top_left[0]:.2f}  "
              f"Y {top_left[1]:.2f} - flip ACROSS_SIGN / DOWN_SIGN if the tour goes the wrong way")
        if not self.marks:
            print("[page] no mistakes loaded - run solve.py first, or pass a report file")
            if tap:
                self.tap_origin()
            return
        for m in self.marks:
            if m.placed:
                ax, ay = self.xy(m.x, m.y)
                print(f"  {m.number:>2}. {m.original!r} -> {m.correction!r}  "
                      f"page ({m.x:.2f}, {m.y:.2f}) in -> arm ({ax:.2f}, {ay:.2f})")
            else:
                print(f"  {m.number:>2}. {m.original!r} -> {m.correction!r}  (no position - skipped)")
        if tap:
            self.tap_origin()
        if self.hover is None:
            print(f"press H to fly at {self.z_travel:.2f} in (the height it just tapped "                  f"from), or press T to travel {PEN_TRAVEL:.2f} in above the paper"
)
        print("press T to mark them all, N one at a time, V to walk the corners")

    def calibrate_hover(self, z):
        """Second calibration point: the pen hovering over the same corner at z.

        That is the height the arm flies at between marks.  Returns False (and
        leaves the height alone) when it is not above the paper surface.
        """
        if not self.calibrated:
            print("[page] press P at the sheet's top-right corner before hovering (H)")
            return False
        if z <= self.ref[2]:
            print(f"[page] hover z {z:.2f} is not above the paper (touch z {self.ref[2]:.2f}) - "
                  f"ignored, still travelling {self.z_travel - self.ref[2]:.2f} in above it")
            return False
        self.hover = z
        print(f"[page] hovering point -> arm Z {z:.2f} ({z - self.ref[2]:.2f} in above the paper); "
              f"marks still land at z {self.z_mark:.2f}")
        return True

    def xy(self, px, py):
        """Sheet inches (from the top-left corner) -> arm (x, y). Needs calibration."""
        rx, ry, _ = self.ref
        fx, fy = self.REF_AT[self.ref_corner]
        rcx, rcy = fx * self.size[0], fy * self.size[1]   # the corner, in page inches
        return (rx + ACROSS_SIGN * (px - rcx), ry + DOWN_SIGN * (py - rcy))

    def corner_xy(self, name):
        w, h = self.size
        at = {"top-right": (w, 0.0), "top-left": (0.0, 0.0),
              "bottom-left": (0.0, h), "bottom-right": (w, h)}[name]
        return self.xy(*at)

    @property
    def z_mark(self):
        """Pen-down height: the paper surface as calibrated, plus any extra press."""
        return self.ref[2] - PRESS

    @property
    def z_travel(self):
        """Pen-up height between marks: the hovering point, or PEN_TRAVEL above the paper."""
        if self.hover is not None:
            return self.hover
        return self.ref[2] + PEN_TRAVEL

    # -- the accuracy test -------------------------------------------------
    def tap_origin(self):
        """Fly to the page origin (0, 0 in - the sheet's top-left corner) and tap
        it: fly over, pen down, pen up.  The dot should land right on that corner;
        if it does not, the sheet is not square to the arm's axes (or flip
        ACROSS_SIGN / DOWN_SIGN)."""
        if not self.calibrated:
            return False
        x, y = self.xy(0.0, 0.0)
        if not reachable(x, y, self.z_travel) or not reachable(x, y, self.z_mark):
            print(f"[page] origin (0, 0) in -> arm ({x:.2f}, {y:.2f}) out of reach - "
                  f"no accuracy tap")
            return False
        print(f"[page] accuracy test: the origin (0, 0) in -> arm ({x:.2f}, {y:.2f}); "
              f"tapping it at z {self.z_mark:.2f}")
        move(x, y, self.z_travel, travel_ms(x, y))   # fly over, pen up
        move(x, y, self.z_mark, 300)                 # pen down on the corner
        delay(0.15)                                  # let the ink land
        move(x, y, self.z_travel, 300)               # pen up
        print("[page] the dot should sit on the sheet's top-left corner - the origin "
              "the report measures every mistake from")
        return True

    # -- the tour ----------------------------------------------------------
    def mark(self, m):
        """Fly over one mistake, pen down, pen up. True when it was marked."""
        if not self.calibrated:
            print(f"[page] not calibrated - jog to the sheet's {self.ref_corner} corner and press P")
            return False
        if not m.placed:
            print(f"  {m.label}  (no position in the report - skipped)")
            return False
        x, y = self.xy(m.x, m.y)
        if not reachable(x, y, self.z_travel) or not reachable(x, y, self.z_mark):
            print(f"  {m.label}  page ({m.x:.2f}, {m.y:.2f}) in -> arm ({x:.2f}, {y:.2f}) "
                  f"- out of reach, skipped")
            return False
        print(f"  {m.label}  page ({m.x:.2f}, {m.y:.2f}) in -> arm ({x:.2f}, {y:.2f})")
        move(x, y, self.z_travel, travel_ms(x, y))      # fly over, pen up
        move(x, y, self.z_mark, 300)                    # pen down
        delay(0.15)                                     # let the ink land
        ex = x
        if MARK_LEN > 0:                                # optional underline in +X
            end = x + ACROSS_SIGN * MARK_LEN
            if reachable(end, y, self.z_mark):
                stroke(x, y, end, y, self.z_mark)
                ex = end
            else:
                print("    (the underline would leave the reachable area - dotting instead)")
        move(ex, y, self.z_travel, 300)                 # pen up
        return True

    def mark_next(self):
        """Mark the next unmarked mistake. Returns True when it landed."""
        if not self.calibrated:
            print(f"[page] not calibrated - jog to the sheet's {self.ref_corner} corner and press P")
            return False
        if self.cursor >= len(self.marks):
            print("[tour] every mistake has been marked")
            return False
        m = self.marks[self.cursor]
        ok = self.mark(m)
        self.cursor += 1
        return ok

    def tour(self):
        """Mark every remaining mistake, in report order. Returns how many landed."""
        if not self.calibrated:
            print(f"[page] not calibrated - jog to the sheet's {self.ref_corner} corner and press P "
                  "(or run with --ref X,Y,Z)")
            return 0
        if not self.marks:
            print("[tour] no mistakes loaded")
            return 0
        start = self.cursor
        if start >= len(self.marks):
            print("[tour] every mistake has been marked")
            return 0
        marked = 0
        while self.cursor < len(self.marks):
            if self.mark_next():
                marked += 1
        print(f"[tour] marked {marked} of {len(self.marks) - start} remaining mistake(s); "
              f"{len(self.marks)} total on the page")
        return marked

    def visit_next_corner(self):
        """Hop to the next corner of the sheet, to check the page mapping."""
        if not self.calibrated:
            print("[page] not calibrated - jog to the sheet's top-right corner and press P")
            return
        name = self.CORNERS[self._corner % len(self.CORNERS)]
        self._corner += 1
        x, y = self.corner_xy(name)
        if not reachable(x, y, self.z_travel):
            print(f"[corner] {name} ({x:.2f}, {y:.2f}) out of reach")
            return
        print(f"[corner] {name} of the sheet -> arm ({x:.2f}, {y:.2f})")
        move(x, y, self.z_travel, travel_ms(x, y))


# ------------------------------------------------------------- simple shapes
def draw_segment(x0, y0, x1, y1):
    """Draw a smooth straight segment between two pen-down points at the
    current height.  Down to (x0,y0), glide to (x1,y1), lift back to (x1,y1)."""
    z_top = curr_z
    z_draw = curr_z - PEN_DRAW
    for gx, gy in ((x0, y0), (x1, y1)):
        if not reachable(gx, gy, z_draw):
            print(f"[draw] point ({gx:.2f}, {gy:.2f}) out of reach - jog first")
            return
    print(f"[draw] from ({x0:.2f}, {y0:.2f}) to ({x1:.2f}, {y1:.2f})")
    move(x0, y0, z_draw, 400)                        # go down
    stroke(x0, y0, x1, y1, z_draw)                  # constant-speed glide
    move(x1, y1, z_top, 400)                        # back up
    print("[draw] done.")


def run_x():
    """Down 0.25", two smooth diagonals crossing in an X, back up.
    Claw untouched."""
    z_top = curr_z
    z_draw = curr_z - PEN_DRAW
    x0, y0 = curr_x, curr_y
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = (
        (x0, y0 - X_HEIGHT / 2), (x0 + X_SPAN, y0 + X_HEIGHT / 2),
        (x0, y0 + X_HEIGHT / 2), (x0 + X_SPAN, y0 - X_HEIGHT / 2))
    for gx, gy in ((ax, ay), (bx, by), (cx, cy), (dx, dy)):
        if not reachable(gx, gy, z_draw):
            print(f"[x] corner ({gx:.2f}, {gy:.2f}) out of reach - jog first")
            return
    print(f"[x] down {PEN_DRAW:.2f}\\\", {X_SPAN:.2f}x{X_HEIGHT:.2f}\\\" X "
          f"at {SPEED} in/sec")
    move(ax, ay, z_top, 400)
    move(ax, ay, z_draw, 400)
    stroke(ax, ay, bx, by, z_draw)
    move(bx, by, z_top, 400)
    move(cx, cy, z_top, 400)
    move(cx, cy, z_draw, 400)
    stroke(cx, cy, dx, dy, z_draw)
    move(dx, dy, z_top, 400)
    print("[x] done.")


def run_four():
    """Down 0.25", draw the digit 4, back up. Claw untouched.
    Starts at the top of the stem; the shape grows down-left of there."""
    z_top = curr_z
    z_draw = curr_z - PEN_DRAW
    x0, y0 = curr_x, curr_y
    d = FOUR_H * 0.45
    strokes = [((x0, y0), (x0 - d, y0 + d)),
               ((x0 - d - 0.1, y0 + d), (x0 + 0.4, y0 + d)),
               ((x0, y0), (x0, y0 + FOUR_H))]
    for (sx, sy), (ex, ey) in strokes:
        for gx, gy in ((sx, sy), (ex, ey)):
            if not reachable(gx, gy, z_draw):
                print(f"[4] point ({gx:.2f}, {gy:.2f}) out of reach - jog first")
                return
    print(f"[4] down {PEN_DRAW:.2f}\\\", digit {FOUR_H:.2f}\" tall at {SPEED} in/sec")
    for i, ((sx, sy), (ex, ey)) in enumerate(strokes):
        if i:
            move(sx, sy, z_top, 400)
        move(sx, sy, z_draw, 400)
        stroke(sx, sy, ex, ey, z_draw)
        move(ex, ey, z_top, 400)
    print("[4] done.")


def run_circle():
    """Down 0.25", smooth circle starting on its left edge, back up.
    Claw untouched."""
    z_top = curr_z
    z_draw = curr_z - PEN_DRAW
    cx, cy = curr_x + CIRCLE_R, curr_y
    n = max(48, int(round(2 * pi * CIRCLE_R / STEP)))
    pts = [(cx + CIRCLE_R * cos(pi + 2 * pi * i / n),
            cy + CIRCLE_R * sin(pi + 2 * pi * i / n)) for i in range(n + 1)]
    for gx, gy in pts[::max(1, n // 16)]:
        if not reachable(gx, gy, z_draw):
            print(f"[circle] point ({gx:.2f}, {gy:.2f}) out of reach - jog left first")
            return
    print(f"[circle] down {PEN_DRAW:.2f}\\\", r={CIRCLE_R:.2f}\\\" "
          f"at {SPEED} in/sec")
    move(pts[0][0], pts[0][1], z_draw, 400)
    for x, y in pts[1:]:
        move(x, y, z_draw, STEP_MS)
    move(pts[-1][0], pts[-1][1], z_top, 400)
    print("[circle] done.")


# ------------------------------------------------------------------- keys
def on_press(key):
    try:
        name = getattr(key, 'name', '')
        if name == 'up':
            jog(1, 0, 0)
        elif name == 'down':
            jog(-1, 0, 0)
        elif name == 'left':
            jog(0, 1, 0)
        elif name == 'right':
            jog(0, -1, 0)
        elif name == 'space':
            run_circle()
        else:
            ch = (getattr(key, 'char', None) or '').lower()
            if ch == 'q':
                return False
            if ch == 'w':
                jog(0, 0, 1)
            elif ch == 's':
                jog(0, 0, -1)
            elif ch == 'p':
                SHEET.calibrate(curr_x, curr_y, curr_z)
            elif ch == 'h':
                SHEET.calibrate_hover(curr_z)
            elif ch == 't':
                SHEET.tour()
            elif ch == 'n':
                SHEET.mark_next()
            elif ch == 'v':
                SHEET.visit_next_corner()
            elif ch == '`':
                set_gripper(GRIPPER_OPEN_PULSE, "open")
            elif ch == 'c':
                set_gripper(GRIPPER_CLOSE_PULSE, "closed")
            elif ch == 'x':
                run_x()
            elif ch == '4':
                run_four()
            elif ch == 'd':
                draw_segment(curr_x, curr_y, curr_x + LINE_LEN, curr_y)
    except Exception as e:
        print(f"[listener] {e}")


# -------------------------------------------------------------------- main
def parse_ref(text):
    """'X,Y,Z' -> the arm point of the sheet's top-right corner."""
    parts = [p for p in text.replace(" ", ",").split(",") if p]
    if len(parts) != 3:
        raise ValueError(f"--ref wants three numbers X,Y,Z (got {text!r})")
    return tuple(float(p) for p in parts)


def show_report(sheet):
    print(f"Found {len(sheet.marks)} mistake(s) on a {sheet.label} page "
          f"({sheet.size[0]:.2f} x {sheet.size[1]:.2f} in):")
    for m in sheet.marks:
        where = f"page ({m.x:.2f}, {m.y:.2f}) in" if m.placed else "no position"
        kind = f"  ({m.kind})" if m.kind else ""
        print(f"  {m.number}. {m.original!r} -> {m.correction!r}{kind}  {where}")


def open_serial():
    global ser
    if serial is None:
        print("[serial] pyserial missing -> running SIMULATED.")
        return
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        delay(2)
        move(curr_x, curr_y, curr_z, 800)
        print(f"[serial] {SERIAL_PORT} open.")
    except Exception as e:
        print(f"[serial] {e} -> running SIMULATED.")
        ser = None


def close_serial():
    if ser is not None and ser.is_open:
        ser.close()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Point the pen at every mistake in a solve.py report.",
        epilog="Two calibration points, both over the sheet's top-right corner:\n"
               "press P with the pen touching the paper, then lift it (W) to travel\n"
               "height and press H. Press T to tour, N for one mistake, V for the corners.",
    )
    parser.add_argument("report", nargs="?", default=None,
                        help=f"the 'Found N mistake(s)' listing (default: {DEFAULT_REPORT})")
    parser.add_argument("--ref", metavar="X,Y,Z",
                        help="arm position of the reference corner (the sheet's top-right "
                             "by default) with the pen touching the paper - skips the "
                             "first calibration point")
    parser.add_argument("--corner", choices=SheetMap.CORNERS, default="top-right",
                        help="which paper corner the pen touches at calibration; pick the "
                             "one on the side the arm can reach (default top-right)")
    parser.add_argument("--hover", type=float, metavar="Z",
                        help="arm Z to fly at between marks: the pen hovering over that "
                             "corner (default: the touch point + 0.25 in)")
    parser.add_argument("--tour", action="store_true",
                        help="mark every mistake, then exit (needs --ref)")
    parser.add_argument("--paper", choices=("auto", "a4", "letter"), default="auto",
                        help="sheet to assume; 'auto' reads it out of the report (default)")
    parser.add_argument("--mark-len", type=float, default=None, metavar="IN",
                        help=f"draw a stroke this long at each mistake, in inches "
                             f"(default {MARK_LEN:g} = a dot)")
    parser.add_argument("--sim", action="store_true",
                        help="don't open the serial port - dry run on a desk with no arm")
    parser.add_argument("--fast", action="store_true",
                        help="don't wait for the arm between moves (simulation/testing)")
    return parser


def main():
    global FAST, SHEET, MARK_LEN
    args = build_parser().parse_args()
    FAST = args.fast
    if args.mark_len is not None:
        MARK_LEN = args.mark_len

    path = args.report or DEFAULT_REPORT
    if os.path.exists(path):
        with open(path) as f:
            text = f.read()
    elif args.report:
        sys.exit(f"report not found: {path}")
    else:
        text = ""
        print(f"[no {path} - run solve.py first, or pass the report as an argument]")

    marks = parse_report(text)
    size, label = choose_page(marks, args.paper)
    SHEET = SheetMap(size, marks, label, args.corner)
    show_report(SHEET)

    if args.tour and not args.ref:
        sys.exit("--tour needs --ref X,Y,Z (or run interactively and press T)")
    if args.hover is not None and not args.ref:
        sys.exit("--hover needs --ref too: the hovering point is measured from the "
                 "corner the pen first touched")

    load_cal()
    if args.sim:
        print("[serial] --sim: not opening the port -> running SIMULATED.")
    else:
        open_serial()

    if args.ref:
        try:
            SHEET.calibrate(*parse_ref(args.ref))
        except ValueError as exc:
            sys.exit(str(exc))
        if args.hover is not None and not SHEET.calibrate_hover(args.hover):
            sys.exit(f"--hover {args.hover} must be above the touch point "
                     f"z {SHEET.ref[2]:.2f}")
    if args.tour and SHEET.hover is None and SHEET.calibrated:
        print(f"[page] no hovering point - flying {PEN_TRAVEL:.2f} in above the paper it touched")

    if args.tour:
        SHEET.tour()
        close_serial()
        return

    if not args.ref:
        print(__doc__)
        print(f"Jog the pen to the sheet's {args.corner.upper()} corner (arrows + W/S, pen up),\n"
              "lower it until it just touches the paper (S), then press P to calibrate.")
    if keyboard is None:
        print("pynput missing -> can't jog. pip install pynput")
        close_serial()
        return

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    close_serial()


if __name__ == '__main__':
    main()
