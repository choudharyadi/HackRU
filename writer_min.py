#!/usr/bin/env python3
"""
writer_min.py - the whole arm in one small file: jog, calibrate, point.

No drawing, no shapes, no flags to wade through.  The arm flies to a spot,
taps it (pen down, pen up), and that is all it ever does.  The mechanics live
in writer.py next door (serial, inverse kinematics, the report, the page map);
this file is only the keys and the order of things.

  1. Jog (arrows + W/S) to a CORNER of the sheet with the pen just touching the
     paper, pick that corner with 1-4, then press P.  It writes the corner down.
     The arm only spans about half a sheet at a time, so pick whichever corner
     is on the side you want to work from.
  2. Lift with W to travel height and press H (skip it and it flies 0.25" up).
  3. Ask for any point: --goto 5.20,0.65 (page inches from the top-left corner),
     or T / V to walk the mistakes and the corners.

  ALL FOUR CORNERS (the usual way in).  Touch each corner of the sheet with the
  pen, say which one it is with 1-4, press P.  With all four recorded the sheet
  is mapped properly - even lying at an angle to the arm - the page size is
  read off the corners themselves (the report is optional for pointing), and
  it flies to the MIDDLE of the paper and taps it: that is the real accuracy
  test.  M taps the middle again, any time.  With just one corner recorded,
  T / N / V / --goto still work off that corner, square to the arm.

Keys:
  Arrows / W / S   jog X / Y / Z, 0.01" per tap
  1 / 2 / 3 / 4    the corner the pen is touching: top-left / top-right /
                   bottom-left / bottom-right (before pressing P)
  P                calibrate: pen touching that corner of the sheet
  H                lock the height you are at as the travel height
  M                fly to the middle of the sheet and tap it
  V                point at the next corner of the sheet (and tap it)
  T                point at every mistake (tap each, in report order)
  N                point at the next mistake only
  O / C            open / close the claw
  Q                quit

    python writer_min.py                              # uses last_report.txt
    python writer_min.py report.txt --ref 12,-4.13,0 --hover 0.4 --tour --sim --fast
    python writer_min.py report.txt --ref 3.77,5.02,-3.31 --corner top-left \\
        --goto 5.20,0.65 --goto 0.25,3.82 --tap --sim --fast
    python writer_min.py --corner-at top-right 6.60,2.94,-1.70 \\
        --corner-at top-left 6.60,-2.52,-1.70 \\
        --corner-at bottom-right 3.66,2.58,-2.30 \\
        --corner-at bottom-left 3.66,-2.88,-2.30 --middle --sim --fast
"""
from __future__ import annotations

import argparse
import os
import sys
from math import sqrt
from pynput import keyboard

import writer  # the arm itself: serial, IK, jogging, the report, the page map

REPORT = 'last_report.txt'
QUAD = ("top-left", "top-right", "bottom-left", "bottom-right")
CORNER_KEYS = {'1': 'top-left', '2': 'top-right', '3': 'bottom-left', '4': 'bottom-right'}
CORNER_KEYS_TEXT = "1/2/3/4 (top-left/top-right/bottom-left/bottom-right)"
SHEET = None


class MinSheet(writer.SheetMap):
    """The page map, with corners you can tap instead of only fly to, and a
    four-corner calibration that maps the sheet even at an angle to the arm."""

    def __init__(self, size, marks, label="a4", ref_corner="top-right"):
        super().__init__(size, marks, label, ref_corner)
        self.recorded = {}       # corner name -> arm (x, y, z), pen touching it

    @property
    def quad(self):
        """True once all four corners are touched: the sheet is fully mapped."""
        return all(c in self.recorded for c in QUAD)

    def record_corner(self, name, x, y, z):
        """Write down the pen-touching point at one corner of the sheet.
        Returns True once the fourth corner completes the mapping."""
        if name not in QUAD:
            raise ValueError(f"unknown corner {name!r} - one of {', '.join(QUAD)}")
        self.recorded[name] = (x, y, z)
        self.ref_corner, self.ref = name, (x, y, z)
        print(f"[quad] {name} corner -> arm X {x:.2f}  Y {y:.2f}  Z {z:.2f} "
              f"recorded ({len(self.recorded)} of 4)")
        if not self.quad:
            left = [c for c in QUAD if c not in self.recorded]
            print(f"still to touch: {', '.join(left)} - press 1-4 for the corner the "
                  f"pen is on, then P")
            return False
        self.size = self.measured_size()
        # The paper's middle is also its middle height: a flat enough sheet.
        self.ref = tuple(sum(self.recorded[c][i] for c in QUAD) / 4.0 for i in range(3))
        w, h = self.size
        print(f"[quad] all four corners in - the sheet is {w:.2f} x {h:.2f} in, mapped "
              f"even at an angle to the arm")
        return True

    def measured_size(self):
        """Sheet size in inches, read off the four measured corners."""
        tl, tr = self.recorded["top-left"], self.recorded["top-right"]
        bl, br = self.recorded["bottom-left"], self.recorded["bottom-right"]

        def span(a, b):
            return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        return ((span(tl, tr) + span(bl, br)) / 2.0,      # across the page
                (span(tl, bl) + span(tr, br)) / 2.0)      # down the page

    def _quad_point(self, px, py):
        """Page inches -> arm (x, y, z), blended between the four measured
        corners (bilinear), so a sheet lying at an angle still maps right."""
        tl, tr = self.recorded["top-left"], self.recorded["top-right"]
        bl, br = self.recorded["bottom-left"], self.recorded["bottom-right"]
        u, v = px / self.size[0], py / self.size[1]
        return tuple((1 - u) * (1 - v) * tl[i] + u * (1 - v) * tr[i]
                     + (1 - u) * v * bl[i] + u * v * br[i] for i in range(3))

    def xy(self, px, py):
        if self.quad:
            x, y, _ = self._quad_point(px, py)
            return (x, y)
        return super().xy(px, py)

    @property
    def z_mark(self):
        """The paper's surface: the four measured corners, or the touched one."""
        if self.quad:
            return sum(self.recorded[c][2] for c in QUAD) / 4.0
        return super().z_mark

    def z_surface(self, px, py):
        """The paper's height at this page point."""
        if self.quad:
            return self._quad_point(px, py)[2]
        return self.z_mark

    def tap_middle(self):
        """Fly to the middle of the sheet and tap it - the accuracy test."""
        w, h = self.size
        return self.goto(w / 2.0, h / 2.0, tap=True)

    def tap_next_corner(self):
        """Point at the next corner of the sheet: fly over, pen down, pen up."""
        if not self.calibrated:
            print("[page] not calibrated - jog to the sheet's top-right corner and press P")
            return False
        name = self.CORNERS[self._corner % len(self.CORNERS)]
        self._corner += 1
        x, y = self.corner_xy(name)
        if not writer.reachable(x, y, self.z_travel) or not writer.reachable(x, y, self.z_mark):
            print(f"[corner] {name} ({x:.2f}, {y:.2f}) out of reach")
            return False
        print(f"[corner] {name} of the sheet -> arm ({x:.2f}, {y:.2f}) - tapping")
        writer.move(x, y, self.z_travel, writer.travel_ms(x, y))   # fly over, pen up
        writer.move(x, y, self.z_mark, 300)                        # pen down
        writer.delay(0.15)                                         # let the ink land
        writer.move(x, y, self.z_travel, 300)                      # pen up
        return True

    def goto(self, px, py, tap=False):
        """Point at any page position asked for: inches from the sheet's top-left
        corner, exactly as the report prints them.  Flies over at travel height;
        with tap=True it also puts the pen down and back up."""
        if not self.calibrated:
            print(f"[page] not calibrated - jog to the sheet's {self.ref_corner} corner and press P")
            return False
        x, y = self.xy(px, py)
        z_down = self.z_surface(px, py)
        if not writer.reachable(x, y, self.z_travel) or not writer.reachable(x, y, z_down):
            print(f"[goto] page ({px:.2f}, {py:.2f}) in -> arm ({x:.2f}, {y:.2f}) out of reach")
            return False
        print(f"[goto] page ({px:.2f}, {py:.2f}) in -> arm ({x:.2f}, {y:.2f})"
              + (" - tapping" if tap else ""))
        writer.move(x, y, self.z_travel, writer.travel_ms(x, y))   # fly over, pen up
        if tap:
            writer.move(x, y, z_down, 300)                         # pen down
            writer.delay(0.15)                                     # let the ink land
            writer.move(x, y, self.z_travel, 300)                  # pen up
        return True


def on_press(key):
    try:
        name = getattr(key, 'name', '')
        if name in ('up', 'down', 'left', 'right'):
            dx = {'up': 1, 'down': -1}.get(name, 0)
            dy = {'left': 1, 'right': -1}.get(name, 0)
            writer.jog(dx, dy, 0)
        elif name == 'space':
            pass                                   # nothing to draw here
        else:
            ch = (getattr(key, 'char', None) or '').lower()
            if ch == 'q':
                return False
            if ch in CORNER_KEYS:
                SHEET.ref_corner = CORNER_KEYS[ch]
                print(f"corner for the next P: {SHEET.ref_corner} - jog the pen onto it, "
                      f"then press P")
            elif ch == 'w':
                writer.jog(0, 0, 1)
            elif ch == 's':
                writer.jog(0, 0, -1)
            elif ch == 'p':
                if SHEET.record_corner(SHEET.ref_corner, writer.curr_x,
                                       writer.curr_y, writer.curr_z):
                    SHEET.tap_middle()          # all four in - tap the middle
            elif ch == 'h':
                SHEET.calibrate_hover(writer.curr_z)
            elif ch == 'm':
                SHEET.tap_middle()
            elif ch == 'v':
                SHEET.tap_next_corner()
            elif ch == 't':
                SHEET.tour()
            elif ch == 'n':
                SHEET.mark_next()
            elif ch == 'o':
                writer.set_gripper(writer.GRIPPER_OPEN_PULSE, "open")
            elif ch == 'c':
                writer.set_gripper(writer.GRIPPER_CLOSE_PULSE, "closed")
    except Exception as e:
        print(f"[listener] {e}")


def main():
    global SHEET
    parser = argparse.ArgumentParser(
        description="Jog, calibrate, point: the arm taps corners and mistakes.",
        epilog="Touch a corner with the pen, press 1-4 for which one, then P. "
               "All four corners map the sheet and it taps the middle. H locks "
               "the travel height; T taps every mistake, V the corners, N one "
               "at a time, M the middle.",
    )
    parser.add_argument("report", nargs="?", default=None,
                        help=f"the 'Found N mistake(s)' listing (default: {REPORT})")
    parser.add_argument("--ref", metavar="X,Y,Z",
                        help="arm position of the corner the pen touches (the sheet's "
                             "top-right by default) - skips jogging to it")
    parser.add_argument("--corner", choices=writer.SheetMap.CORNERS, default="top-right",
                        help="which paper corner the pen is touching; pick the one on "
                             "the side the arm can reach (default top-right)")
    parser.add_argument("--hover", type=float, metavar="Z",
                        help="arm Z to fly at between taps (default: touch + 0.25 in)")
    parser.add_argument("--goto", action="append", default=[], metavar="X,Y",
                        help="page position to fly to - inches from the sheet's top-left "
                             "corner, like the report prints; repeat for several")
    parser.add_argument("--tap", action="store_true",
                        help="with --goto: also put the pen down and back up there")
    parser.add_argument("--corner-at", action="append", default=[], nargs=2,
                        metavar=("NAME", "XYZ"),
                        help="a measured corner of the sheet: NAME is top-left, "
                             "top-right, bottom-left or bottom-right, XYZ its arm "
                             "position with the pen touching the paper; repeat for "
                             "all four and the sheet is fully mapped")
    parser.add_argument("--middle", action="store_true",
                        help="fly to the middle of the sheet and tap it")
    parser.add_argument("--tour", action="store_true",
                        help="tap every mistake, then exit (needs --ref or --corner-at)")
    parser.add_argument("--sim", action="store_true",
                        help="don't open the serial port - dry run with no arm")
    parser.add_argument("--fast", action="store_true",
                        help="don't wait for the arm between moves")
    args = parser.parse_args()

    writer.FAST = args.fast
    path = args.report or REPORT
    if os.path.exists(path):
        with open(path) as f:
            text = f.read()
    elif args.report:
        sys.exit(f"report not found: {path}")
    else:
        text = ""
        print(f"[no {path} - run solve.py first, or pass the report as an argument]")

    marks = writer.parse_report(text)
    size, label = writer.choose_page(marks)
    SHEET = MinSheet(size, marks, label, args.corner)
    print(f"Found {len(marks)} mistake(s) on a {label} page "
          f"({size[0]:.2f} x {size[1]:.2f} in)")
    for m in marks:
        print(f"  {m.number}. {m.original!r} -> {m.correction!r}")

    if args.corner_at and args.ref:
        sys.exit("use --ref for the one corner the pen touches, or --corner-at for "
                 "the measured corners - not both")
    if args.hover is not None and not args.ref and not args.corner_at:
        sys.exit("--hover needs --ref (or --corner-at) too: the hovering point is "
                 "measured from the corner the pen first touched")
    if args.goto and not args.ref and not args.corner_at:
        sys.exit("--goto needs --ref or --corner-at too: the page only maps to arm "
                 "inches once a corner is calibrated")
    if args.tour and not args.ref and not args.corner_at:
        sys.exit("--tour needs --ref X,Y,Z (or --corner-at, or run interactively "
                 "and press T)")

    writer.load_cal()
    if args.sim:
        print("[serial] --sim: not opening the port -> running SIMULATED.")
    else:
        writer.open_serial()

    if args.corner_at:
        for name, corner_xyz in args.corner_at:
            try:
                xyz = writer.parse_ref(corner_xyz)
            except ValueError as exc:
                sys.exit(f"--corner-at {name}: {exc}")
            SHEET.record_corner(name.strip().lower(), *xyz)
        if not SHEET.quad:
            left = [c for c in writer.SheetMap.CORNERS if c not in SHEET.recorded]
            print(f"[quad] only {len(SHEET.recorded)} of 4 corners - still missing "
                  f"{', '.join(left)}; --middle needs all four")
        elif args.hover is not None and not SHEET.calibrate_hover(args.hover):
            sys.exit(f"--hover {args.hover} must be above the paper it measured "
                     f"(about z {SHEET.z_mark:.2f})")

    if args.ref:
        SHEET.calibrate(*writer.parse_ref(args.ref))
        if args.hover is not None and not SHEET.calibrate_hover(args.hover):
            sys.exit(f"--hover {args.hover} must be above the touch point "
                     f"z {SHEET.ref[2]:.2f}")

    for asked in args.goto:
        try:
            px, py = (float(p) for p in asked.replace(" ", ",").split(",") if p)
        except ValueError:
            sys.exit(f"--goto wants two numbers X,Y in page inches (got {asked!r})")
        SHEET.goto(px, py, args.tap)
    if args.middle:
        SHEET.tap_middle()
    if args.tour:
        SHEET.tour()
        writer.close_serial()
        return

    if not args.ref and not args.corner_at:
        print(f"Jog the pen onto a corner of the sheet (arrows + W/S, pen just\n"
              f"touching), press {CORNER_KEYS_TEXT} to say which corner it is,\n"
              f"then press P to record it. All four corners map the sheet and it\n"
              f"taps the middle.")
    if not sys.stdin.isatty():
        print("[no keyboard on this terminal - done)")
        writer.close_serial()
        return
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    writer.close_serial()


if __name__ == '__main__':
    main()
