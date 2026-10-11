#!/usr/bin/env python3
r"""
motions.py - hardcoded highlight motions: press a key, the pen draws it.

Nothing is measured at run time.  Two blocks at the top of this file are the
whole program, and they are yours to edit:

  CORNERS  the four points the arm touched on your sheet (pen on the paper),
           copied from your measurement.  Re-measure -> edit here.
  MOTIONS  key -> (label, points): the pen path, written as fractions of the
           sheet, (across, down), 0.0-1.0 - exactly what solve.py's report
           prints in brackets, so you can copy them straight across:

               position: ... -> 5.20 in from the left, 0.65 in down
                           (0.44, 0.08 of the page)
                 \--------------------------------> 0.44, 0.08

The four corners turn those fractions into arm inches with a bilinear blend, so
a sheet lying at an angle to the arm still gets stroked square, and the pen's
height follows the paper.

Run it:

    python motions.py                              # prints the keys, waits
    python motions.py --run 1 --run 9 --sim --fast # fire some from a script

Keys: whatever MOTIONS names (1-9 below), M the middle of the sheet, O / C the
claw, arrows / W / S to nudge the pen, Q quit.
"""
from __future__ import annotations

import argparse
import sys

from pynput import keyboard

import writer          # the arm itself: serial, IK, jogging, motion primitives
from writer_min import MinSheet   # the four-corner sheet map

# ---------------------------------------------------------------- the sheet
# Pen-touching arm positions at the four corners of the sheet, as measured.
CORNERS = {
    "top-left":     (6.60, -2.52, -1.70),
    "top-right":    (6.60,  2.94, -1.70),
    "bottom-left":  (3.66, -2.88, -2.30),
    "bottom-right": (3.66,  2.58, -2.30),
}

# --------------------------------------------------------------- the motions
# One printed character on the page, as a share of the sheet's width: the
# report walks 0.44 -> 0.89 of the page over 20 characters (0.0222 each).
CHAR_FRAC = 0.02
# How far below the reported word the stroke sits (share of the page height).
UNDERLINE_DROP = 0.25


def underline(x, y, word):
    """The stroke under one word: from its reported spot, as wide as the word."""
    width = len(word) * CHAR_FRAC
    return [(x, y + UNDERLINE_DROP), (x + width, y + UNDERLINE_DROP)]


def box(x, y, w, h):
    """A rectangle around a passage: four corners, back to the start."""
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]


# key -> (label, the points the pen walks, pen down between them)
MOTIONS = {
    "1": ('underline "go" -> "went"',      underline(0.38, 0.43, "go")),
    "2": ('underline "buyed" -> "bought"', underline(0.69, 0.08, "buyed")),
    "3": ('underline "apple" -> "apples"', underline(0.13, 0.21, "apple")),
    "4": ('underline "knew" -> "know"',    underline(0.51, 0.21, "knew")),
    "5": ('underline "has" -> "had"',      underline(0.45, 0.33, "has")),
    "6": ('underline "Their" -> "There"',  underline(0.02, 0.46, "Their")),
    "7": ('underline "was" -> "were"',     underline(0.15, 0.46, "was")),
    "8": ('underline "eveent" -> "event"', underline(0.04, 0.72, "eveent")),
    "9": ('box around "Their was"',        box(0.01, 0.43, 0.44, 0.13)),
}

SHEET = None


def build_sheet():
    """The sheet, mapped straight from the hardcoded corners - no report needed."""
    sheet = MinSheet(writer.PAGE_SIZES["a4"], [], "measured corners")
    for name in ("top-left", "top-right", "bottom-left", "bottom-right"):
        sheet.record_corner(name, *CORNERS[name])
    return sheet


def run_motion(key):
    """Fly to a motion's first point, pen down, walk the path, pen up.

    The whole path is checked for reach before the pen goes down at all, so a
    bad point never leaves the arm parked mid-stroke.
    """
    if SHEET is None or not SHEET.calibrated:
        print("[motion] not calibrated")
        return False
    if key not in MOTIONS:
        print(f"[motion] no motion on {key!r} - keys: {', '.join(MOTIONS)}")
        return False
    label, points = MOTIONS[key]
    if not points:
        return False

    def at(frac):
        # sheet fractions -> page inches (the sheet map works in inches)
        return (frac[0] * SHEET.size[0], frac[1] * SHEET.size[1])

    path = [at(p) for p in points]
    for px, py in path:                       # check every point first
        x, y = SHEET.xy(px, py)
        if not writer.reachable(x, y, SHEET.z_travel) or \
                not writer.reachable(x, y, SHEET.z_surface(px, py)):
            print(f"[motion] {key} {label}: page ({px:.2f}, {py:.2f}) in "
                  f"out of reach - not started")
            return False

    print(f"[motion] {key}: {label} - {len(path)} point(s)")
    x, y = SHEET.xy(*path[0])
    writer.move(x, y, SHEET.z_travel, writer.travel_ms(x, y))   # fly over, pen up
    writer.move(x, y, SHEET.z_surface(*path[0]) - writer.PEN_DRAW, 300)  # pen down, press a bit
    writer.delay(0.15)                                          # let the ink land
    for i in range(1, len(path)):
        (ax, ay), (bx, by) = path[i - 1], path[i]
        x0, y0 = SHEET.xy(ax, ay)
        x1, y1 = SHEET.xy(bx, by)
        writer.stroke(x0, y0, x1, y1, SHEET.z_surface(ax, ay) - writer.PEN_DRAW)  # constant speed, pressed
    x, y = SHEET.xy(*path[-1])
    writer.move(x, y, SHEET.z_travel, 300)                      # pen up
    print(f"[motion] {key} done.")
    return True


def show_keys():
    print("motions - press the key:")
    for key, (label, points) in MOTIONS.items():
        print(f"  {key}: {label}  ({len(points)} points)")
    print("  M: tap the middle   O / C: claw open / close   Q: quit")


def on_press(key):
    try:
        name = getattr(key, 'name', '')
        if name in ('up', 'down', 'left', 'right'):
            dx = {'up': 1, 'down': -1}.get(name, 0)
            dy = {'left': 1, 'right': -1}.get(name, 0)
            writer.jog(dx, dy, 0)
        elif name == 'space':
            pass
        else:
            ch = (getattr(key, 'char', None) or '').lower()
            if ch == 'q':
                return False
            if ch == 'w':
                writer.jog(0, 0, 1)
            elif ch == 's':
                writer.jog(0, 0, -1)
            elif ch == 'm':
                SHEET.tap_middle()
            elif ch == 'o':
                writer.set_gripper(writer.GRIPPER_OPEN_PULSE, "open")
            elif ch == 'c':
                writer.set_gripper(writer.GRIPPER_CLOSE_PULSE, "closed")
            elif ch in MOTIONS:
                run_motion(ch)
    except Exception as e:
        print(f"[listener] {e}")


def main():
    global SHEET
    parser = argparse.ArgumentParser(
        description="Hardcoded highlight motions: press a key, the pen draws it.",
        epilog="The points are in CORNERS and MOTIONS at the top of this file.",
    )
    parser.add_argument("--run", action="append", default=[], metavar="KEY",
                        help=f"fire this motion and carry on (keys: "
                             f"{', '.join(MOTIONS)})")
    parser.add_argument("--hover", type=float, metavar="Z",
                        help="arm Z to fly at between strokes (default: the "
                             "measured paper + 0.25 in)")
    parser.add_argument("--sim", action="store_true",
                        help="don't open the serial port - dry run with no arm")
    parser.add_argument("--fast", action="store_true",
                        help="don't wait for the arm between moves")
    args = parser.parse_args()

    writer.FAST = args.fast
    for key in args.run:
        if key not in MOTIONS:
            sys.exit(f"--run {key}: no motion on that key - "
                     f"one of {', '.join(MOTIONS)}")

    SHEET = build_sheet()
    show_keys()

    writer.load_cal()
    if args.sim:
        print("[serial] --sim: not opening the port -> running SIMULATED.")
    else:
        writer.open_serial()
    if args.hover is not None and not SHEET.calibrate_hover(args.hover):
        sys.exit(f"--hover {args.hover} must be above the measured paper "
                 f"(about z {SHEET.z_mark:.2f})")

    for key in args.run:
        run_motion(key)

    if not sys.stdin.isatty():
        print("[no keyboard on this terminal - done)")
        writer.close_serial()
        return
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    writer.close_serial()


if __name__ == '__main__':
    main()
