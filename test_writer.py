"""Check writer.py against the listing solve.py prints.

Runs the real code - parse the report, recover the sheet size, calibrate at the
sheet's top-right corner (pen touching, then pen hovering), tour every mistake -
and asserts the arm is sent to the right inches, flown over and marked down and up
at each one, in report order. Then runs the commands a person would type, end to
end, with no arm attached.

    python test_writer.py
"""
import os
import subprocess
import sys
import tempfile

import writer

# Calibrating taps the page origin as an accuracy test, so the arm moves - no
# point sleeping through it.
writer.FAST = True

# The listing solve.py printed for a photographed page (8 mistakes), position lines
# included - that is the contract between solve.py and writer.py.
SAMPLE = """Found 8 mistake(s):
  1. "go" -> "went"  (grammar: past tense required for past narrative)
      in: "Yesterday, the boy go to the store and buyed"
      position: line 1, char 20 -> 5.20 in from the left, 0.65 in down (0.44, 0.08 of the page)
  2. "buyed" -> "bought"  (grammar: irregular past tense of 'buy' is 'bought')
      in: "the store and buyed some apple"
      position: line 1, char 40 -> 10.40 in from the left, 0.65 in down (0.89, 0.08 of the page)
  3. "apple" -> "apples"  (grammar: plural noun needed after 'some')
      in: "some apple. He didn't knew that the shop was"
      position: line 2, char 6 -> 1.56 in from the left, 1.71 in down (0.13, 0.21 of the page)
  4. "knew" -> "know"  (grammar: base verb form follows auxiliary 'didn't')
      in: "He didn't knew that the shop was"
      position: line 2, char 23 -> 5.98 in from the left, 1.71 in down (0.51, 0.21 of the page)
  5. "has" -> "had"  (grammar: past tense needed for consistency)
      in: "closing early, so he has to wait outside for his"
      position: line 3, char 22 -> 5.25 in from the left, 2.76 in down (0.45, 0.33 of the page)
  6. "Their" -> "There"  (word choice: use 'there' for existence or place)
      in: "Their was many people waiting too, but"
      position: line 4, char 1 -> 0.25 in from the left, 3.82 in down (0.02, 0.46 of the page)
  7. "was" -> "were"  (grammar: plural verb needed to agree with 'many people')
      in: "Their was many people waiting too, but"
      position: line 4, char 7 -> 1.74 in from the left, 3.82 in down (0.15, 0.46 of the page)
  8. "eveent" -> "event"  (spelling: misspelled word)
      in: "eveent for him."
      position: line 6, char 1 -> 0.51 in from the left, 5.92 in down (0.04, 0.72 of the page)
"""

# Where the arm was jogged to for this dry run: the sheet's top-right corner with
# the pen on the paper. Far enough out that every mark on the sheet is reachable.
REF = (12.0, -4.13, 0.0)


def sheet():
    """The sample report, measured as solve.py measured it."""
    marks = writer.parse_report(SAMPLE)
    size, label = writer.choose_page(marks)
    page = writer.SheetMap(size, marks, label)
    page.calibrate(*REF)
    return page


def check_parse():
    marks = writer.parse_report(SAMPLE)
    assert len(marks) == 8, f"expected 8 mistakes, got {len(marks)}"

    first = marks[0]
    assert (first.original, first.correction) == ("go", "went"), first
    assert (first.line, first.char) == (1, 20), (first.line, first.char)
    assert abs(first.x - 5.20) < 1e-9 and abs(first.y - 0.65) < 1e-9, (first.x, first.y)
    assert first.kind.startswith("grammar"), first.kind

    last = marks[-1]
    assert (last.original, last.correction) == ("eveent", "event"), last
    assert (last.line, last.char) == (6, 1), (last.line, last.char)
    assert abs(last.y - 5.92) < 1e-9, last.y
    assert all(m.placed for m in marks), "every position line should parse"

    # A listing with no measurable page must not blow up - the marks just have no home.
    clean = writer.parse_report("No mistakes found - the writing looks clean.")
    assert clean == [], clean
    no_page = writer.parse_report(
        '  1. "teh" -> "the"  (spelling: transposed)\n'
        "      position: unknown (the page could not be measured)\n")
    assert len(no_page) == 1 and not no_page[0].placed, no_page

    print(f"parse: {len(marks)} mistakes, first {first.label} at "
          f"({first.x:.2f}, {first.y:.2f}) in, last {last.label} at "
          f"({last.x:.2f}, {last.y:.2f}) in")


def check_page_size():
    marks = writer.parse_report(SAMPLE)
    size, label = writer.choose_page(marks)
    # The report's own inches and fractions say 11.7 x 8.2 in: A4 on its side.
    assert abs(size[0] - 11.693) < 0.02 and abs(size[1] - 8.268) < 0.02, size
    assert label == "a4 landscape", label
    print(f"page: {label} {size[0]:.3f} x {size[1]:.3f} in (read back out of the report)")

    # An override still comes out in the orientation the report showed.
    forced, forced_label = writer.choose_page(marks, "letter")
    assert forced == (11.0, 8.5) and forced_label == "letter landscape", (forced, forced_label)

    # Nothing to measure -> the default sheet.
    plain, plain_label = writer.choose_page([])
    assert plain == writer.PAGE_SIZES["a4"] and plain_label == "a4", (plain, plain_label)
    print(f"page: --paper letter -> {forced_label}, no fractions -> {plain_label}")


def check_calibration():
    page = sheet()
    w, h = page.size

    # The reference is the sheet's top-right corner, so it maps back to itself.
    assert page.xy(w, 0.0) == (REF[0], REF[1]), page.xy(w, 0.0)
    # Top-left is one page-width back along X; down the page walks along Y.
    assert abs(page.corner_xy("top-left")[0] - (REF[0] - w)) < 1e-9
    assert abs(page.corner_xy("bottom-right")[1] - (REF[1] + h)) < 1e-9

    # "5.20 in from the left, 0.65 in down" -> arm inches, and it is on the sheet.
    x, y = page.xy(5.20, 0.65)
    assert abs(x - (REF[0] + 5.20 - w)) < 1e-9 and abs(y - (REF[1] + 0.65)) < 1e-9, (x, y)
    assert page.z_travel > page.z_mark, "the pen must travel above the sheet and mark on it"
    assert abs(page.z_mark - REF[2]) < 1e-9, "the calibrated corner is the paper surface"
    print(f"calibration: top-right {REF} -> mark z {page.z_mark:.2f}, travel z {page.z_travel:.2f}; "
          f"mistake 1 at arm ({x:.2f}, {y:.2f})")

    # Second point: the pen hovering over that same corner fixes the travel height.
    assert page.hover is None
    assert page.z_travel == REF[2] + writer.PEN_TRAVEL, "no hover point -> PEN_TRAVEL above the paper"
    assert page.calibrate_hover(0.4), "hovering above the paper should be accepted"
    assert page.z_travel == 0.4 and page.z_mark == REF[2], (page.z_travel, page.z_mark)
    assert not page.calibrate_hover(REF[2]), "a hover at or below the paper must be refused"
    assert page.z_travel == 0.4, "a refused hover must leave the travel height alone"
    print(f"calibration: hover z 0.4 -> travels {page.z_travel - page.z_mark:.2f} in above "
          f"the paper it touched; a hover on the paper is refused")

    # Calibrating taps the page origin - the accuracy test at the start.
    sent = []

    def fake_send(x, y, z, t_ms):
        sent.append((x, y, z))
        writer.curr_x, writer.curr_y, writer.curr_z = x, y, z
        return True

    real_send, real_fast = writer.send, writer.FAST
    writer.send, writer.FAST = fake_send, True
    try:
        tapped = writer.SheetMap(page.size, writer.parse_report(SAMPLE), page.label)
        tapped.calibrate(*REF)
    finally:
        writer.send, writer.FAST = real_send, real_fast
    assert len(sent) == 3, f"the accuracy tap should be fly/down/up, got {len(sent)} moves"
    want = tapped.xy(0.0, 0.0)
    assert abs(sent[0][0] - want[0]) < 1e-9 and abs(sent[0][1] - want[1]) < 1e-9, sent
    z_up = REF[2] + writer.PEN_TRAVEL
    assert [p[2] for p in sent] == [z_up, REF[2], z_up], sent
    print(f"calibration: taps the page origin at arm ({want[0]:.2f}, {want[1]:.2f}) - "
          f"pen {sent[0][2]:.2f} -> {sent[1][2]:.2f} -> {sent[2][2]:.2f}")

    # A second point without the first is nonsense.
    bare = writer.SheetMap(page.size, writer.parse_report(SAMPLE), page.label)
    assert not bare.calibrate_hover(0.4), "hovering before touching the corner is refused"

    # A mark with no position is skipped rather than sent somewhere random.
    marks = writer.parse_report('  1. "teh" -> "the"  (spelling: transposed)\n'
                                "      position: unknown (no line/character from Gemini)\n")
    unplaced = writer.SheetMap(page.size, marks, page.label)
    unplaced.calibrate(*REF)
    assert not unplaced.mark(marks[0]), "a mistake with no position must not be marked"
    print("calibration: a mistake with no position is skipped")


def check_tour():
    marks = writer.parse_report(SAMPLE)
    size, label = writer.choose_page(marks)
    page = writer.SheetMap(size, marks, label)
    page.calibrate(*REF)

    sent = []

    def fake_send(x, y, z, t_ms):
        # Stand in for the arm: record where the pen tip was told to go.
        sent.append((x, y, z))
        writer.curr_x, writer.curr_y, writer.curr_z = x, y, z
        return True

    real_send, real_fast, real_mark_len = writer.send, writer.FAST, writer.MARK_LEN
    writer.send, writer.FAST, writer.MARK_LEN = fake_send, True, 0.0
    try:
        assert page.tour() == len(marks), "the whole tour should land"
        # Nothing moves before calibration would have been checked; 3 moves per mark:
        # fly over with the pen up, down, back up.
        assert len(sent) == 3 * len(marks), f"{len(sent)} moves for {len(marks)} mistakes"

        z_up, z_down = page.z_travel, page.z_mark
        for i, m in enumerate(marks):
            fly, down, up = sent[3 * i:3 * i + 3]
            want = page.xy(m.x, m.y)
            assert abs(fly[0] - want[0]) < 1e-9 and abs(fly[1] - want[1]) < 1e-9, \
                f"{m.label} flew to {fly[:2]}, expected {want}"
            assert fly[:2] == down[:2] == up[:2], f"{m.label}: down/up must stay put"
            assert down[2] == z_down, f"{m.label}: should touch the paper at z {z_down:.2f}"
            assert fly[2] == z_up and up[2] == z_up, f"{m.label}: pen up between marks"
            assert down[2] < fly[2], \
                f"{m.label}: expected down {down[2]:.2f} then up {up[2]:.2f}"
            print(f"  {m.label}: arm ({want[0]:.2f}, {want[1]:.2f}), "
                  f"pen {fly[2]:.2f} -> {down[2]:.2f} -> {up[2]:.2f}")

        # A second tour has nothing left to do.
        assert page.tour() == 0, "the tour must not repeat marks"
        assert page.cursor == len(marks), page.cursor

        # --mark-len 0.3: same walk, but each mark underlines 0.3 in instead of dotting.
        writer.MARK_LEN = 0.3
        underlined = writer.SheetMap(size, marks, label)
        underlined.calibrate(*REF)
        sent.clear()
        underlined.tour()
        assert len(sent) == 9 * len(marks), \
            f"fly + down + 6 underline steps + up = 9 moves each, got {len(sent)}"
        for i, m in enumerate(marks):
            seq = sent[9 * i:9 * i + 9]
            want = page.xy(m.x, m.y)
            assert [p[2] for p in seq] == [z_up, z_down] + [z_down] * 6 + [z_up], \
                f"{m.label}: pen must stay down for the whole underline"
            assert abs(seq[-1][0] - (want[0] + writer.ACROSS_SIGN * 0.3)) < 1e-9, \
                f"{m.label}: underline should end 0.3 in along the page"
            print(f"  {m.label}: underlined to arm ({seq[-1][0]:.2f}, {seq[-1][1]:.2f})")

        # The second calibration point (hovering) sets the height it flies at.
        writer.MARK_LEN = 0.0
        hovered = writer.SheetMap(size, marks, label)
        hovered.calibrate(*REF)
        assert hovered.calibrate_hover(0.45), "hovering above the paper should be accepted"
        sent.clear()
        hovered.tour()
        assert len(sent) == 3 * len(marks), f"{len(sent)} moves for {len(marks)} mistakes"
        for i, m in enumerate(marks):
            fly, down, up = sent[3 * i:3 * i + 3]
            assert fly[2] == 0.45 and up[2] == 0.45, \
                f"{m.label}: must fly at the hovering height, got z {fly[2]:.2f}"
            assert down[2] == REF[2], f"{m.label}: must mark on the paper it touched"
        print(f"  hover point 0.45: flew at z 0.45 and marked at z {REF[2]:.2f} for all "
              f"{len(marks)} mistakes")
    finally:
        writer.send, writer.FAST, writer.MARK_LEN = real_send, real_fast, real_mark_len
    print(f"tour: {len(marks)} mistakes, {len(sent)} moves, down then up at each one")


def check_keys():
    """The keys a person presses: P, H, T, N, V, Q through the real listener."""
    import contextlib
    import io
    from types import SimpleNamespace

    page = sheet()
    sent = []

    def fake_send(x, y, z, t_ms):
        sent.append((x, y, z))
        writer.curr_x, writer.curr_y, writer.curr_z = x, y, z
        return True

    real_send, real_fast, real_sheet = writer.send, writer.FAST, writer.SHEET
    writer.send, writer.FAST, writer.SHEET = fake_send, True, page
    log = io.StringIO()
    try:
        # Jog to the corner with the pen on the paper, then lift it to hover height.
        writer.curr_x, writer.curr_y, writer.curr_z = REF
        assert writer.on_press(SimpleNamespace(name=None, char="p")) is not False
        assert page.ref == REF, f"P should record the touch point, got {page.ref}"
        writer.curr_z = 0.4
        assert writer.on_press(SimpleNamespace(name=None, char="h")) is not False
        assert page.hover == 0.4, f"H should record the hovering height, got {page.hover}"

        with contextlib.redirect_stdout(log):
            assert writer.on_press(SimpleNamespace(name=None, char="t")) is not False
            assert writer.on_press(SimpleNamespace(name=None, char="n")) is not False
            assert writer.on_press(SimpleNamespace(name=None, char="v")) is not False
            quit_requested = writer.on_press(SimpleNamespace(name=None, char="q"))
        out = log.getvalue()
        assert "[listener]" not in out, f"the listener swallowed an error: {out}"
        assert "every mistake has been marked" in out, out
        assert quit_requested is False, "Q must stop the listener"

        # P's accuracy tap (3 moves to the page origin), then T toured all 8
        # (3 moves each), N had nothing left, V hopped a corner.
        assert len(sent) == 3 + 3 * len(page.marks) + 1, f"{len(sent)} moves, expected 28"
        origin = page.xy(0.0, 0.0)
        assert abs(sent[0][0] - origin[0]) < 1e-9 and abs(sent[0][1] - origin[1]) < 1e-9, \
            f"P's accuracy tap should fly to the page origin, went to {sent[0][:2]}"
        assert sent[0][2] == REF[2] + writer.PEN_TRAVEL and sent[1][2] == REF[2], sent[:3]
        for i, m in enumerate(page.marks):
            fly, down, up = sent[3 + 3 * i:3 + 3 * i + 3]
            assert fly[2] == 0.4 and up[2] == 0.4 and down[2] == REF[2], \
                f"{m.label}: T must fly at the hover height and mark on the paper"
        corner = sent[-1]
        want = page.corner_xy("top-left")
        assert abs(corner[0] - want[0]) < 1e-9 and abs(corner[1] - want[1]) < 1e-9, \
            f"V should hop to the top-left corner, went to {corner[:2]}"
        print("keys: P touch -> origin tap -> H hover -> T marked 8 at z 0.40/0.00, "
              "N idle, V to the top-left corner, Q quits")
    finally:
        writer.send, writer.FAST, writer.SHEET = real_send, real_fast, real_sheet
        writer.curr_x, writer.curr_y, writer.curr_z = 6.0, 0.0, 4.0


def check_command():
    """The commands a person types, with no arm attached."""
    here = os.path.dirname(os.path.abspath(__file__))
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    handle.write(SAMPLE)
    handle.close()

    def run(*extra):
        try:
            return subprocess.run(
                [sys.executable, "writer.py", handle.name, "--ref", "12,-4.13,0",
                 "--tour", "--sim", "--fast", *extra],
                capture_output=True, text=True, cwd=here)
        finally:
            pass

    try:
        plain = run()
        out = plain.stdout + plain.stderr
        assert plain.returncode == 0, out
        assert "Found 8 mistake(s)" in out, out
        assert "a4 landscape" in out, out
        assert "[tour] marked 8 of 8" in out, out
        assert "SIMULATED" in out, out
        assert "no hovering point" in out, out
        assert "accuracy test" in out, out
        assert "the dot should sit on the sheet's top-left corner" in out, out
        print("command: writer.py --ref ... --tour --sim --fast walked all 8 mistakes "
              "(0.25 in above the paper)")

        hovered = run("--hover", "0.45")
        out = hovered.stdout + hovered.stderr
        assert hovered.returncode == 0, out
        assert "hovering point -> arm Z 0.45" in out, out
        assert "[tour] marked 8 of 8" in out, out
        print("command: --hover 0.45 flew between marks at z 0.45")

        bad = run("--hover", "0.0")
        assert bad.returncode != 0, "a hover below the touched paper must be refused"
        assert "must be above the touch point" in bad.stderr, bad.stderr
        print("command: --hover 0.0 refused - it is not above the touched paper")
    finally:
        os.unlink(handle.name)


def main():
    check_parse()
    check_page_size()
    check_calibration()
    check_tour()
    check_keys()
    check_command()
    print("\nall writer checks passed")


if __name__ == "__main__":
    sys.exit(main())
