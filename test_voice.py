"""Check Redline's voice behaviour with no microphone, camera or API keys.

    python test_voice.py

The microphone is a fake that plays a scripted score, and the Gemini call is stubbed, so
this runs anywhere and costs nothing. It checks the things you say out loud on stage:

* Redline greets you by name before the mic opens, so it never hears its own voice;
* hearing the name alone - "Red line!" - triggers a check, with no fixed phrase to remember;
* the check narrates while it works, then says where the marks are;
* "goodbye" gets a sign-off and returns from the loop.

The audio is synthetic, so this proves which lines are spoken and in what order, not that a
real microphone works.
"""
import sys
import types
from pathlib import Path

import numpy as np

import solve

BLOCK = solve.BLOCK
LOUD, QUIET = 30000, 0
SPEECH_BLOCKS, SILENCE_BLOCKS = 15, 40  # long enough to count as speech, then a pause long enough to end it

FAKE_REVIEW = solve.Review(
    text="i recieve teh letter.\nIt was nice.",
    mistakes=[
        solve.Mistake(
            original="i",
            correction="I",
            type="capitalization",
            explanation="capitalize the first-person pronoun",
            context="i recieve teh letter.",
            line=1,
            char=1,
        )
    ],
    corrected="I receive the letter.\nIt was nice.",
    spoken="I found one mistake: 'i' should be 'I'.",
)


class FakeStream:
    """A microphone playing a script: (level, how many blocks) pairs, one after another."""

    read_available = 0

    def __init__(self, score):
        self.score = [[level, count] for level, count in score]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frames):
        level = self.score[0][0] if self.score else QUIET
        if self.score:
            self.score[0][1] -= 1
            if self.score[0][1] <= 0:
                self.score.pop(0)
        return np.full((frames, 1), level, dtype=np.int16), None


def install_fake_mic(score):
    """Make `import sounddevice as sd` inside solve.py hand back our scripted microphone."""
    module = types.ModuleType("sounddevice")
    module.InputStream = lambda **kwargs: FakeStream(score)
    sys.modules["sounddevice"] = module


class RecordingSpeaker(solve.Speaker):
    """The real narration code with no client: it keeps every line, and never opens the network."""

    said = []

    def __init__(self, api_key=None, enabled=True):
        super().__init__(None, enabled=False)

    def say(self, text):
        text = (text or "").strip()
        if text:
            RecordingSpeaker.said.append(text)
        super().say(text)


def fake_take_photo(camera_index, **kwargs):
    """A blank sheet instead of the webcam."""
    import cv2

    blank = np.full((900, 700, 3), 255, np.uint8)
    return cv2.imencode(".jpg", blank)[1].tobytes()


def main():
    RecordingSpeaker.said = []
    said = RecordingSpeaker.said
    install_fake_mic([
        (QUIET, 50),                                    # the quiet second spent setting the threshold
        (LOUD, SPEECH_BLOCKS), (QUIET, SILENCE_BLOCKS),  # "red line"
        (LOUD, SPEECH_BLOCKS), (QUIET, SILENCE_BLOCKS),  # "goodbye"
    ])

    heard = ["red line", "goodbye"]  # what speech-to-text hands back, in order
    checks = []
    marks_before = Path("last_marked.jpg").stat().st_mtime if Path("last_marked.jpg").exists() else 0

    def fake_transcribe(eleven, audio):
        return heard.pop(0) if heard else ""

    real_run_check = solve.run_check

    def spy_run_check(*args, **kwargs):
        checks.append(args[4] if len(args) > 4 else kwargs.get("speaker"))
        return real_run_check(*args, **kwargs)

    solve.Speaker = RecordingSpeaker
    solve.transcribe = fake_transcribe
    solve.take_photo = fake_take_photo
    solve.check_text = lambda *a, **k: FAKE_REVIEW  # no Gemini call
    solve.run_check = spy_run_check

    solve.voice_loop("dummy-gemini-key", "dummy-eleven-key", 0)

    # What should have been said, in this order: hello, then the narration of one check, then goodbye.
    expected = [
        ("greeting by name", lambda line: line in solve.NARRATE_HELLO),
        ("as it takes the photo", lambda line: line in solve.NARRATE_LOOK),
        ("while Gemini reads", lambda line: line in solve.NARRATE_SCAN),
        ("the mistakes themselves", lambda line: line == FAKE_REVIEW.spoken),
        ("where the marks are", lambda line: line in solve.NARRATE_LOCATED),
        ("signing off", lambda line: line in solve.NARRATE_BYE),
    ]

    failures = []
    print("\n-- what Redline said --")
    for i, (what, matches) in enumerate(expected):
        line = said[i] if i < len(said) else "(nothing)"
        ok = i < len(said) and matches(line)
        print(f"  {'ok ' if ok else 'BAD'} {what:24} {line!r}")
        if not ok:
            failures.append(f"{what}: said {line!r}")

    print("\n-- the run --")
    if len(said) != len(expected):
        failures.append(f"spoke {len(said)} line(s), expected {len(expected)}: {said}")
    print(f"  {'ok ' if len(said) == len(expected) else 'BAD'} lines spoken: {len(said)} (expected {len(expected)})")

    print(f"  {'ok ' if len(checks) == 1 else 'BAD'} hearing just 'red line' triggered {len(checks)} check(s) (expected 1)")
    if len(checks) != 1:
        failures.append(f"the wake word triggered {len(checks)} checks")

    marks = Path("last_marked.jpg")
    marked = marks.exists() and marks.stat().st_mtime > marks_before
    print(f"  {'ok ' if marked else 'BAD'} the check marked the page -> {marks}")
    if not marked:
        failures.append("no marked page written from the voice path")

    print("\nfailures:", len(failures))
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
