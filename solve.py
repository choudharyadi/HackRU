"""Redline - the robot that marks your paper.

Say "hey Redline check this text": the camera photographs the writing and Gemini finds the mistakes.
Hold a page, whiteboard, or screen under the camera, and Redline reads it, lists every
spelling/grammar/punctuation mistake with a correction, and says them out loud. Say its name
on its own ("Redline") and it will take that as "take a look at this".

Setup:
    pip install -r requirements.txt
    .env needs GEMINI_API_KEY and ELEVENLABS_API_KEY (one per line, no spaces or quotes)

Run:
    python solve.py                # hands-free: listen, then say "hey Redline check this text"
    python solve.py --manual       # no voice: press Enter to take the photo
    python solve.py --image p.jpg  # skip the camera and check a photo you already have
    python solve.py --preview      # live camera window to line up the writing (s = save, q = quit)
    python solve.py --settle 3 --burst 3   # give autofocus longer / sample more frames
    python solve.py --no-speak     # find mistakes silently
    python solve.py --paper letter     # say the sheet is Letter, not A4 (default: guess)

Redline speaks the result (ElevenLabs), e.g. "I found three mistakes: 'recieve' should be
'receive', and two comma fixes." It also mutters while it works ("Hmm, reading this...").
Speech runs on a background thread, so the talking never holds up the next photo.

Where is the mistake? A page has a consistent structure - margins, line spacing, character
pitch - so [layout.py](layout.py) flattens the sheet found in the photo, measures the real
lines of ink, and turns "line 3, character 10" into exact inches from the top-left corner of
the page. Every mistake is printed with its position, and the marked page is saved to
last_marked.jpg. That is the map the arm needs to go and point at the mistake.

In voice mode, say "hey Redline check this text" (or "proofread this", "check my grammar") to
trigger a photo. Just saying the name works too. Say "goodbye" or press Ctrl+C to quit.
"""
import argparse
import io
import os
import queue
import random
import re
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import layout as pagemap

# Redline's identity. Everything it says out loud is in character.
NAME = "Redline"
TAGLINE = "the robot that marks your paper"
BANNER = "\n".join([
    "=" * 58,
    f"  {NAME.upper()}  -  {TAGLINE}",
    "  photo -> Gemini -> every mistake marked, to the inch",
    "=" * 58,
])

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# If the preferred model is busy (503) or out of quota (429), fall back to these in order.
MODEL_FALLBACKS = ("gemini-3.7-flash", "gemini-flash-latest", "gemini-2.5-flash")
STT_MODEL = os.getenv("ELEVEN_STT_MODEL", "scribe_v2")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
TTS_MODEL = os.getenv("ELEVEN_TTS_MODEL", "eleven_flash_v2_5")
TTS_SAMPLE_RATE = 24000  # matches output_format="pcm_24000"

# What Redline mutters while it works, so the wait doesn't feel dead.
NARRATE_HELLO = (
    f"{NAME} ready. Hold your writing under the camera and say, hey {NAME}, check this text.",
    f"Hey, {NAME} here. Put a page in front of me and say, hey {NAME}, check this text.",
)
NARRATE_LOOK = (f"{NAME} here - let me take a look.", "Okay, let me have a look.", "Hold on, taking a photo.")
NARRATE_SCAN = ("Hmm, reading this...", "Okay, scanning for mistakes...", "Let me proofread this...")
NARRATE_LOCATED = (
    "I know where each one is on the page.",
    "I've worked out where they sit on the page.",
    f"{NAME} has the marks ready.",
)
NARRATE_BYE = (
    f"{NAME} signing off. Happy writing!",
    f"Okay, that's enough marking for now. {NAME} out.",
)

SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK = SAMPLE_RATE * BLOCK_MS // 1000

PROMPT = """The photo shows a piece of writing: handwriting or print on paper, a whiteboard, or a screen.
It may be prose, an essay, notes, a message, a caption, or a list.

Transcribe it, then check it like a careful proofreader. Report:
- text: the writing exactly as it appears, keeping the original line breaks.
- mistakes: every real error you can justify, each with
    - original: the exact word or phrase that is wrong, copied character for character;
    - correction: what it should say;
    - type: spelling, grammar, punctuation, capitalization, or word choice;
    - explanation: one short phrase saying why, e.g. "i before e here";
    - context: the sentence or fragment it sits in, so it can be found on the page.
    - line: which line of `text` it is on, counting the first line as 1;
    - char: which character of that line it starts at, counting the first character as 1.
  List them in the order they appear. Do not invent errors, and do not "fix" deliberate style.
  The line and char are used to point at the mistake on the page, so count them carefully
  against your own transcription.
- corrected: the full writing with every mistake fixed and nothing else changed.
- spoken: one short, friendly sentence Redline (a desktop robot proofreader) would say out loud,
  naming the most important mistakes, e.g. "I found three mistakes: 'recieve' should be 'receive',
  and two comma fixes."
  If the writing is clean, say it looks good instead. Never mention line numbers or coordinates.

Only report a mistake you are confident about; ignore matters of taste unless they are clear errors.
If part of the writing is blurry, cut off, or unreadable, say which part in `spoken` instead of
guessing at it, and leave it out of `text`."""

CHECK_PHRASES = (
    "check this text",
    "check my text",
    "check the text",
    "check this writing",
    "check my writing",
    "check the writing",
    "check my grammar",
    "check the grammar",
    "proofread",
    "proof read",
    "find mistakes",
    "find the mistakes",
    "spot mistakes",
    "check for mistakes",
    "mistakes",
    "grammar",
    "check this",
    "check this page",
    "check this paper",
    "check the page",
    "mark this",
    "mark my paper",
    "mark this up",
    "mark it up",
)
QUIT_PHRASES = ("quit", "exit", "goodbye", "stop listening")
# Hearing its own name is enough. Speech-to-text spells "Redline" loosely, so compare the
# letters only: "Redline", "Red Line!", "red-line" and "read line" all count.
NAME_SPELLINGS = ("redline", "redlion", "readline")


# ------------------------------------------------------------- review shape
class Mistake(BaseModel):
    """One error the proofreader found, and where on the page it sits."""

    original: str
    correction: str
    type: str
    explanation: str
    context: str
    line: int = 0  # 1-based line of the writing, 0 if Gemini did not say
    char: int = 0  # 1-based character where the mistake starts, 0 if it did not say


class Review(BaseModel):
    """Gemini's structured answer, so we can list and speak the mistakes."""

    text: str = ""
    mistakes: list[Mistake] = Field(default_factory=list)
    corrected: str = ""
    spoken: str = ""


# ------------------------------------------------------------------ camera
def focus_score(frame):
    """How sharp a frame is: variance of the Laplacian. Higher = sharper, low = blurry."""
    import cv2

    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def take_photo(camera_index, save_path="last_capture.jpg", settle_s=1.5, burst_s=1.5):
    """Grab the sharpest frame from a short burst, after the camera has time to focus.

    Webcams autofocus slowly, so the very first frame after the camera opens is
    usually blurry. We let focus and exposure settle, then keep the sharpest
    frame of a burst instead of trusting a single grab.
    """
    import cv2  # pip install opencv-python

    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}. Try --camera 1.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # keep autofocus on while we settle

        end = time.time() + settle_s  # let auto-exposure and autofocus settle
        while time.time() < end:
            cap.read()

        best, best_score = None, -1.0
        end = time.time() + burst_s
        while time.time() < end:  # keep the sharpest frame of the burst
            ok, frame = cap.read()
            if not ok:
                continue
            score = focus_score(frame)
            if score > best_score:
                best, best_score = frame, score
    finally:
        cap.release()

    if best is None:
        raise RuntimeError("The camera did not return a picture.")
    if best_score < 25:
        print("[the photo still looks soft - hold steady, move a bit closer, and light the page]")
    ok, buf = cv2.imencode(".jpg", best, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode the picture.")
    Path(save_path).write_bytes(buf.tobytes())
    print(f"Saved the photo to {save_path} (focus {best_score:.0f})")
    return buf.tobytes()


def preview(camera_index):
    """Live window to aim the camera and check focus. Press s to save test.jpg, q to quit."""
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera_index}. Try --camera 1.")
    print("Aim at the writing. The focus score is shown top-left - make it as HIGH as you can.")
    print("A webcam cannot focus closer than ~30 cm; if it stays low, move the page back.")
    print("Press s to save test.jpg, q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = focus_score(frame)
        cv2.putText(frame, f"focus {score:.0f}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("camera preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite("test.jpg", frame)
            print(f"Saved test.jpg (focus {score:.0f})")
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def load_image(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"Image not found: {path}")
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return p.read_bytes(), mime


# ------------------------------------------------------------------ gemini
def parse_review(response):
    """Turn Gemini's answer into a Review, tolerating a plain-text reply."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, Review):
        return parsed
    if isinstance(parsed, dict):
        return Review.model_validate(parsed)
    text = (response.text or "").strip()
    try:
        return Review.model_validate_json(text)
    except Exception:
        return Review(spoken=text)  # last resort: read the raw answer aloud


def position_text(mistake, fitted):
    """Where the mistake is, in inches from the top-left of the page."""
    if fitted is None:
        return "position: unknown (the page could not be measured)"
    if not mistake.line or not mistake.char:
        return "position: unknown (no line/character from Gemini)"
    x, y = fitted.position(mistake.line, mistake.char)
    width, height = fitted.size
    return (
        f"position: line {mistake.line}, char {mistake.char} -> "
        f"{x:.2f} in from the left, {y:.2f} in down "
        f"({x / width:.2f}, {y / height:.2f} of the page)"
    )


def format_review(review, fitted=None):
    """Human-readable listing of the mistakes found, with their places on the page."""
    if not review.mistakes:
        return "No mistakes found - the writing looks clean."
    lines = [f"Found {len(review.mistakes)} mistake(s):"]
    for i, m in enumerate(review.mistakes, 1):
        lines.append(f'  {i}. "{m.original}" -> "{m.correction}"  ({m.type}: {m.explanation})')
        if m.context:
            lines.append(f'      in: "{m.context}"')
        lines.append(f"      {position_text(m, fitted)}")
    if review.corrected:
        lines.append("\nCorrected:\n" + review.corrected)
    return "\n".join(lines)


# --------------------------------------------------------------- the page map
def locate_mistakes(image_bytes, review, dpi=pagemap.DEFAULT_DPI, paper=None):
    """Work out the page layout so each mistake has an (x, y) in inches.

    Returns (sheet, fitted) or (None, None) if the picture cannot be measured.
    """
    try:
        sheet = pagemap.scan_page(pagemap.decode(image_bytes), dpi=dpi, paper=paper)
    except Exception as exc:
        print(f"[could not measure the page: {exc}]")
        return None, None
    lines = review.text.splitlines() or [""]
    fitted = pagemap.fit_layout(sheet, [len(line) for line in lines])
    where = "found the sheet in the photo" if sheet.detected else "no sheet in view, using the whole picture"
    print(f"[page] {where}: {sheet.paper} {sheet.size[0]:.2f} x {sheet.size[1]:.2f} in at {sheet.px_per_in:.0f} px/in")
    print(f"[layout] {fitted.describe()}")
    for note in fitted.notes:
        print(f"[layout] {note}")
    return sheet, fitted


def mark_page(sheet, review, fitted, save_path="last_marked.jpg"):
    """Draw a numbered box on each mistake, at the inches the model computed."""
    marks = [
        (m.line, m.char, len(m.original or m.correction or "?"), m.original or "?")
        for m in review.mistakes
        if m.line and m.char and fitted
    ]
    if not marks:
        return None
    import cv2

    marked = pagemap.draw_marks(sheet, fitted, marks)
    cv2.imwrite(save_path, marked)
    print(f"Marked where they are on the page -> {save_path}")
    return marked


def check_text(image_bytes, mime_type, api_key):
    """Ask Gemini to proofread the photo. Falls back to another model if one is busy."""
    client = genai.Client(api_key=api_key)
    contents = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type), PROMPT]
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=Review
    )

    models = [MODEL] + [m for m in MODEL_FALLBACKS if m != MODEL]
    last_error = None
    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                if model != MODEL:
                    print(f"[used {model} - {MODEL} was busy]")
                return parse_review(response)
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    break  # this model's quota is used up; go straight to the next one
                if "503" in message or "UNAVAILABLE" in message:
                    time.sleep(1.5 * (attempt + 1))  # busy model: retry once, then move on
                    continue
                raise
    raise last_error


# ------------------------------------------------------------------- voice
def parse_command(text):
    """Turn what was heard into 'check', 'quit', or None.

    Saying Redline's name is a command in itself: "Redline, have a look at this." beats
    remembering an exact phrase at demo time.
    """
    t = text.lower()
    if any(p in t for p in QUIT_PHRASES):
        return "quit"
    letters_only = re.sub(r"[^a-z]", "", t)  # "Red Line!" -> "redline"
    if any(spelling in letters_only for spelling in NAME_SPELLINGS):
        return "check"
    if any(w in t for w in CHECK_PHRASES):
        return "check"
    return None


def rms(chunk):
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


class UtteranceDetector:
    """Feed it 30 ms blocks of audio; it returns the audio of a full spoken phrase once you stop talking."""

    def __init__(self, threshold, silence_s=0.9, max_s=8.0, min_s=0.4):
        self.threshold = threshold
        self.silence_blocks = int(silence_s * 1000 / BLOCK_MS)
        self.max_blocks = int(max_s * 1000 / BLOCK_MS)
        self.min_blocks = int(min_s * 1000 / BLOCK_MS)
        self.pre = []  # ~0.3 s of audio before speech starts, so the first word isn't clipped
        self.reset()

    def reset(self):
        self.pre = []
        self.voiced = []
        self.speaking = False
        self.loud = 0
        self.quiet = 0

    def feed(self, chunk):
        loud = rms(chunk) > self.threshold
        if not self.speaking:
            self.pre.append(chunk)
            self.pre = self.pre[-10:]
            self.loud = self.loud + 1 if loud else 0
            if self.loud >= 3:  # three loud blocks in a row = someone is talking
                self.speaking = True
                self.voiced = list(self.pre)
                self.quiet = 0
            return None

        self.voiced.append(chunk)
        self.quiet = 0 if loud else self.quiet + 1
        if self.quiet >= self.silence_blocks or len(self.voiced) >= self.max_blocks:
            spoken_blocks = len(self.voiced) - self.quiet
            audio = np.concatenate(self.voiced)
            self.reset()
            return audio if spoken_blocks >= self.min_blocks else None
        return None


def transcribe(eleven, audio):
    """Send a spoken phrase to ElevenLabs speech-to-text and return the text."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.astype(np.int16).tobytes())
    last_error = None
    for model in (STT_MODEL, "scribe_v1"):  # fall back to the older model if needed
        try:
            buf.seek(0)
            buf.name = "speech.wav"
            return eleven.speech_to_text.convert(file=buf, model_id=model).text.strip()
        except Exception as exc:
            last_error = exc
    raise last_error


class Speaker:
    """Speaks with ElevenLabs on a background thread, so talking never blocks the camera.

    say() queues a line and returns right away; one worker plays queued lines in order.
    wait() blocks until everything queued has finished, so a one-shot run doesn't exit
    before its audio plays. Without a key it just prints the line.
    """

    def __init__(self, api_key=None, enabled=True):
        self.client = None
        self.queue = queue.Queue()
        if api_key and enabled:
            from elevenlabs.client import ElevenLabs

            self.client = ElevenLabs(api_key=api_key)
            threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text):
        """Queue a line to be spoken. Returns immediately."""
        text = (text or "").strip()
        if not text:
            return
        print(f"[{NAME} says] {text}")
        if self.client:
            self.queue.put(text)

    def wait(self):
        """Block until every queued line has been spoken."""
        self.queue.join()

    def _worker(self):
        while True:
            text = self.queue.get()
            try:
                self._play(text)
            except Exception as exc:  # audio problems must never crash the demo
                print(f"[could not play the audio: {exc}]")
            finally:
                self.queue.task_done()

    def _play(self, text):
        import sounddevice as sd

        audio = self.client.text_to_speech.stream(
            voice_id=VOICE_ID,
            text=text,
            model_id=TTS_MODEL,
            output_format="pcm_24000",  # raw 16-bit mono PCM, plays with no extra software
        )
        leftover = b""
        with sd.RawOutputStream(
            samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16"
        ) as out:
            for chunk in audio:
                if not chunk:
                    continue
                data = leftover + chunk
                usable = len(data) - (len(data) % 2)  # keep whole 16-bit samples
                out.write(data[:usable])
                leftover = data[usable:]


def voice_loop(gemini_key, eleven_key, camera_index, settle_s=1.5, burst_s=1.5, speak=True,
               dpi=pagemap.DEFAULT_DPI, paper=None):
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        sys.exit(f"Could not load the microphone library: {exc}")
    from elevenlabs.client import ElevenLabs

    eleven = ElevenLabs(api_key=eleven_key)
    speaker = Speaker(eleven_key, enabled=speak)
    if speak:  # say hello before the mic opens, so Redline doesn't hear its own greeting
        speaker.say(random.choice(NARRATE_HELLO))
        speaker.wait()
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=BLOCK) as stream:
            print("Calibrating the microphone. Stay quiet for a second...")
            levels = [rms(stream.read(BLOCK)[0][:, 0]) for _ in range(33)]
            detector = UtteranceDetector(threshold=max(400.0, 3 * float(np.mean(levels))))
            print(f'Listening. Say "hey {NAME} check this text" when the writing is under the camera.')
            print(f'(Just "{NAME}" works too. Say "goodbye" or press Ctrl+C to quit.)\n')

            while True:
                audio = detector.feed(stream.read(BLOCK)[0][:, 0])
                if audio is None:
                    continue
                try:
                    text = transcribe(eleven, audio)
                except Exception as exc:
                    print(f"[could not understand the audio: {exc}]")
                    continue
                if not text:
                    continue
                print(f'Heard: "{text}"')

                command = parse_command(text)
                if command == "quit":
                    speaker.say(random.choice(NARRATE_BYE))
                    speaker.wait()
                    print("Bye!")
                    return
                if command == "check":
                    run_check(gemini_key, camera_index, settle_s, burst_s, speaker, dpi, paper)
                    speaker.wait()  # finish talking first, so the robot can't hear itself
                    # drop audio that piled up while the camera and Gemini were busy
                    if stream.read_available:
                        stream.read(stream.read_available)
                    detector.reset()
                    print(f'\nListening again. Say "hey {NAME} check this text" for another page.\n')
    except KeyboardInterrupt:
        speaker.say(random.choice(NARRATE_BYE))
        speaker.wait()
        print("\nBye!")


def run_check(gemini_key, camera_index, settle_s=1.5, burst_s=1.5, speaker=None,
              dpi=pagemap.DEFAULT_DPI, paper=None):
    if speaker:
        speaker.say(random.choice(NARRATE_LOOK))  # chatter while the camera settles
    print("Taking a photo of the text...")
    try:
        image = take_photo(camera_index, settle_s=settle_s, burst_s=burst_s)
    except Exception as exc:
        print(f"[camera problem: {exc}]")
        return None
    return report(image, "image/jpeg", gemini_key, speaker, dpi=dpi, paper=paper)


def report(image_bytes, mime_type, gemini_key, speaker=None, dpi=pagemap.DEFAULT_DPI, paper=None):
    """Check a photo, then say where on the page each mistake is.

    Shared by the voice loop, --manual and --image so all three behave the same.
    Returns the Review, or None if the request failed.
    """
    if speaker:
        speaker.say(random.choice(NARRATE_SCAN))  # covers the Gemini round-trip
    print("Reading the text with Gemini...\n")
    try:
        review = check_text(image_bytes, mime_type, gemini_key)
    except Exception as exc:
        print(f"[Gemini request failed: {exc}]")
        return None
    sheet, fitted = locate_mistakes(image_bytes, review, dpi=dpi, paper=paper)
    print(format_review(review, fitted))
    if sheet is not None:
        mark_page(sheet, review, fitted)
    if speaker:
        speaker.say(review.spoken)
        if review.mistakes:
            speaker.say(random.choice(NARRATE_LOCATED))
    return review


# -------------------------------------------------------------------- main
def main():
    load_dotenv()
    print(BANNER)
    parser = argparse.ArgumentParser(
        description=f"{NAME} - {TAGLINE}: a photo of some writing -> Gemini finds the mistakes"
    )
    parser.add_argument("--image", help="use this photo instead of the webcam")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--preview", action="store_true", help="show a live camera window and exit")
    parser.add_argument("--manual", action="store_true", help="press Enter instead of using your voice")
    parser.add_argument("--settle", type=float, default=1.5, help="seconds to let the camera focus before capturing")
    parser.add_argument("--burst", type=float, default=1.5, help="seconds of frames to sample for the sharpest shot")
    parser.add_argument("--no-speak", action="store_true", help="don't read the mistakes out loud")
    parser.add_argument("--dpi", type=float, default=pagemap.DEFAULT_DPI,
                        help="pixels per inch used when flattening the page (default %(default)s)")
    parser.add_argument("--paper", choices=("auto", "a4", "letter"), default="auto",
                        help="paper size to assume; 'auto' picks the nearest to the photo")
    args = parser.parse_args()
    paper = None if args.paper == "auto" else args.paper

    if args.preview:
        preview(args.camera)
        return

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        sys.exit("Set GEMINI_API_KEY in your .env file (see .env.example).")
    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    speaker = Speaker(eleven_key, enabled=not args.no_speak)

    if args.image:
        image, mime = load_image(args.image)
        review = report(image, mime, gemini_key, speaker, dpi=args.dpi, paper=paper)
        speaker.wait()  # one-shot run: don't exit before the audio has played
        if review is None:
            sys.exit("[Gemini request failed]")
        return

    if args.manual:
        input("Put the writing under the camera and press Enter... ")
        run_check(gemini_key, args.camera, args.settle, args.burst, speaker, args.dpi, paper)
        speaker.wait()  # one-shot run: don't exit before the audio has played
        return

    if not eleven_key:
        sys.exit("Voice mode needs ELEVENLABS_API_KEY in your .env file (or run with --manual).")
    voice_loop(
        gemini_key, eleven_key, args.camera, args.settle, args.burst,
        speak=not args.no_speak, dpi=args.dpi, paper=paper,
    )


if __name__ == "__main__":
    main()
