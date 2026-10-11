# Redline

**The robot that marks your paper.**

Hold a page of writing under the camera and say **"hey Redline, check this text."** Redline
photographs the page, has Gemini proofread it, reads every spelling, grammar and punctuation
mistake out loud, and then works out *where* each one sits on the paper - in inches from the
top-left corner - so the arm can go and point at it.

```
==========================================================
  REDLINE  -  the robot that marks your paper
  photo -> Gemini -> every mistake marked, to the inch
==========================================================
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env needs GEMINI_API_KEY=... and ELEVENLABS_API_KEY=... (one per line, no quotes or spaces)

python solve.py                  # hands-free: listen for its name
python solve.py --manual         # no microphone: press Enter to take the photo
python solve.py --image page.jpg # check a photo you already have
python solve.py --preview        # line the page up and watch the focus score (s = save, q = quit)
python solve.py --no-speak       # proofread silently
```

## Talking to it

| you say | Redline does |
| --- | --- |
| "hey Redline, check this text" | photographs the page and marks it up |
| "Redline" - just the name | the same; the name alone is a command |
| "proofread this", "check my grammar", "find mistakes", "mark this up" | the same |
| "goodbye" or Ctrl+C | signs off |

Speech-to-text spells the name loosely, so "Red Line!", "red-line" and "read line" all count.
It greets you before it opens the microphone, so it never hears its own voice.

## How it knows *where* the mistake is

A sheet of paper is a predictable thing - margins, line spacing, character pitch - so
[layout.py](layout.py) flattens the sheet found in the photo to a straight top-down view,
measures the real lines of ink, and turns Gemini's "line 3, character 10" into an exact
`(x, y)` in inches from the corner of the page. Every mistake is printed with that position,
and the marked page is written to `last_marked.jpg`.

## Tests

```bash
python make_test_page.py && python test_layout.py   # page geometry, from a generated photo of a page
python test_voice.py                                # every spoken line, fake microphone, no API keys
```
