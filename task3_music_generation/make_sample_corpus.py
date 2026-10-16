"""
Create a small sample MIDI corpus so the project can train without needing
you to download a dataset first.

Real-world use: drop your own .mid files into data/midi/ instead — the rest
of the pipeline works the same. This script only makes starter material.

Run:
    python make_sample_corpus.py
"""

import os
import random

from music21 import chord, instrument, note, stream

import config


SCALES = {
    "C_major": [60, 62, 64, 65, 67, 69, 71, 72],
    "G_major": [67, 69, 71, 72, 74, 76, 78, 79],
    "A_minor": [69, 71, 72, 74, 76, 77, 79, 81],
    "D_major": [62, 64, 66, 67, 69, 71, 73, 74],
    "F_major": [65, 67, 69, 70, 72, 74, 76, 77],
    "E_minor": [64, 66, 67, 69, 71, 72, 74, 76],
}

# Simple diatonic triad shapes (scale degrees) used to sprinkle in chords.
CHORD_SHAPES = [(0, 2, 4), (2, 4, 6), (4, 6, 0), (5, 0, 2)]


def _make_piece(scale_notes, rng, num_notes):
    """Build one short monophonic-ish melody stream."""
    part = stream.Part()
    part.append(instrument.Piano())

    for _ in range(num_notes):
        r = rng.random()
        if r < 0.15:
            # Add a chord every so often.
            shape = rng.choice(CHORD_SHAPES)
            pitches = [scale_notes[(i + rng.randint(0, 1)) % len(scale_notes)]
                       for i in shape]
            part.append(chord.Chord(sorted(pitches), quarterLength=rng.choice([1, 2])))
        else:
            # Mostly stepwise motion with occasional leaps.
            idx = rng.randrange(len(scale_notes))
            if rng.random() < 0.7:
                idx = (idx + rng.choice([-2, -1, 1, 2])) % len(scale_notes)
            pitch = scale_notes[idx] + (12 if rng.random() < 0.15 else 0)
            part.append(note.Note(pitch, quarterLength=rng.choice([0.5, 1, 1])))
    return part


def main():
    config.ensure_dirs()
    rng = random.Random(config.RANDOM_SEED)
    scales = list(SCALES.values())

    for i in range(config.SAMPLE_CORPUS_SIZE):
        scale_notes = rng.choice(scales)
        length = rng.randint(40, 90)
        piece = _make_piece(scale_notes, rng, length)
        out = os.path.join(config.MIDI_DIR, f"sample_{i:03d}.mid")
        piece.write("midi", fp=out)

    print(f"Wrote {config.SAMPLE_CORPUS_SIZE} sample MIDI files to {config.MIDI_DIR}")


if __name__ == "__main__":
    main()
