"""
Data collection and preprocessing.

- Collect MIDI files (yours, or the generated sample corpus).
- Parse them with music21 into a flat sequence of notes and chords.
- Build a vocabulary and turn the sequence into (input, output) training
  windows, then save everything to a pickle for the training step.
"""

import glob
import os
import pickle

from music21 import chord, converter, note

import config


def collect_midi_files(directory=None):
    directory = directory or config.MIDI_DIR
    files = sorted(glob.glob(os.path.join(directory, "*.mid")) +
                   glob.glob(os.path.join(directory, "*.midi")))
    return files


def extract_notes(files):
    """Return a flat list of tokens like 'C4' or 'C4.E4.G4' for chords."""
    notes = []
    for path in files:
        try:
            midi = converter.parse(path)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue

        parts = midi.parts if midi.parts else [midi]
        elements = []
        for part in parts:
            try:
                elements.extend(part.recurse().notes)
            except Exception:
                continue

        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(p) for p in element.pitches))
    return notes


def build_vocab(notes):
    pitches = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitches)}
    return pitches, note_to_int


def make_sequences(notes, note_to_int, seq_len=None, step=None):
    seq_len = seq_len or config.SEQUENCE_LENGTH
    step = step or config.STEP
    inputs, outputs = [], []
    for i in range(0, len(notes) - seq_len, step):
        seq_in = notes[i:i + seq_len]
        seq_out = notes[i + seq_len]
        inputs.append([note_to_int[n] for n in seq_in])
        outputs.append(note_to_int[seq_out])
    return inputs, outputs


def prepare(save=True):
    """Full pipeline: collect -> parse -> sequence -> save pickle."""
    config.ensure_dirs()
    files = collect_midi_files()
    if not files:
        raise SystemExit(
            "No MIDI files found in data/midi/. Run 'python make_sample_corpus.py' "
            "or add your own .mid files."
        )

    print(f"Parsing {len(files)} MIDI files ...")
    notes = extract_notes(files)
    if len(notes) <= config.SEQUENCE_LENGTH:
        raise SystemExit("Not enough notes to build sequences. Add more MIDI files.")

    pitches, note_to_int = build_vocab(notes)
    network_input, network_output = make_sequences(notes, note_to_int)

    data = {
        "notes": notes,
        "pitches": pitches,
        "note_to_int": note_to_int,
        "network_input": network_input,
        "network_output": network_output,
        "n_vocab": len(pitches),
        "sequence_length": config.SEQUENCE_LENGTH,
    }

    print(f"Total notes: {len(notes)}   Vocabulary: {len(pitches)}   "
          f"Training windows: {len(network_input)}")

    if save:
        with open(config.NOTES_PICKLE, "wb") as fh:
            pickle.dump(data, fh)
        print(f"Saved preprocessed data -> {config.NOTES_PICKLE}")
    return data


def load_prepared():
    if not os.path.exists(config.NOTES_PICKLE):
        raise SystemExit("No prepared data found. Run 'python main.py prepare' first.")
    with open(config.NOTES_PICKLE, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    prepare()
