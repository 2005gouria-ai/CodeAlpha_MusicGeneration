"""
Turn generated note tokens into a MIDI file (and, if possible, audio).
"""

import os
import shutil
import subprocess

from music21 import chord, instrument, note, stream

import config


def notes_to_stream(prediction_output, offset_step=0.5):
    """Convert a list of tokens ('C4', 'C4.E4.G4') into a music21 Stream."""
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if "." in pattern:  # a chord
            pitches = pattern.split(".")
            new_chord = chord.Chord(pitches)
            new_chord.quarterLength = offset_step
            output_notes.append(new_chord)
        else:               # a single note
            new_note = note.Note(pattern)
            new_note.quarterLength = offset_step
            output_notes.append(new_note)
        offset += offset_step

    midi_stream = stream.Stream(output_notes)
    midi_stream.insert(0, instrument.Piano())
    return midi_stream


def save_midi(prediction_output, path=None):
    path = path or os.path.join(config.OUTPUT_DIR, "generated.mid")
    config.ensure_dirs()
    midi_stream = notes_to_stream(prediction_output)
    midi_stream.write("midi", fp=path)
    print(f"Saved MIDI -> {path}")
    return path


def try_render_audio(midi_path):
    """
    Best-effort MIDI -> WAV using fluidsynth, if it happens to be installed.
    Audio rendering is optional; the MIDI file always works on its own.
    """
    if not shutil.which("fluidsynth"):
        print("(Audio preview skipped: install 'fluidsynth' to render a WAV/MP3.)")
        return None
    wav_path = os.path.splitext(midi_path)[0] + ".wav"
    try:
        subprocess.run(["fluidsynth", "-ni", "-F", wav_path, "-r", "44100", midi_path],
                       check=True, capture_output=True)
        print(f"Rendered audio -> {wav_path}")
        return wav_path
    except Exception as exc:
        print(f"(Audio render failed: {exc})")
        return None
