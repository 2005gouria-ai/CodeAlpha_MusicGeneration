# Task 3 — Music Generation with AI

Train a deep-learning model on MIDI music and let it compose new melodies.
Built for the CodeAlpha Artificial Intelligence internship.

## How it works

1. **Collect MIDI music data** — use your own `.mid` files, or generate a
   starter corpus with the built-in script.
2. **Preprocess the data** — `music21` parses each file into a flat sequence
   of notes and chords; the sequence is mapped to integers and sliced into
   fixed-length training windows.
3. **Build a deep-learning model** — a two-layer **LSTM** (TensorFlow/Keras)
   that learns to predict the next note.
4. **Train the model** on the sequences.
5. **Convert generated sequences to MIDI** and save them (rendered to audio
   automatically if `fluidsynth` is installed).

A TensorFlow-free **Markov chain** fallback is included so the project always
produces music, even without a deep-learning install.

## Requirements

- Python 3.8+
- See `requirements.txt` (TensorFlow is the heavy one)

## Setup (VS Code terminal)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick start

```bash
# 1. Create a starter MIDI dataset (skip if you have your own in data/midi/)
python main.py corpus

# 2. Preprocess MIDI into note sequences
python main.py prepare

# 3. Train the LSTM
python main.py train --epochs 40

# 4. Generate new music -> output/generated.mid
python main.py generate --length 200 --temperature 1.0
```

Or do everything in one go:

```bash
python main.py all --epochs 40
```

### No TensorFlow? Use the fallback

```bash
python main.py markov --length 200
```

This writes `output/generated_markov.mid` using a second-order Markov chain
and needs only `music21` and `numpy`.

## Using your own music

Drop any `.mid` / `.midi` files into `data/midi/` and run `python main.py prepare`
followed by `python main.py train`. The pipeline is dataset-agnostic.

## Command reference

| Command | What it does |
|---------|--------------|
| `corpus` | Generate a small sample MIDI dataset |
| `prepare` | Parse MIDI → note sequences → `data/notes.pkl` |
| `train` | Train the LSTM (`--epochs`, `--batch`, `--units`) |
| `generate` | Generate music (`--length`, `--temperature`, `--output`) |
| `markov` | TensorFlow-free fallback generator |
| `all` | corpus + prepare + train + generate |

## Files

| File | Purpose |
|------|---------|
| `main.py` | Command-line entry point |
| `config.py` | Paths and hyperparameters |
| `make_sample_corpus.py` | Builds a starter MIDI dataset |
| `data_utils.py` | MIDI collection + preprocessing |
| `model.py` | LSTM build / train / generate (TensorFlow) |
| `markov_model.py` | Dependency-light fallback generator |
| `generate.py` | Turns note tokens into MIDI (and audio) |
| `requirements.txt` | Dependencies |

## Notes

- Training is CPU-friendly at the defaults. Lower `--units` (e.g. 64) if you
  are short on memory.
- `--temperature` below 1.0 gives safer, more repetitive music; above 1.0
  gives more surprising results.
- For a richer dataset, public MIDI collections such as the Classical Piano
  MIDI archive work well — just drop the files into `data/midi/`.
