"""Central configuration for the Music Generation project."""

import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MIDI_DIR = os.path.join(DATA_DIR, "midi")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Saved artefacts
NOTES_PICKLE = os.path.join(DATA_DIR, "notes.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "music_lstm.keras")

# Training / sequence settings
SEQUENCE_LENGTH = 24      # how many past notes the model sees
STEP = 1                  # stride when slicing training windows
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
LSTM_UNITS = 128          # hidden units per LSTM layer
DROPOUT = 0.3

# Generation settings
DEFAULT_GENERATE_LENGTH = 120
DEFAULT_TEMPERATURE = 1.0  # <1 = safer, >1 = more surprising

# Sample corpus
SAMPLE_CORPUS_SIZE = 60    # number of short MIDI pieces to synthesise
RANDOM_SEED = 42


def ensure_dirs():
    for path in (DATA_DIR, MIDI_DIR, OUTPUT_DIR, MODEL_DIR):
        os.makedirs(path, exist_ok=True)
