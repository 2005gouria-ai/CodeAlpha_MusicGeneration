"""
The deep-learning model: a two-layer LSTM that learns note sequences and
predicts the next note.

Uses TensorFlow / Keras. Install it with:  pip install tensorflow
"""

import os

# Keep TensorFlow within modest memory/CPU limits (helps on laptops and
# small machines). Set before TensorFlow is imported.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

import config


def build_model(n_vocab, seq_len=None, units=None, dropout=None):
    units = units or config.LSTM_UNITS
    dropout = config.DROPOUT if dropout is None else dropout
    """Create and compile the LSTM model."""
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential

    seq_len = seq_len or config.SEQUENCE_LENGTH

    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units),
        Dropout(dropout),
        Dense(n_vocab, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def _one_hot(sequences, n_vocab):
    from tensorflow.keras.utils import to_categorical

    return to_categorical(sequences, num_classes=n_vocab)


def train(data, epochs=None, batch_size=None, units=None, verbose=1):
    """Train the model on the prepared data and save it."""
    epochs = epochs or config.DEFAULT_EPOCHS
    batch_size = batch_size or config.DEFAULT_BATCH_SIZE

    n_vocab = data["n_vocab"]
    seq_len = data["sequence_length"]

    X = np.reshape(data["network_input"], (len(data["network_input"]), seq_len, 1))
    X = X / float(n_vocab)
    y = _one_hot(data["network_output"], n_vocab)

    model = build_model(n_vocab, seq_len, units=units)
    print(f"Training LSTM: {len(X)} windows, {n_vocab} notes, {epochs} epochs ...")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.MODEL_FILE)
    print(f"Saved model -> {config.MODEL_FILE}")
    return model


def load_model():
    from tensorflow.keras.models import load_model

    if not os.path.exists(config.MODEL_FILE):
        raise SystemExit("No trained model found. Run 'python main.py train' first.")
    return load_model(config.MODEL_FILE)


def _sample(preds, temperature=1.0):
    """Draw an index from a probability vector, with a temperature knob."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp = np.exp(preds)
    preds = exp / np.sum(exp)
    return int(np.random.choice(len(preds), p=preds))


def generate_sequence(model, data, length=None, temperature=None):
    """Generate a list of note tokens continuing from a random seed."""
    length = length or config.DEFAULT_GENERATE_LENGTH
    temperature = temperature or config.DEFAULT_TEMPERATURE

    n_vocab = data["n_vocab"]
    seq_len = data["sequence_length"]
    int_to_note = {i: n for n, i in data["note_to_int"].items()}

    start = np.random.randint(0, len(data["network_input"]) - 1)
    pattern = list(data["network_input"][start])

    prediction_output = []
    for _ in range(length):
        x = np.reshape(pattern, (1, seq_len, 1)) / float(n_vocab)
        preds = model.predict(x, verbose=0)[0]
        index = _sample(preds, temperature)
        prediction_output.append(int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output
