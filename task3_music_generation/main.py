"""
Music Generation with AI
========================

An LSTM learns note patterns from MIDI files and generates new music, which
is written out as a MIDI file (and rendered to audio if fluidsynth exists).

Run from the VS Code terminal:

    python main.py corpus          # create a starter MIDI dataset
    python main.py prepare         # preprocess MIDI -> note sequences
    python main.py train           # train the LSTM
    python main.py generate        # generate new music (MIDI in ./output)
    python main.py all             # do all of the above with defaults
    python main.py markov          # TensorFlow-free fallback generator

Requirements: see requirements.txt
"""

import argparse
import os

import config


def cmd_corpus(_args):
    import make_sample_corpus
    make_sample_corpus.main()


def cmd_prepare(_args):
    import data_utils
    data_utils.prepare()


def cmd_train(args):
    import data_utils
    import model as model_mod

    data = data_utils.load_prepared()
    model_mod.train(data, epochs=args.epochs, batch_size=args.batch, units=args.units)


def cmd_generate(args):
    import data_utils
    import model as model_mod
    import generate as gen

    data = data_utils.load_prepared()
    net = model_mod.load_model()
    print(f"Generating {args.length} steps (temperature={args.temperature}) ...")
    tokens = model_mod.generate_sequence(net, data, length=args.length,
                                         temperature=args.temperature)
    path = gen.save_midi(tokens, path=args.output)
    gen.try_render_audio(path)


def cmd_markov(args):
    import data_utils
    import generate as gen
    from markov_model import MarkovModel

    files = data_utils.collect_midi_files()
    if not files:
        raise SystemExit("No MIDI files. Run 'python main.py corpus' first.")
    notes = data_utils.extract_notes(files)
    print(f"Fitting Markov chain on {len(notes)} notes ...")
    mm = MarkovModel(order=2, seed=config.RANDOM_SEED).fit(notes)
    tokens = mm.generate(length=args.length)
    path = gen.save_midi(tokens, path=args.output)
    gen.try_render_audio(path)


def cmd_all(args):
    cmd_corpus(args)
    cmd_prepare(args)
    cmd_train(args)
    cmd_generate(args)


def build_parser():
    parser = argparse.ArgumentParser(description="Music Generation with AI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("corpus", help="generate a sample MIDI dataset")
    sub.add_parser("prepare", help="preprocess MIDI into note sequences")

    p_train = sub.add_parser("train", help="train the LSTM model")
    p_train.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    p_train.add_argument("--batch", type=int, default=config.DEFAULT_BATCH_SIZE)
    p_train.add_argument("--units", type=int, default=config.LSTM_UNITS)

    p_gen = sub.add_parser("generate", help="generate music with the trained model")
    p_gen.add_argument("--length", type=int, default=config.DEFAULT_GENERATE_LENGTH)
    p_gen.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    p_gen.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, "generated.mid"))

    p_mk = sub.add_parser("markov", help="generate music with the Markov fallback")
    p_mk.add_argument("--length", type=int, default=config.DEFAULT_GENERATE_LENGTH)
    p_mk.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, "generated_markov.mid"))

    p_all = sub.add_parser("all", help="corpus + prepare + train + generate")
    p_all.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    p_all.add_argument("--batch", type=int, default=config.DEFAULT_BATCH_SIZE)
    p_all.add_argument("--units", type=int, default=config.LSTM_UNITS)
    p_all.add_argument("--length", type=int, default=config.DEFAULT_GENERATE_LENGTH)
    p_all.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    p_all.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, "generated.mid"))

    return parser


def main():
    config.ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "corpus": cmd_corpus,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "generate": cmd_generate,
        "markov": cmd_markov,
        "all": cmd_all,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
