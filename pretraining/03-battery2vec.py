from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence
import gensim
import logging
import os
import argparse
import regex
import pickle
from tqdm import tqdm

import sys
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
preprocessor_dir = os.path.join(parent_dir, "preprocessor")
sys.path.append(preprocessor_dir)
from mat2vec_origin_training.helpers.utils import EpochSaver, compute_epoch_accuracies, \
    keep_simple_formula, load_obj, COMMON_TERMS, EXCLUDE_PUNCT, INCLUDE_PHRASES
from eval_utils import DOIsDataLoader, callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", required=True, help="The path to the corpus to train on.")
    parser.add_argument("--formulas", required=True, help="The path to the formulas to keep with. (requires 01-getCorpus-result-formula.txt)")
    parser.add_argument("--model_name", required=True, help="Name for saving the model (in the models folder).")
    parser.add_argument("--model_type", choices=['word2vec', 'fasttext'], default="word2vec", help="The type of model, which determine how to train.")
    parser.add_argument("--epochs", default=30, help="Number of epochs.")
    parser.add_argument("--size", default=200, help="Size of the embedding.")
    parser.add_argument("--window", default=8, help="Context window size.")
    parser.add_argument("--min_count", default=5, help="Minimum number of occurrences for word.")
    parser.add_argument("--workers", default=16, help="Number of workers.")
    parser.add_argument("--alpha", default=0.01, help="Learning rate.")
    parser.add_argument("--batch", default=10000, help="Minibatch size.")
    parser.add_argument("--negative", default=15, help="Number of negative samples.")
    parser.add_argument("--subsample", default=0.0001, help="Subsampling rate.")
    parser.add_argument("-include_extra_phrases",
                        action="store_true",
                        help="If true, will look for all_ents.p and add extra phrases.")
    parser.add_argument("-sg", action="store_true", help="If set, will train a skip-gram, otherwise a CBOW.")
    parser.add_argument("-hs", action="store_true", help="If set, hierarchical softmax will be used.")
    parser.add_argument("-keep_formula", action="store_true",
                        help="If set, keeps simple chemical formula independent on count.")
    parser.add_argument("-notmp", action="store_true", help="If set, will not store the progress in tmp folder.")
    args = parser.parse_args()

    all_formula = []
    if args.keep_formula:
        try:
            dl = DOIsDataLoader(args.formulas)
            all_formula = dl.load(issplit=True)  # list of formula is supplied

            def keep_formula_list(word, count, min_count):
                if word in all_formula:
                    return gensim.utils.RULE_KEEP
                else:
                    return gensim.utils.RULE_DEFAULT
            trim_rule_formula = keep_formula_list
            logging.info("Using a supplied list of formula to keep simple formula.")
        except:
            # no list is supplied, use the simple formula rule
            trim_rule_formula = keep_simple_formula
            logging.info("Using a function to keep material mentions.")
    else:
        logging.info("Basic min_count trim rule for formula.")
        trim_rule_formula = None

    # The trim rule for extra phrases to always keep them, similar to the formulae.
    if args.include_extra_phrases:
        INCLUDE_PHRASES_SET = set(INCLUDE_PHRASES)
        try:
            with open("all_ents.p", "rb") as f:
                INCLUDE_PHRASES += list(set(pickle.load(f)))
                INCLUDE_PHRASES_SET = set([ip.replace("_", "$@$@$") for ip in INCLUDE_PHRASES])
                logging.info("Included the supplied {} additional phrases.".format(len(INCLUDE_PHRASES)))
        except:
            logging.info("No specific phrases supplied, using the defaults.")

        def keep_extra_phrases(word, count, min_count):
            if word in INCLUDE_PHRASES_SET or trim_rule_formula is not None and \
                    trim_rule_formula(word, 1, 2) == gensim.utils.RULE_KEEP:
                return gensim.utils.RULE_KEEP
            else:
                return gensim.utils.RULE_DEFAULT

        trim_rule = keep_extra_phrases
        logging.info("Keeping the extra phrases independent on their count.")
    else:
        trim_rule = trim_rule_formula
        logging.info("Not including extra phrases, option not specified.")

    # Loading text and generating the phrases.
    sentences = LineSentence(args.corpus)

    # Pre-process everything to force the supplied phrases before it even goes to the phraser.
    processed_sentences = sentences
    if args.include_extra_phrases:
        phrases_by_length = dict()
        for phrase in INCLUDE_PHRASES:
            phrase_split = phrase.split("_")
            if len(phrase_split) not in phrases_by_length:
                phrases_by_length[len(phrase_split)] = [phrase]
            else:
                phrases_by_length[len(phrase_split)].append(phrase)
        max_len = max(phrases_by_length.keys())

        processed_sentences = []
        for sentence in tqdm(sentences):
            for cl in reversed(range(2, max_len + 1)):
                repl_phrases = set(phrases_by_length[cl])
                si = 0
                while si <= len(sentence) - cl:
                    if "_".join(sentence[si:cl + si]) in repl_phrases:
                        sentence[si] = "$@$@$".join(sentence[si:cl + si])
                        del(sentence[si + 1:cl + si])
                    else:
                        si += 1
            processed_sentences.append(sentence)

    if not args.notmp:
        callbacks = [callback()]
    else:
        callbacks = []

    model_type = args.model_type
    model = None
    if model_type == "word2vec": model=Word2Vec
    elif model_type == "fasttext": model=FastText

    my_model = model(
        sentences,
        # size=int(args.size),
        vector_size=int(args.size),
        window=int(args.window),
        min_count=int(args.min_count),
        sg=bool(args.sg),
        hs=bool(args.hs),
        trim_rule=trim_rule,
        workers=int(args.workers),
        alpha=float(args.alpha),
        sample=float(args.subsample),
        negative=int(args.negative),
        sorted_vocab=True,
        batch_words=int(args.batch),
        # iter=int(args.epochs),
        epochs=int(args.epochs),
        callbacks=callbacks)
    my_model.save(os.path.join("./", args.model_name))

    # analogy_file = os.path.join("data", "analogies.txt")
    # Save the accuracies in the tmp folder.
    # compute_epoch_accuracies("tmp", args.model_name, analogy_file)
