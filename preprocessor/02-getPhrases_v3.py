import gensim
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
import regex
import argparse
import pdb
import os
from tqdm import tqdm
from mat2vec_origin_training.helpers.utils import EpochSaver, compute_epoch_accuracies, \
    keep_simple_formula, load_obj, COMMON_TERMS, EXCLUDE_PUNCT, INCLUDE_PHRASES, EXCLUDE_TERMS
"""
requirements

    pip install unidecode
    pip install monty
    pip install chemdataextractor
    pip install pymatgen

usage

    python 02-getPhrases_v3.py --dir_name ../../corpus/SemanticScholar/year_3_1000/query_1C+rate_1922_2023
    python 02-getPhrases_v3.py --dir_name ../../corpus/SemanticScholar/year_3_1000/query_1C+rate_1922_2023 --input_file_name 01-getCorpus-result-filtered.txt

    python 02-getPhrases_v3.py \
         --dir_name ../../corpus/corpus_with_improved_normalization/for_publishing/corpus_filtered_mat2vec+after2019+elbattery+ssbattery

"""

def exclude_words(phrasegrams, words):
    """Given a list of words, excludes those from the keys of the phrase dictionary."""
    new_phrasergrams = {}
    words_re_list = []
    for word in words:
        we = regex.escape(word)
        words_re_list.append("^" + we + "$|^" + we + "_|_" + we + "$|_" + we + "_")
    word_reg = regex.compile(r""+"|".join(words_re_list))
    for gram in tqdm(phrasegrams, desc='phrasegrams'):
        valid = True
        if word_reg.search(gram.encode().decode("unicode_escape", "ignore")) is not None:
            valid = False
            if gram == "As_a_result": valid =True
        if valid:
            new_phrasergrams[gram] = phrasegrams[gram]
    return new_phrasergrams

# Generating word grams.
def wordgrams(sent, depth, pc, th, ct, et, ip, d=0):
    print("wordgrams depth : ", d)
    if depth == 0:
        return sent, None
    else:
        """Builds word grams according to the specification."""
        print("Phrases working")
        phrases = Phrases(
            sent,
            connector_words=ct, 
            min_count=pc,
            threshold=th)

        grams = Phraser(phrases)
        print("exclude_words working")
        grams.phrasegrams = exclude_words(grams.phrasegrams, et)
        d += 1
        if d < depth:
            return wordgrams(grams[sent], depth, pc, th, ct, et, ip, d)
        else:
            return grams[sent], grams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir_name", required=True)
    parser.add_argument("--input_file_name", default="01-getCorpus-result.txt")
    parser.add_argument("--input_formula_name", default="01-getCorpus-result-formula.txt")
    parser.add_argument("--output_file_name", default="02-getPhrases-result.txt")
    parser.add_argument("--output_formula_name", default="02-getPhrases-result-formula.txt")
    # parser.add_argument("--batch", default=10000, help="Minibatch size.")
    parser.add_argument("--phrase_depth", default=2, help="The number of passes to perform for phrase generation.")
    parser.add_argument("--phrase_count", default=10, help="Minimum number of occurrences for phrase to be considered.")
    parser.add_argument("--phrase_threshold", default=15.0, help="Phrase importance threshold.")
    # parser.add_argument("-include_extra_phrases",
    #                     action="store_true",
    #                     help="If true, will look for all_ents.p and add extra phrases.")
    # parser.add_argument("-keep_formula", action="store_true",
    #                     help="If set, keeps simple chemical formula independent on count.")
    args = parser.parse_args()


    # discard formula, which have freque under phrase_count.
    all_formula = []
    all_formula_count = {}
    formula_counts = []
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    target_dir_paht = os.path.join(cur_dir_path, args.dir_name)
    formula_file = open(os.path.join(target_dir_paht, args.input_formula_name), "r")
    for line in formula_file:
        words = line.strip().split()
        all_formula.append(words[0])
        all_formula_count[words[0]] = 0
    formula_file.close()

    # get corpus and check formulas over min-count
    with open(os.path.join(target_dir_paht, args.input_file_name), "r") as f:
        for line in f:
            line = line.strip().split()
            for word in line:
                if word in all_formula_count:
                    all_formula_count[word] += 1
    all_formula = []
    for k, v in all_formula_count.items():
        if v > args.phrase_count:
            all_formula.append(k)
    processed_formulas = all_formula
    print("min count formula size : ", len(processed_formulas))


    # Loading text and generating the phrases.
    sentences = LineSentence(os.path.join(target_dir_paht, args.input_file_name))



    
    # Pre-process everything to force the supplied phrases before it even goes to the phraser.
    processed_sentences = sentences
    # if args.include_extra_phrases:
    #     phrases_by_length = dict()
    #     for phrase in INCLUDE_PHRASES:
    #         phrase_split = phrase.split("_")
    #         if len(phrase_split) not in phrases_by_length:
    #             phrases_by_length[len(phrase_split)] = [phrase]
    #         else:
    #             phrases_by_length[len(phrase_split)].append(phrase)
    #     max_len = max(phrases_by_length.keys())

    #     processed_sentences = []
    #     for sentence in tqdm(sentences):
    #         for cl in reversed(range(2, max_len + 1)):
    #             repl_phrases = set(phrases_by_length[cl])
    #             si = 0
    #             while si <= len(sentence) - cl:
    #                 if "_".join(sentence[si:cl + si]) in repl_phrases:
    #                     sentence[si] = "$@$@$".join(sentence[si:cl + si])
    #                     del(sentence[si + 1:cl + si])
    #                 else:
    #                     si += 1
    #         processed_sentences.append(sentence)





    # Process sentences to force the extra phrases.
    sentences, phraser = wordgrams(processed_sentences,
                          depth=int(args.phrase_depth),
                          pc=int(args.phrase_count),
                          th=float(args.phrase_threshold),
                          ct=COMMON_TERMS,
                          et=EXCLUDE_PUNCT + EXCLUDE_TERMS + processed_formulas,
                          ip=INCLUDE_PHRASES)
    # phraser.save(os.path.join("models", args.model_name + "_phraser.pkl"))

    output_corpus_file = open(os.path.join(target_dir_paht, args.output_file_name), "w")
    for s in sentences:
        output_corpus_file.write(' '.join(s).strip())
        output_corpus_file.write("\n")
    output_corpus_file.close()

    print("done ! ")
    print("formula vocab size : ", len(processed_formulas))