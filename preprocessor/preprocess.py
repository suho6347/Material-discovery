import pdb
import pandas as pd
from normalization_utils import Battery2Vec_Processor
import sys
import os
import re
import argparse
from tqdm import tqdm


# init args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_dir", default="./")
parser.add_argument("--input_file_name", default="toyset.txt")
parser.add_argument("--output_dir", default="./")
parser.add_argument("--output_file_name", default="01-getCorpus-result.txt")
parser.add_argument("--output_formula_name", default="01-getCorpus-result-formula.txt")
args = parser.parse_args()

#init
test_write_file = open(os.path.join(args.output_dir, args.output_file_name), "w")
test_formulae_write_file = open(os.path.join(args.output_dir, args.output_formula_name), "w")
test_input_file = open(os.path.join(args.input_dir, args.input_file_name), "r")

battery2vec_processor = Battery2Vec_Processor()

material_dict = {}

## check file
for i, line in tqdm(enumerate(test_input_file)):
    print(f"line count : {i}", end="\r")
    line = line.strip()

    # normalization
    st1, material_list = battery2vec_processor.process(abst=line, ttl=None, doi=None)
    
    # write preprocessed data
    test_write_file.write(st1 + "\n")

    # make material dict by dividing
    for l in material_list:
        if l[1] not in material_dict:
            material_dict[l[1]] = [l[0]]
        else:
            material_dict[l[1]].append(l[0])

for i, (k, vs) in enumerate(material_dict.items()):
    test_formulae_write_file.write(k + " : ") # normalized material formula
    for v in vs:
        test_formulae_write_file.write(v + " ") # original material formulas
    test_formulae_write_file.write("\n")

test_write_file.close()
test_formulae_write_file.close()
test_input_file.close()