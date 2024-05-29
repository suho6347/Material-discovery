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
parser.add_argument("--input_file", default="dataset/toyset.txt")
parser.add_argument("--output_file", default="01-getCorpus-result.txt")
parser.add_argument("--output_formula_file", default="01-getCorpus-result-formula.txt")
args = parser.parse_args()

#init
test_input_file = open(args.input_file, "r")

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
test_write_file = open(os.path.join(output_dir, args.output_file_name), "w")
test_formulae_write_file = open(os.path.join(output_dir, args.output_formula_name), "w")


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
