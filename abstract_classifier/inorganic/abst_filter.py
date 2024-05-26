import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", default="./MatBERT_16000", help="The path of the classifier model")
parser.add_argument("--dir_name", required=True, help="The path of file for filtering.")
parser.add_argument("--postfix", default="result.txt", help="convert files only if with this postfix")
parser.add_argument('--device', default="", type=str, help="Running Device")
args = parser.parse_args()
print(args.dir_name)

if args.device:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from abstract import BaseClsDataset
import torch

# For now, disable Torch2 Dynamo : caused with https://github.com/d8ahazard/sd_dreambooth_extension/pull/1186
os.environ["TORCHDYNAMO_DISABLE"] = '1'

cur_dir_path = os.getcwd()
target_dir_path = os.path.join(cur_dir_path, args.dir_name)
files = [f for f in os.listdir(target_dir_path) if f.endswith(args.postfix)]

model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    num_labels=2
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    model_max_length=512
)

for fname in tqdm(files,
        position=0,
        leave=True):

        with open(os.path.join(target_dir_path, fname), 'r') as f:
            lines = f.readlines()
        
        label_tmp = [0] * len(lines)

        predset = BaseClsDataset(lines, label_tmp, tokenizer)
        
        trainer = Trainer(
            model=model,
            )
        
        out = trainer.predict(predset)

        pred = torch.tensor(out.predictions).softmax(dim=1)
        pred = (pred[:,1] > 0.5).tolist()

        assert len(pred) == len(lines)

        filtered = []
        for line, lab in zip(lines, pred):
            if lab: filtered.append(line)

        print(f"In {predset.__len__()}, {len(filtered)} doc left.")
            
        with open(os.path.join( target_dir_path, "-filtered.".join(fname.split(".")) ), 'w') as f:
            f.writelines(filtered)