import os
import torch
import argparse
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from abstract import BaseClsDataset


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, default="./MatBERT_16000", help="The path of the classifier model")
parser.add_argument('--input_file_dir', type=str, default="output/")
parser.add_argument("--postfix", type=str, default="result.txt", help="convert files only if with this postfix")
parser.add_argument('--device', type=str, default="", help="Running Device")
args = parser.parse_args()


if args.device:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For now, disable Torch2 Dynamo : caused with https://github.com/d8ahazard/sd_dreambooth_extension/pull/1186
os.environ["TORCHDYNAMO_DISABLE"] = '1'



files = [f for f in os.listdir(args.input_file_dir) if f.endswith(args.postfix)]

model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    num_labels=2
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    model_max_length=512
)

for fname in tqdm(files, position=0, leave=True, bar_format="{l_bar}{bar:20}{r_bar}"):
    with open(os.path.join(args.input_file_dir, fname), 'r') as fr:
        lines = fr.readlines()
        fr.close()
    
    label_tmp = [0] * len(lines)
    
    predset = BaseClsDataset(lines, label_tmp, tokenizer)
    
    trainer = Trainer(model=model)
    
    out = trainer.predict(predset)
    
    pred = torch.tensor(out.predictions).softmax(dim=1)
    pred = (pred[:,1] > 0.5).tolist()
    
    assert len(pred) == len(lines)
    
    filtered = []
    for line, lab in zip(lines, pred):
        if lab: filtered.append(line)
    
    print(f"In {predset.__len__()}, {len(filtered)} doc left.")

    file_cont, file_ext = os.path.splitext(fname)
    output_file = os.path.join(args.input_file_dir, file_cont + "-filtered" + file_ext)
    with open(output_file, 'w') as fw:
        fw.writelines(filtered)
        fw.close()
    
