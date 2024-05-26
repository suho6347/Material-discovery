import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd

class BaseClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = tokenizer(texts, truncation=True, padding=True)
        self.labels= labels
    
    def __len__(self,):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def cutdown(self, cut=0):
        if cut: # cut=0 means not to cut
            if isinstance(cut, float):
                cut = int(cut * self.__len__())
            
            self.labels = self.labels[:cut]

class AbstClsDataset(BaseClsDataset):
    def __init__(self, trueset, falseset):
        texts = trueset + falseset
        labels = [1]*len(trueset) + [0]*len(falseset)

        self.texts, self.labels = shuffle(texts, labels)
    
    def train_test_split(self, tokenizer, ratio=.1, seed:int=0):
        
        X_train, X_test, Y_train, Y_test = train_test_split(self.texts, self.labels, test_size=ratio, random_state=seed)

        return BaseClsDataset(X_train, Y_train, tokenizer), BaseClsDataset(X_test, Y_test, tokenizer)

class CSV_AbstDataset(BaseClsDataset):
    def __init__(self, csv_name, tokenizer):
        data = pd.read_csv(csv_name)
        self.texts = tokenizer(data.text.tolist(), truncation=True, padding=True)
        self.labels = data.label.tolist()


def cls_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auc
    }