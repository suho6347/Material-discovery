import datetime
from gensim.models.callbacks import CallbackAny2Vec

class DOIsDataLoader:
    file_path = ""

    def __init__(self, fp):
        self.file_path = fp

    def load(self, issplit=False, islower=False):
        f = open(self.file_path, "r")
        doi_vocab = {}
        for line in f:
            words = line.strip()
            if islower: words = words.lower()
            if issplit: words = words.split()[0]
            if words not in doi_vocab: doi_vocab[words] = 1
            else: doi_vocab[words] += 1

        print("checking vocab length : ", len(doi_vocab))
        return doi_vocab
    
    def load_split(self, islower=False):
        f = open(self.file_path, "r")
        doi_vocab = {}
        for line in f:
            line = line.strip()
            if islower: line = line.lower()
            words = line.split()
            k = words[0] # Li
            if len(words) == 1: v = ["__empty__"]
            else: v = words[1:]
            if k not in doi_vocab: doi_vocab[k] = [v] # [["__empty__"]]
            else: doi_vocab[k].append(v) # [["__empty__"], [":", "Lithium"]]
        
        print("checking vocab length : ", len(doi_vocab))
        return doi_vocab

class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self, dir=None):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.logdir = dir
        self.clock = datetime.datetime.now()
        
    def on_epoch_end(self, model):
        # loss = model.get_latest_training_loss()
        # loss_now = loss - self.loss_to_be_subed
        # self.loss_to_be_subed = loss
        
        temp = datetime.datetime.now()
        td = temp - self.clock
        
        print(f'Loss at epoch {self.epoch}\t{str(td).split(".")[0]} passed.')
        self.clock = temp
        
        # if self.logdir:
        #     model.wv.save(self.logdir + f"/{self.epoch}.vec")
        
        self.epoch += 1