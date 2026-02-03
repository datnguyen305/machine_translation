import torch 
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_fn(items: list[dict]) -> dict:
    vi_sents = [item["vietnamese"] for item in items]
    en_sents = [item["english"] for item in items]

    vi_sents = pad_sequence(vi_sents, batch_first=True, padding_value=0)
    en_sents = pad_sequence(en_sents, batch_first=True, padding_value=0)

    return {
        "vietnamese": vi_sents,
        "english": en_sents
    }

class PhoMTDataset(Dataset):
    def __init__(self, path, vocab):
        super().__init__()

        data = json.load(open(path, 'r', encoding='utf-8'))
        self.data = data 
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        vi_sentence = item["vietnamese"]
        en_sentence = item["english"]

        encoded_vi = self.vocab.encode_sentence(vi_sentence, language="vietnamese")
        encoded_en = self.vocab.encode_sentence(en_sentence, language="english")
        
        return {
            "vietnamese": encoded_vi,
            "english": encoded_en   
        }