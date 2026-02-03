import os 
import json
import re

import torch 

class Vocab: 
    def __init__ (self, path, src_language, tgt_language):
        self.initialize_special_tokens()
        self.make_vocab(path, src_language, tgt_language)
        self.src_language = src_language
        self.tgt_language = tgt_language

    def initialize_special_tokens(self):
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.specials = [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.unk_token,
        ]

        self.pad_idx = 0 
        self.bos_idx = 1 
        self.eos_idx = 2
        self.unk_idx = 3

    def make_vocab(self, path, src_language, tgt_language):
        json_files = os.listdir(path)
        src_words = set()
        tgt_words = set()
        for json_file in json_files: 
            data = json.load(open(os.path.join(path, json_file), 'r', encoding='utf-8'))
            for item in data: 
                src_sentence = item[src_language]
                tgt_sentence = item[tgt_language]

                src_tokens = self.preprocess_sentence(src_sentence)
                tgt_tokens = self.preprocess_sentence(tgt_sentence)

                src_words.update(src_tokens)
                tgt_words.update(tgt_tokens)

        src_itos = self.specials + list(src_words)
        self.src_itos = {i: tok for i, tok in enumerate(src_itos)}
        self.src_stoi = {tok: i for i, tok in self.src_itos.items()}

        # src_itos = {
        #     0: "<pad>",
        #     1: "<bos>",
        #     2: "<eos>",
        #     3: "<unk>"
        # }

        # src_stoi = {
        #     "<pad>": 0,
        #     "<bos>": 1,
        #     "<eos>": 2,
        #     "<unk>": 3
        # }

        tgt_itos = self.specials + list(tgt_words)
        self.tgt_itos = {i: tok for i, tok in enumerate(tgt_itos)}
        self.tgt_stoi = {tok: i for i, tok in self.tgt_itos.items()}

    def total_tgt_tokens(self) -> int: 
        return len(self.tgt_itos)
    
    @property
    def src_vocab_size(self) -> int:
        return len(self.src_itos)
    
    @property
    def tgt_vocab_size(self) -> int:
        return len(self.tgt_itos)
    

    def preprocess_sentence(self, sentence: str): # Mục tiêu giữ lại từ và số 
        sentence = sentence.lower()
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"\!", " ! ", sentence)
        sentence = re.sub(r"\/", " / ", sentence)
        sentence = re.sub(r"\?", " ? ", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r":", " : ", sentence)
        sentence = re.sub(r";", " ; ", sentence)
        sentence = re.sub(r"\"", " \ ", sentence)
        sentence = re.sub(r"\(", " ( ", sentence)
        sentence = re.sub(r"\)", " ) ", sentence)
        sentence = re.sub(r"\[", " [ ", sentence)
        sentence = re.sub(r"\]", " ] ", sentence)
        sentence = re.sub(r"\{", " { ", sentence)
        sentence = re.sub(r"\}", " } ", sentence)
        sentence = re.sub(r"\.", " . ", sentence)
        sentence = re.sub(r"-", " - ", sentence)
        sentence = re.sub(r"\$", " $ ", sentence)
        sentence = re.sub(r"\&", " & ", sentence)
        sentence = re.sub(r"\*", " * ", sentence)
        sentence = re.sub(r"%", " % ", sentence)

        sentence = " ".join(sentence.strip().split())  # remove duplicated spaces
        tokens = sentence.strip().split() # "Đạt bị    ngu \t" -> "Đạt bị ngu" -> ["Đạt", "bị", "ngu"]
        return tokens
    def encode_sentence(self, sentence, language) -> torch.Tensor: 
        tokens = self.preprocess_sentence(sentence)
        stoi = self.src_stoi if language == self.src_language else self.tgt_stoi
        vec = [stoi[token] if token in stoi else self.unk_idx for token in tokens]
        vec = [self.bos_idx] + vec + [self.eos_idx]
        vec = torch.tensor(vec, dtype=torch.long)
        return vec
    def decode_sentence(self, tensor: list[int], language) -> str:
        sentences_ids = tensor.tolist()
        sentences = []
        itos = self.src_itos if language == self.src_language else self.tgt_itos
        for sentence_ids in sentences_ids:
            words = [itos[idx] for idx in sentence_ids if idx not in self.specials]
            sentence = " ".join(words)
            sentences.append(sentence)
        return sentences