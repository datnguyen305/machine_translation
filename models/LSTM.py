import torch 
import torch.nn as nn
from vocab import Vocab

class Encoder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.config = config 
        self.embedding = nn.Embedding(vocab.src_vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim,
                            config.hidden_dim,
                            num_layers=config.num_layers,
                            dropout=config.dropout,
                            bidirectional=config.bidirectional,
                            batch_first=True)
        
    def forward(self, input):
        embedded = self.embedding(input)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, vocab, config): 
        super().__init__()
        decoder_hidden_input_size = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.vocab = vocab
        self.config = config 
        self.embedding = nn.Embedding(vocab.tgt_vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            decoder_hidden_input_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=False,
            batch_first=True
        )
        self.fc_out = nn.Linear(decoder_hidden_input_size, vocab.tgt_vocab_size)
    def forward(self, encoder_outputs, states, target):
        batch_size = encoder_outputs.size(0)
        target_len = target.size(1)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.vocab.bos_idx).to(self.config.device)
        decoder_outputs = []
        for i in range(target_len):
            decoder_output, states = self.forward_step(decoder_input, states)
            decoder_outputs.append(decoder_output)
            # Here comes the teacher forcing
            decoder_input = target[:, i].unsqueeze(1)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, states


    def forward_step(self, input, states):
        embedded = self.embedding(input)
        outputs, states = self.lstm(embedded, states)
        prediction = self.fc_out(outputs.squeeze(1))
        return prediction, states
    
class LSTM(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.encoder = Encoder(vocab, config)
        self.decoder = Decoder(vocab, config)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.vocab = vocab
        self.config = config
        encoder_state_dim = config.hidden_dim
        decoder_state_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim

    def forward(self, src, tgt):
        encoder_outputs, states = self.encoder(src)
        hidden_reshaped, cell_reshaped = self._reshape_encoder_states(states[0], states[1])
        states_reshaped = (hidden_reshaped, cell_reshaped)
        outs, _ = self.decoder(encoder_outputs, states_reshaped, tgt)
        
        loss = self.loss(outs.reshape(-1, self.vocab.tgt_vocab_size), tgt.reshape(-1)) # loss input: (N, C), target: (N)
        return loss
    
    def _reshape_encoder_states(self, hidden, cell):
        # Đã giả định config.bidirectional=True
        if self.config.bidirectional:
            # Chuyển đổi (4, B, 256) -> (2, 2, B, 256)
            hidden = hidden.view(self.config.num_layers, 2, hidden.shape[1], hidden.shape[2])
            cell = cell.view(self.config.num_layers, 2, cell.shape[1], cell.shape[2])
            
            # Concatenate 2 hướng -> (2, B, 512)
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
            cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
            
        return hidden, cell
    
    def predict(self, src, tgt):
        encoder_outputs, states = self.encoder(src)
        hidden_reshaped, cell_reshaped = self._reshape_encoder_states(states[0], states[1])
        states_reshaped = (hidden_reshaped, cell_reshaped)
        
        batch_size = src.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.vocab.bos_idx).to(self.config.device)
        predicted_tokens = []
        max_len = tgt.size(1)
        
        for _ in range(max_len):
            decoder_output, states_reshaped = self.decoder.forward_step(decoder_input, states_reshaped)
            top1 = decoder_output.argmax(1) 
            predicted_tokens.append(top1.unsqueeze(1)) 
            decoder_input = top1.unsqueeze(1) 
            
        predicted_tokens = torch.cat(predicted_tokens, dim=1) 
        return predicted_tokens