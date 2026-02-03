import torch 

class LSTM_config:
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 3
    dropout = 0.5
    bidirectional = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50
    num_workers = 4
    clip = 1.0

class BahdanauLSTM_config:
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 3
    dropout = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50
    num_workers = 4
    bidirectional = True
    clip = 1.0

class LuongLSTM_config:
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 3
    dropout = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50
    num_workers = 4
    bidirectional = True
    clip = 1.0
    attention_type = 'dot'