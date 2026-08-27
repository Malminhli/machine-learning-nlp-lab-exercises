import torch
import torch.nn as nn
import numpy as np

# Data definitions required for vocabulary size
text = """To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles..."""
chars = sorted(list(set(text)))
vocab_size = len(chars)


# Define the LSTM model[cite: 1]
class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        out = self.fc(lstm_out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(num_layers, batch_size, hidden_dim),
                torch.zeros(num_layers, batch_size, hidden_dim))


# Model parameters[cite: 1]
embedding_dim = 64
hidden_dim = 128
num_layers = 2
batch_size = 1  # For text generation, we'll use batch size 1[cite: 1]

model = LSTMTextGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)
print("Task 2 completed successfully! Model initialized:")
print(model)