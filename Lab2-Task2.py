import torch
import torch.nn as nn


# Task 2: Building the LSTM model in PyTorch
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


# Model parameters
vocab_size = 40  # (كمثال إذا كنت ستشغله منفصلاً، أو تأكده من Task 1)
embedding_dim = 64
hidden_dim = 128
num_layers = 2
batch_size = 1  # For text generation, we'll use batch size 1

model = LSTMTextGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)

print("Task 2 جاهز وتم بناء النموذج بنجاح!")