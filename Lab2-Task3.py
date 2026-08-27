import torch
import numpy as np
import torch.nn as nn

# متطلبات أساسية لتعريف البيانات والنموذج ليعمل Task 3 بمفرده
text = """To be, or not to be, that is the question: Whether 'tis
nobler in the mind to suffer
The slings and arrows of outrageous fortune, Or to take arms against a
sea of troubles..."""

chars = sorted(list(set(text)))
vocab_size = len(chars)
data = np.array([{ch: i for i, ch in enumerate(chars)}[c] for c in text])

embedding_dim = 64
hidden_dim = 128
num_layers = 2
batch_size = 1


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


model = LSTMTextGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)

# ==========================================
# Task 3: Training the LSTM model
# ==========================================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Function to get batches of input and target data
def get_batch(data, seq_length):
    start_idx = np.random.randint(0, len(data) - seq_length)
    inputs = torch.tensor(data[start_idx:start_idx + seq_length], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(data[start_idx + 1:start_idx + seq_length + 1], dtype=torch.long).unsqueeze(0)
    return inputs, targets


seq_length = 100  # Length of input sequences
num_epochs = 500  # Training epochs

# Train the model
for epoch in range(num_epochs):
    model.train()
    inputs, targets = get_batch(data, seq_length)
    hidden = model.init_hidden(batch_size)

    optimizer.zero_grad()
    outputs, hidden = model(inputs, hidden)

    loss = criterion(outputs.squeeze(), targets.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

print("Task 3 جاهز وتم تدريب النموذج بنجاح!")