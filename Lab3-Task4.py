import torch
import torch.nn as nn
import numpy as np

# Prerequisites & quick pre-training setup to enable generation
text = """To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles..."""
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
data = np.array([char_to_idx[c] for c in text])


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


embedding_dim, hidden_dim, num_layers, batch_size = 64, 128, 2, 1
model = LSTMTextGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)

# Quick training loop so generation has trained weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for _ in range(200):
    start_idx = np.random.randint(0, len(data) - 100)
    inputs = torch.tensor(data[start_idx:start_idx + 100], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(data[start_idx + 1:start_idx + 101], dtype=torch.long).unsqueeze(0)
    hidden = model.init_hidden(batch_size)
    optimizer.zero_grad()
    outputs, hidden = model(inputs, hidden)
    loss = criterion(outputs.squeeze(), targets.view(-1))
    loss.backward()
    optimizer.step()


# Text generation function[cite: 1]
def generate_text(model, start_text, length, temperature=0.8):
    model.eval()
    input_seq = torch.tensor([char_to_idx[c] for c in start_text], dtype=torch.long).unsqueeze(0)
    hidden = model.init_hidden(batch_size)
    generated_text = start_text

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        output_dist = output.squeeze().div(temperature).exp()
        predicted_char_idx = torch.multinomial(output_dist, 1)[-1].item()

        generated_char = idx_تعليم_char = idx_to_char[predicted_char_idx]  # إصلاح التسمية
        generated_text += idx_to_char[predicted_char_idx]

        input_seq = torch.tensor([[predicted_char_idx]], dtype=torch.long)

    return generated_text


# Generate 200 characters of text[cite: 1]
start_text = "To be, or not to be"
generated_text = generate_text(model, start_text, 200)
print("Task 4 completed - Generated Text:\n", generated_text)