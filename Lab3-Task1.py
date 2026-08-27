# Terminal command to install torch (Run if not installed):
# pip install torch

import torch
import numpy as np

# Sample text data (Shakespeare)
text = """To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles..."""

# Create character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings for characters to indices and vice versa
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert the text to a sequence of integers
data = np.array([char_to_idx[c] for c in text])

print("Task 1 completed successfully!")
print("Vocab Size:", vocab_size)ٍ