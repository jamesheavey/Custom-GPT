import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)

        loss = None

        if targets is not None:
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_chars):
        # idx is (B, T) array of indices in the current context
        vocab_size = self.token_embedding_table.num_embeddings
        for _ in range(max_new_chars):
            # Ensure idx is within the valid range
            idx = idx % vocab_size
            # get the predictions
            logits, loss = self.forward(idx)

            # focus only on the last time step
            if len(logits.shape) == 3:
                logits = logits[:, -1, :]  # becomes (B, C)
            else:
                logits = logits[:, -1]  # becomes (C,) for single sample

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C) or (C,)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) or (1,)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next.unsqueeze(0)), dim=1)  # (B, T+1)
        return idx
