import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # number of sequences to process in parallel
context_window = 256  # maximum context length for predictions
max_iterations = 5000  # total number of training iterations
evaluation_interval = 500  # interval for evaluating the model
learning_rate = 3e-4  # learning rate for the optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if available
evaluation_iterations = 200  # number of iterations for evaluation
embedding_size = 384  # size of the embedding vectors
num_heads = 6  # number of attention heads
num_layers = 6  # number of transformer layers
dropout_rate = 0.2  # dropout rate for regularization
# ------------

# Read the text data
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a set of unique characters in the text
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
# Create mappings from characters to integers and vice versa
encoding_map = {char: i for i, char in enumerate(vocab)}
decoding_map = {i: char for i, char in enumerate(vocab)}


def encode(string: str) -> list[int]:
    # Convert a string to a list of integers based on the encoding map
    return [encoding_map[char] for char in string if char in encoding_map]


def decode(encoded_list: list[int]) -> str:
    # Convert a list of integers back to a string based on the decoding map
    return "".join([decoding_map[i] for i in encoded_list if i in decoding_map])


# Split the data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:train_size]
val_data = data[train_size:]


# Function to generate a batch of data
def get_batch(split):
    # Select the appropriate data split
    data = train_data if split == "train" else val_data
    # Generate random starting indices for the batch
    indices = torch.randint(len(data) - context_window, (batch_size,))
    # Create input and target batches
    input_batch = torch.stack([data[i : i + context_window] for i in indices])
    target_batch = torch.stack([data[i + 1 : i + context_window + 1] for i in indices])
    # Move batches to the appropriate device
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    return input_batch, target_batch


@torch.no_grad()
def estimate_loss():
    # Function to estimate the loss on training and validation sets
    losses = {}
    model.eval()
    for split in ["train", "val"]:
        split_losses = torch.zeros(evaluation_iterations)
        for k in range(evaluation_iterations):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean()
    model.train()
    return losses


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_window, context_window)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Compute key, query, and value matrices
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        # Compute attention scores
        attention_scores = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        attention_scores = F.softmax(attention_scores, dim=-1)  # (B, T, T)
        attention_scores = self.dropout(attention_scores)
        # Compute weighted sum of values
        v = self.value(x)  # (B,T,head_size)
        out = attention_scores @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Concatenate the outputs of all attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, embedding_size, num_heads):
        # embedding_size: embedding dimension, num_heads: number of attention heads
        super().__init__()
        head_size = embedding_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(embedding_size)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        # Apply self-attention and add residual connection
        x = x + self.self_attention(self.layer_norm1(x))
        # Apply feed-forward network and add residual connection
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Token and position embedding tables
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(context_window, embedding_size)
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(embedding_size, num_heads=num_heads) for _ in range(num_layers)])
        self.layer_norm_final = nn.LayerNorm(embedding_size)  # final layer norm
        self.language_model_head = nn.Linear(embedding_size, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Initialize weights of the model
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Get token and position embeddings
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_embeddings + position_embeddings  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.layer_norm_final(x)  # (B,T,C)
        logits = self.language_model_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Generate new tokens from the model
        for _ in range(max_new_tokens):
            # Crop idx to the last context_window tokens
            idx_cond = idx[:, -context_window:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTLanguageModel()
model = model.to(device)
# Print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iteration in range(max_iterations):

    # Evaluate the loss on train and val sets at regular intervals
    if iteration % evaluation_interval == 0 or iteration == max_iterations - 1:
        losses = estimate_loss()
        print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    input_batch, target_batch = get_batch("train")

    # Evaluate the loss
    logits, loss = model(input_batch, target_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
