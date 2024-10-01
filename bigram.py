import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32  # how many independent sequences will we process in parallel?
# the chunk size is also the maximum context window for generation
context_window = 8
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# set of unique characters in the data set
vocab = sorted(list(set(text)))
encoding_map = {char: i for i, char in enumerate(vocab)}
decoding_map = {i: char for i, char in enumerate(vocab)}


def encode(string: str) -> list[int]:
    return [encoding_map[char] for char in string if char in encoding_map]


def decode(encoded_list: list[int]) -> str:
    return "".join([decoding_map[i] for i in encoded_list if i in decoding_map])


# encode the entire data set and store in a pytorch tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Adjust context_window and batch_size if necessary
if len(data) < context_window * 2:
    context_window = len(data) // 4
    batch_size = max(1, context_window // 2)
    print(f"Adjusted context_window to {context_window} and batch_size to {batch_size} due to small dataset.")

# Split the data into train and validation sets
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# training the model using chunks of the training data

train_data[: context_window + 1]

x = train_data[:context_window]
y = train_data[1 : context_window + 1]

# for every chunk of training data we have a cumulative training example to predict the sequential character
for t in range(context_window):
    context = x[: t + 1]
    target = y[t]


def get_batch(data):
    # Ensure that we have enough data to create a batch
    if len(data) <= context_window:
        raise ValueError("Data length must be greater than context_window")

    # Calculate the maximum valid index
    max_index = len(data) - context_window

    # Generate random indices
    indices = torch.randint(0, max_index, (batch_size,))

    # Create input and target batches
    input_batch = torch.stack([data[i : i + context_window] for i in indices])
    target_batch = torch.stack([data[i + 1 : i + context_window + 1] for i in indices])

    return input_batch.to(device), target_batch.to(device)


input_batch, target_batch = get_batch(train_data)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Create a lookup table that predicts the next token for each input token
        self.token_prediction_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_sequence, target_sequence=None):
        # input_sequence: current sequence of token indices
        # target_sequence: expected next tokens (optional)

        # Predict the next token for each input token
        next_token_predictions = self.token_prediction_table(
            input_sequence
        )  # Shape: (Batch, Sequence_Length, Vocabulary_Size)
        batch_size, sequence_length, vocabulary_size = next_token_predictions.shape
        # Reshape predictions for loss calculation
        flattened_predictions = next_token_predictions.view(batch_size * sequence_length, vocabulary_size)

        prediction_loss = None
        if target_sequence is not None:
            # If target sequence is provided, calculate the prediction loss
            flattened_targets = target_sequence.view(batch_size * sequence_length)
            prediction_loss = F.cross_entropy(flattened_predictions, flattened_targets)

        return flattened_predictions, prediction_loss

    def generate(self, initial_sequence, num_new_tokens):
        # initial_sequence: starting sequence of token indices
        # num_new_tokens: number of new tokens to generate

        vocabulary_size = self.token_prediction_table.num_embeddings
        current_sequence = initial_sequence

        for _ in range(num_new_tokens):
            # Ensure all token indices are within the valid range
            current_sequence = current_sequence % vocabulary_size

            # Predict the next token
            next_token_predictions, _ = self.forward(current_sequence)

            # Focus on the prediction for the last token in the sequence
            if len(next_token_predictions.shape) == 3:
                last_token_prediction = next_token_predictions[:, -1, :]  # For batched input
            else:
                last_token_prediction = next_token_predictions[:, -1]  # For single sample input

            # Convert predictions to probabilities
            next_token_probabilities = F.softmax(last_token_prediction, dim=-1)

            # Randomly select the next token based on the probabilities
            next_token = torch.multinomial(next_token_probabilities, num_samples=1)

            # Add the new token to the sequence
            current_sequence = torch.cat((current_sequence, next_token.unsqueeze(0)), dim=1)

        return current_sequence[0].tolist()  # Return the generated sequence


model = BigramLanguageModel(vocab_size=len(vocab)).to(device)
logits, loss = model.forward(input_batch, target_batch)
print(f"shape: {logits.shape}")
print(f"loss: {loss}")


optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    data = {"train": train_data, "val": val_data}
    for key, d in data.items():  # Use .items() to iterate over key-value pairs
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(d)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[key] = losses.mean()
    model.train()
    return out


for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    input_batch, target_batch = get_batch(train_data)

    # evaluate the loss
    logits, loss = model.forward(input_batch, target_batch)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()


# initiate generation with a " " encoded
print(decode(model.generate(initial_sequence=torch.tensor([encode(" ")], dtype=torch.long), num_new_tokens=100)))
