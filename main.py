import torch
from bigram import BigramLanguageModel

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# set of unique characters in the data set
vocab = sorted(list(set(text)))
encoding_map = {char: i for i, char in enumerate(vocab)}
decoding_map = {i: char for i, char in enumerate(vocab)}


def encode(string: str, encoding_map: dict = encoding_map) -> list[int]:
    return [encoding_map[char] for char in string if char in encoding_map]


def decode(encoded_list: list[int], decoding_map: dict = decoding_map) -> str:
    return "".join([decoding_map[i] for i in encoded_list if i in decoding_map])


# encode the entire data set and store in a pytorch tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split the data into train and validation sets
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# training the model using chunks of the training data
# the chunk size is also the maximum context window for generation
context_window = 8
train_data[: context_window + 1]

x = train_data[:context_window]
y = train_data[1 : context_window + 1]

# for every chunk of training data we have a cumulative training example to predict the sequential character
for t in range(context_window):
    context = x[: t + 1]
    target = y[t]


batch_size = 4  # how many independent sequences will we process in parallel?


def get_batch(data):
    # get a random set of 4 indices from the data of size context window
    indices = torch.randint(len(data) - context_window, (batch_size,))

    # at each index create, extract the encoded char seq of length context window and stack them
    input_batch = torch.stack([data[i : i + context_window] for i in indices])
    # do the same for the targets
    target_batch = torch.stack([data[i + 1 : i + context_window + 1] for i in indices])
    return input_batch, target_batch


input_batch, target_batch = get_batch(train_data)

# for batch_index in range(batch_size):
#     for time_step in range(context_window):
#         context = input_batch[batch_index, : time_step + 1]
#         target = target_batch[batch_index, time_step]
#         print(f"when input is {context.tolist()} the target: {target}")

print(input_batch)

print(target_batch)

model = BigramLanguageModel(vocab_size=len(vocab))
logits, loss = model.forward(input_batch, target_batch)
print(f"shape: {logits.shape}")
print(f"loss: {loss}")

# initiate generation with a 0 encoded
print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_chars=100)[0].tolist()))
