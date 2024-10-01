import torch

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:1000])

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

print(data.shape, data.dtype)

print(data[:1000])


# Split the data into train and validation sets
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
