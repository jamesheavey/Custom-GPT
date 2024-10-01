with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:1000])

# set of unique characters in the data set
chars = sorted(list(set(text)))
encoding_map = {char: i for i, char in enumerate(chars)}
decoding_map = {i: char for i, char in enumerate(chars)}


def encoder(string: str, encoding_map: dict = encoding_map) -> list[int]:
    return [encoding_map[char] for char in string if char in encoding_map]


def decoder(encoded_list: list[int], decoding_map: dict = decoding_map) -> str:
    return "".join([decoding_map[i] for i in encoded_list if i in decoding_map])


test_string = "First Citizen: Before we proceed any further, hear me speak."
encoded = encoder(test_string)
decoded = decoder(encoded)

print(f"Original: {test_string}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

assert test_string == decoded, "The decoded string does not match the original"
print("Test passed: The encoder and decoder work correctly.")
