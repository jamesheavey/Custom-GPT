with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:1000])

# set of unique characters in the data set
chars = sorted(list(set(text)))
char_map = {char: i for i, char in enumerate(chars)}


def encoder(string: str, encoding_map: dict = char_map) -> list[int]:
    return [encoding_map[char] for char in string]


def decoder(encoded_list: list[int], decoding_map: dict = {i: char for char, i in char_map.items()}) -> str:
    return "".join([decoding_map[i] for i in encoded_list])


test_string = "First Citizen: Before we proceed any further, hear me speak."
encoded = encoder(test_string)
decoded = decoder(encoded)

print(f"Original: {test_string}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

assert test_string == decoded, "The decoded string does not match the original"
print("Test passed: The encoder and decoder work correctly.")
