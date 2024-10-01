import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocabulary_size):
        super().__init__()
        # Create a lookup table that predicts the next token for each input token
        self.token_prediction_table = nn.Embedding(vocabulary_size, vocabulary_size)

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

        return current_sequence  # Return the generated sequence
