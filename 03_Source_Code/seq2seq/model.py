"""Attention-based sequence-to-sequence model for Hindi summarisation."""
# This docstring explains that the file defines the Seq2Seq neural network.

from __future__ import annotations
# Allow forward references and modern type annotations without runtime evaluation.

import random
# Import random so each training step can choose teacher-forced or predicted tokens.

import torch
# Import PyTorch for tensors and neural-network execution.
from torch import nn
# Import the module classes used to construct the neural-network layers.
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# Import helpers that process batches of variable-length sequences efficiently.


class Seq2SeqAttention(nn.Module):
    """Bidirectional LSTM encoder with additive attention and an LSTM decoder."""
    # Describe the complete encoder-attention-decoder architecture.

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        pad_id: int,
        dropout: float = 0.2,
    ) -> None:
        # Create embeddings, bidirectional encoder, attention, and decoder layers.
        super().__init__()
        # Initialize the base PyTorch neural-network module.
        self.pad_id = pad_id
        # Store the padding token ID for source-mask construction.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        # Create a token embedding table that ignores padding in embedding operations.
        self.encoder = nn.LSTM(
            # Create the bidirectional source-sequence encoder.
            embedding_dim,
            # Set the input embedding size.
            hidden_size,
            # Set the encoder hidden state size.
            batch_first=True,
            # Make batch dimensions the first dimension of input tensors.
            bidirectional=True,
            # Process source text forward and backward.
        )
        self.initial_hidden = nn.Linear(hidden_size * 2, hidden_size)
        # Map the final encoder hidden pair into the decoder hidden state.
        self.initial_cell = nn.Linear(hidden_size * 2, hidden_size)
        # Map the final encoder cell pair into the decoder cell state.
        self.attention_encoder = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # Transform encoder states into the attention compatibility space.
        self.attention_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        # Transform the decoder state into the same attention compatibility space.
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        # Convert each compatibility vector into one scalar attention score.
        self.decoder = nn.LSTM(
            # Create the decoder LSTM that produces summary states.
            embedding_dim + hidden_size * 2,
            # Combine the embedded target token and attended context vector.
            hidden_size,
            # Set the decoder hidden state size.
            batch_first=True,
            # Make batch dimensions the first tensor dimension.
        )
        self.dropout = nn.Dropout(dropout)
        # Create dropout regularization for embeddings and decoder input.
        self.output = nn.Linear(hidden_size * 3, vocab_size)
        # Map decoder output and context to logits for every vocabulary token.

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode padded source tokens and initialize the decoder state.
        mask = source.ne(self.pad_id)
        # Mark real source positions as true and padding positions as false.
        embedded = self.dropout(self.embedding(source))
        # Convert source IDs to embeddings and apply dropout.
        lengths = mask.sum(dim=1).cpu()
        # Count real tokens in each source sequence for packed processing.
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        # Pack sequences together while excluding padding from LSTM computation.
        packed_encoded, (hidden, cell) = self.encoder(packed)
        # Run the bidirectional encoder and obtain states and hidden values.
        encoded, _ = pad_packed_sequence(
            packed_encoded, batch_first=True, total_length=source.size(1)
        )
        # Restore all encoder states to the original padded batch shape.
        summary_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        # Concatenate forward and backward final encoder hidden states.
        summary_cell = torch.cat((cell[-2], cell[-1]), dim=-1)
        # Concatenate forward and backward final encoder cell states.
        decoder_hidden = torch.tanh(self.initial_hidden(summary_hidden)).unsqueeze(0)
        # Project and activate the encoder summary into the decoder hidden state.
        decoder_cell = torch.tanh(self.initial_cell(summary_cell)).unsqueeze(0)
        # Project and activate the encoder summary into the decoder cell state.
        return encoded, mask, decoder_hidden, decoder_cell
        # Return encoded states, padding mask, and initial decoder states.

    def decode_step(
        self,
        token: torch.Tensor,
        encoded: torch.Tensor,
        source_mask: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attend to encoder states and predict the next-token logits.
        # Mask padding before softmax so attention only uses real source tokens.
        query = self.attention_decoder(hidden[-1]).unsqueeze(1)
        # Transform the latest decoder state and add a position-compatibility axis.
        energy = self.attention_score(
            torch.tanh(self.attention_encoder(encoded) + query)
        ).squeeze(-1)
        # Calculate one additive-attention score for every source position.
        energy = energy.masked_fill(~source_mask, torch.finfo(energy.dtype).min)
        # Replace padding positions with the lowest representable finite value.
        attention = torch.softmax(energy, dim=-1)
        # Normalize source attention scores into a probability distribution.
        context = torch.bmm(attention.unsqueeze(1), encoded)
        # Build a weighted sum of encoder states for the current decoder step.
        decoder_input = torch.cat((self.dropout(self.embedding(token)), context), dim=-1)
        # Combine the target token embedding and attention context.
        decoded, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
        # Run the decoder LSTM once and update hidden and cell states.
        logits = self.output(
            torch.cat((decoded, context), dim=-1)
        ).squeeze(1)
        # Flatten the decoder/context output and produce vocabulary logits.
        return logits, hidden, cell
        # Return logits and the updated decoder states.

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Return token logits for targets after the initial BOS token."""
        # Explain that forward mode returns logits for every target position after BOS.
        encoded, source_mask, hidden, cell = self.encode(source)
        # Encode sources and create the initial decoder states.
        token = target[:, 0]
        # Start decoding with the target batch's BOS token.
        outputs: list[torch.Tensor] = []
        # Create an empty list for logits from each target position.
        for step in range(1, target.size(1)):
            # Iterate through every target position after BOS.
            logits, hidden, cell = self.decode_step(
                token.unsqueeze(1), encoded, source_mask, hidden, cell
            )
            # Produce logits and update the decoder using the current token.
            outputs.append(logits)
            # Save logits for this target position.
            # Teacher forcing feeds the gold token; otherwise feed this step's prediction.
            use_teacher = random.random() < teacher_forcing_ratio
            # Randomly decide whether to use teacher forcing.
            token = target[:, step] if use_teacher else logits.argmax(dim=-1)
            # Use the gold token or the model's predicted token for the next step.
        return torch.stack(outputs, dim=1)
        # Stack all step logits into one tensor for the loss calculation.

    @torch.inference_mode()
    def generate(
        self,
        source: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_length: int,
    ) -> torch.Tensor:
        """Greedily decode token IDs, including BOS but excluding EOS."""
        # Explain that generation is greedy and begins with BOS.
        encoded, source_mask, hidden, cell = self.encode(source)
        # Encode all source articles and initialize decoder states.
        token = torch.full(
            (source.size(0),), bos_id, dtype=torch.long, device=source.device
        )
        # Create one BOS starting token for every source article.
        generated = [token]
        # Initialize generated sequences with BOS IDs.
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        # Create a flag that records which examples have emitted EOS.
        for _ in range(max_length):
            # Decode up to the requested maximum summary length.
            logits, hidden, cell = self.decode_step(
                token.unsqueeze(1), encoded, source_mask, hidden, cell
            )
            # Predict logits for the next token.
            token = logits.argmax(dim=-1)
            # Choose the highest-logit token for each batch item.
            token = torch.where(finished, torch.full_like(token, eos_id), token)
            # Force EOS for batch items that already finished.
            generated.append(token)
            # Append each selected token to the generated sequence.
            finished |= token.eq(eos_id)
            # Mark batch items that emitted EOS as finished.
            if finished.all():
                break
                # Stop when every sequence has finished.
        return torch.stack(generated, dim=1)
        # Stack generated token IDs into a two-dimensional batch tensor.
