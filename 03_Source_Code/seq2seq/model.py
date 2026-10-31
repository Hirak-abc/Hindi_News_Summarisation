"""Attention-based sequence-to-sequence model for Hindi summarisation."""

from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2SeqAttention(nn.Module):
    """Bidirectional LSTM encoder with additive attention and an LSTM decoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        pad_id: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.initial_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.initial_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_encoder = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.attention_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        self.decoder = nn.LSTM(
            embedding_dim + hidden_size * 2,
            hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size * 3, vocab_size)

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = source.ne(self.pad_id)
        embedded = self.dropout(self.embedding(source))
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_encoded, (hidden, cell) = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_encoded, batch_first=True, total_length=source.size(1)
        )
        summary_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        summary_cell = torch.cat((cell[-2], cell[-1]), dim=-1)
        decoder_hidden = torch.tanh(self.initial_hidden(summary_hidden)).unsqueeze(0)
        decoder_cell = torch.tanh(self.initial_cell(summary_cell)).unsqueeze(0)
        return encoded, mask, decoder_hidden, decoder_cell

    def decode_step(
        self,
        token: torch.Tensor,
        encoded: torch.Tensor,
        source_mask: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.attention_decoder(hidden[-1]).unsqueeze(1)
        energy = self.attention_score(
            torch.tanh(self.attention_encoder(encoded) + query)
        ).squeeze(-1)
        energy = energy.masked_fill(~source_mask, torch.finfo(energy.dtype).min)
        attention = torch.softmax(energy, dim=-1)
        context = torch.bmm(attention.unsqueeze(1), encoded)
        decoder_input = torch.cat((self.dropout(self.embedding(token)), context), dim=-1)
        decoded, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
        logits = self.output(
            torch.cat((decoded, context), dim=-1)
        ).squeeze(1)
        return logits, hidden, cell

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Return token logits for targets after the initial BOS token."""
        encoded, source_mask, hidden, cell = self.encode(source)
        token = target[:, 0]
        outputs: list[torch.Tensor] = []
        for step in range(1, target.size(1)):
            logits, hidden, cell = self.decode_step(
                token.unsqueeze(1), encoded, source_mask, hidden, cell
            )
            outputs.append(logits)
            use_teacher = random.random() < teacher_forcing_ratio
            token = target[:, step] if use_teacher else logits.argmax(dim=-1)
        return torch.stack(outputs, dim=1)

    @torch.inference_mode()
    def generate(
        self,
        source: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_length: int,
    ) -> torch.Tensor:
        """Greedily decode token IDs, including BOS but excluding EOS."""
        encoded, source_mask, hidden, cell = self.encode(source)
        token = torch.full(
            (source.size(0),), bos_id, dtype=torch.long, device=source.device
        )
        generated = [token]
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for _ in range(max_length):
            logits, hidden, cell = self.decode_step(
                token.unsqueeze(1), encoded, source_mask, hidden, cell
            )
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, eos_id), token)
            generated.append(token)
            finished |= token.eq(eos_id)
            if finished.all():
                break
        return torch.stack(generated, dim=1)
