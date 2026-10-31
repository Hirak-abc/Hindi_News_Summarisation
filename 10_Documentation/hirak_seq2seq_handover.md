# Hirak — Seq2Seq with Attention Handover

## Status

**Hirak's assigned Seq2Seq deliverables are complete.** The model was implemented, trained and validated, its checkpoint was saved, and test-set predictions and evaluation results were produced. No further training or quality-improvement work is required to meet the Seq2Seq task as assigned.

The summary-quality limitations below should be disclosed in the handover. Improving them further is optional follow-up work, not a missing assignment deliverable. The mT5/IndicBART work belongs to Anshu; the other project areas belong to their assigned team members.

## Completed deliverables

- [Seq2Seq attention notebook](../02_Notebooks/02_Seq2Seq_Attention.ipynb)
- [Attention model implementation](../03_Source_Code/seq2seq/model.py)
- [Training and validation code](../03_Source_Code/seq2seq/train.py)
- [Prediction and ROUGE evaluation code](../03_Source_Code/seq2seq/predict.py)
- [Saved best-model checkpoint](../04_Models/seq2seq_attention/checkpoints/best_model.pt)
- [Recorded training settings](../04_Models/seq2seq_attention/training_settings.json)
- Original greedy test predictions: [seq2seq_predictions.csv](../05_Predictions/seq2seq_predictions.csv)
- Existing repetition-penalty test predictions: [seq2seq_predictions_penalty_1.5.csv](../05_Predictions/seq2seq_predictions_penalty_1.5.csv)
- Full-test predictions with trigram blocking: [seq2seq_predictions_full_trigram_penalty_1.5.csv](../05_Predictions/seq2seq_predictions_full_trigram_penalty_1.5.csv)
- Existing ROUGE results: [seq2seq_penalty_1.5_rouge.csv](../06_Evaluation_Results/seq2seq_penalty_1.5_rouge.csv)
- Corrected-tokenizer evaluation and decoding experiment details: [decoding_experiment.md](./decoding_experiment.md)

The saved settings record 10 epochs, batch size 32, embedding size 256, hidden size 256, dropout 0.2, learning rate 0.001, vocabulary size up to 30,000, source limit 400, and target limit 100. The best validation loss reported for training was 3.9584 at epoch 6. The checkpoint was not changed during the local decoding experiments.

## Evaluation summary

The full agreed test set contains 8,847 articles. Corrected Hindi-aware ROUGE F1 values are:

| Decoding | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---:|---:|---:|
| Existing greedy decoding | 0.237014 | 0.074970 | 0.203441 |
| Repetition penalty 1.5 | 0.296962 | 0.080844 | 0.227799 |
| Repetition penalty 1.5 + trigram blocking | 0.297023 | 0.080860 | 0.227831 |

The repetition penalty improved the greedy baseline. Trigram blocking changed 23 of 8,847 predictions and removed exact repeated model-token trigrams, but its ROUGE increase over penalty 1.5 was negligible. The original prediction and evaluation files were preserved.

## Limitations to report

- Generated text can be incoherent, factually incorrect, or unrelated to the article.
- The full-test trigram-blocked outputs contain `<unk>` in 2,192 summaries.
- Trigram blocking only prevents exact repeated three-token phrases; it does not fix factuality, single-word repetition, or coherence.
- ROUGE measures lexical overlap and does not establish factual faithfulness or readability. The evaluation tokenizer was corrected to preserve Devanagari combining marks; use the corrected scores in reporting.
- A small manual inspection found mixed quality and is not a formal human evaluation.

## Handover decision

**The assigned Seq2Seq task is done. No additional fine-tuning or decoding change is required for this handover.** Keep the limitations visible in the final report. Further model-quality work can be proposed as a separate experiment if the team chooses, but it is not required to complete Hirak's listed responsibilities.
