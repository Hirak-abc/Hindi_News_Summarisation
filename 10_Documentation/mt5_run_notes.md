# mT5-small fine-tuning and prediction (Anshu)

## Model and data
- Model: google/mt5-small, fine-tuned from the pretrained checkpoint.
- Dataset: csebuetnlp/xlsum Hindi, loaded from the same parquet files as the Seq2Seq trainer.
- Training rows after removing blank article-summary pairs: 70,777 (validation: 8,847).
- Test set: 01_Dataset/processed/common_test_set.csv (8,847 rows), the same as all other models.

## Training settings
- Epochs: 2
- Batch size: 8
- Learning rate: 5e-4 (AdamW, weight decay 0.01)
- Max source length: 400 tokens; max target length: 100 tokens
- Precision: fp32 (fp16 avoided because mT5 is unstable in it)
- Seed: 42
- GPU: Tesla T4 (Kaggle), training time 148.3 minutes

## Training loss
- Epoch 1 validation loss: 3.81
- Epoch 2 validation loss: 3.66
- Training loss was still falling at the end, so 2 epochs is likely under-trained.

## Prediction settings
- Greedy decoding (num_beams=1), repetition penalty 1.5, max length 100.
- Prediction run on Kaggle: about 4 hours.

## Results (full 8,847-row test set, Unicode-aware HindiTokenizer)
- ROUGE-1 F1: 0.295085
- ROUGE-2 F1: 0.090911
- ROUGE-L F1: 0.231269

## Limitations
- One seed, two epochs, no hyperparameter search.
- Learning rate chosen as a starting value, not tuned.
- Hallucination and faithfulness not yet evaluated for mT5.
- The Kaggle session that trained the model did not keep its checkpoint, so the model cannot be re-run from saved weights; retraining is needed to reproduce it.
