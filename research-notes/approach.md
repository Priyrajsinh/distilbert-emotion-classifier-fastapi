# Research Notes — B1 HuggingFace Fine-Tuning
## Papers I Read Before Starting
- DistilBERT (Sanh et al. 2019) — 40% faster than BERT, 97% quality
- GoEmotions (Demszky et al. 2020) — 58K Reddit comments, 27 emotions
## Architecture Decisions
- distilbert-base-uncased: smaller, faster, strong classification baseline
- GoEmotions -> 7 macro-categories: reduces imbalance, cleaner confusion matrix
- HuggingFace Trainer over manual loop: focus on data quality + evaluation
## What I Would Do With More Compute
- Fine-tune RoBERTa-large for +3-5% F1 improvement
## Surprising Findings
- [Fill in after training — what was the actual val F1?]
