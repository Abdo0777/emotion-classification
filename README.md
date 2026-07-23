# 💜 Emotion Classification 

A multi-class text classification project that identifies **6 core emotions** — joy, sadness, anger, fear, surprise, and disgust — using sequence models with attention mechanisms, benchmarked against a fine-tuned Transformer.

---

## 🎯 Objectives

- Classify text into 6 emotion categories using deep learning
- Apply attention mechanisms over LSTM/GRU hidden states to improve accuracy
- Visualize attention weights to explain which words drive each prediction
- Use pre-trained GloVe embeddings for richer word representations
- Fine-tune a Transformer (DistilBERT) for emotion classification using HuggingFace

---

## 🗂️ Dataset

- **Source:** [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) by Google (28 fine-grained emotions + neutral)
- Raw labels mapped to 6 target classes using the standard Ekman mapping
- Only single-label examples were kept, to simplify the classification problem

---

## 🏗️ Models Compared

Four different architectures were built and trained:

| Model | Type | Word Representation |
|---|---|---|
| LSTM | Built from scratch | GloVe (100d) |
| GRU | Built from scratch | GloVe (100d) |
| BiLSTM + Attention | Built from scratch (custom attention layer) | GloVe (100d) |
| DistilBERT | Fine-tuned | Contextual Embeddings |

---

## 📊 Test Set Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **DistilBERT** | **78.55%** | **0.687** |
| BiLSTM + Attention | 70.98% | 0.611 |
| GRU | 71.65% | 0.603 |
| LSTM | 71.55% | 0.594 |

**Key takeaways:**
- **DistilBERT** achieved the best performance on both metrics by a clear margin, thanks to its deep contextual text representations
- **BiLSTM + Attention** outperformed both LSTM and GRU on Macro F1 despite similar accuracy — indicating better balance across classes, especially the underrepresented ones
- All models were evaluated using: accuracy, macro F1, per-class precision/recall, and confusion matrix

---

## 🔍 Interpretability (Attention Visualization)

Using the attention weights from the BiLSTM + Attention model, heatmaps were generated to show which words most influenced the model's prediction for each sentence — providing transparency into why each classification was made.

---

## 🚀 Deployment

The best-performing model (**DistilBERT**) was deployed as an interactive web app using **Gradio**. Users enter free text and receive:

- The dominant emotion, with an emoji indicator
- A confidence bar chart across all 6 emotion classes

### Running locally

```bash
git clone https://github.com/your-username/emotion-classification-attention.git
cd emotion-classification-attention

python -m venv venv
venv\Scripts\activate        # on Windows
# source venv/bin/activate   # on Linux/Mac

pip install -r requirements.txt
python app.py
```

Once running, open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

---

## 📁 Project Structure

```
emotion-classification-attention/
├── app.py                      # Interactive Gradio app
├── requirements.txt            # Required packages
├── notebooks/
│   └── emotion_classification.ipynb
├── distilbert_emotion_final/   # Trained model (or hosted on HuggingFace Hub)
└── README.md
```

---

## 🛠️ Tech Stack

- **Python**, **TensorFlow/Keras** (LSTM, GRU, custom BiLSTM + Attention)
- **HuggingFace Transformers** (DistilBERT + Trainer API)
- **GloVe** Word Embeddings
- **Gradio** for deployment and UI
- **scikit-learn** for evaluation (classification report, confusion matrix)

---

## 📌 Full Pipeline

1. Load and explore the GoEmotions dataset
2. Map raw labels to the 6 target emotion classes
3. Clean and tokenize text, pad sequences
4. Build an embedding layer initialized with pre-trained GloVe vectors
5. Build and train LSTM
6. Build and train GRU
7. Build and train custom BiLSTM + Attention
8. Fine-tune DistilBERT via the HuggingFace Trainer API
9. Full evaluation and comparison across all four models
10. Visualize attention heatmaps
11. Save models for deployment
12. Build and deploy the interactive Gradio app

---

