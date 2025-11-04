
markdown
# Fine-Tuning DistilBERT for Sentiment Analysis on IMDB Reviews

This project demonstrates how to fine-tune a pre-trained DistilBERT model for binary sentiment analysis using the IMDB movie reviews dataset. The model classifies reviews as either positive or negative.

## 📋 Project Overview

- **Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Binary sentiment classification (Positive/Negative)
- **Dataset**: IMDB Movie Reviews (50,000 reviews)
- **Framework**: Hugging Face Transformers + PyTorch
- **Training**: Fine-tuning on balanced dataset (8,000 samples)
- **Evaluation**: Accuracy, Precision, Recall, F1-score

## 🚀 Features

- Data preprocessing and tokenization
- Balanced dataset creation to prevent bias
- GPU-accelerated training
- Comprehensive model evaluation
- Prediction interface for new reviews
- Performance metrics visualization

## 🛠️ Installation & Dependencies

```bash
pip install transformers datasets evaluate torch
```

## 📁 Project Structure

```
Distilbert_sentimental_movie_review.ipynb
├── Dataset Loading & Exploration
├── Data Preprocessing & Balancing
├── Tokenization
├── Model Initialization
├── Training Configuration
├── Model Training
├── Evaluation
└── Prediction Interface
```

## 🏗️ Implementation Details

### 1. Data Preparation
- Loads IMDB dataset with 50,000 movie reviews
- Creates balanced training set (4,000 positive + 4,000 negative)
- Implements data shuffling for better generalization

### 2. Tokenization
- Uses DistilBERT tokenizer (distilbert-base-uncased)
- Maximum sequence length: 256 tokens
- Automatic padding and truncation

### 3. Model Architecture
- **Base Model**: DistilBERT (66 million parameters)
- **Classification Head**: 2-class linear layer
- **Output**: Positive (1) / Negative (0) sentiment

### 4. Training Configuration
```python
TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch"
)
```

### 5. Evaluation Metrics
- Accuracy
- Precision
- Recall 
- F1-Score

## 📊 Results

The model achieves the following performance on the test set:

| Metric | Score |
|--------|-------|
| Accuracy | ~92% |
| F1-Score | ~92% |
| Precision | ~92% |
| Recall | ~92% |

## 💻 Usage

### Making Predictions

```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    
    sentiment = "Positive 😀" if prediction == 1 else "Negative 😠"
    confidence = probs[0][prediction].item()
    
    return sentiment, confidence

# Example usage
review = "This movie was absolutely fantastic! Great acting and storyline."
sentiment, confidence = predict_sentiment(review)
print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
```

### Sample Output
```
Sample Review 1:
Review: "This film captures the essence of storytelling with brilliant performances..."
Prediction: Positive 😀 (Confidence: 0.95)

Sample Review 2:
Review: "A disappointing attempt that fails to deliver on its promising premise..."
Prediction: Negative 😠 (Confidence: 0.89)
```

## 🔧 Technical Specifications

- **Model Size**: ~268MB
- **Training Time**: ~10-15 minutes on GPU (Google Colab T4)
- **Inference Time**: ~10-50ms per review
- **Vocabulary Size**: 30,522 tokens
- **Layers**: 6 transformer layers
- **Hidden Size**: 768 dimensions
- **Attention Heads**: 12

## 🎯 Key Features

1. **Efficient Training**: Uses DistilBERT for faster training compared to BERT-base
2. **Bias Mitigation**: Balanced dataset prevents model bias
3. **GPU Optimization**: Automatic device detection (CPU/GPU)
4. **Comprehensive Metrics**: Multiple evaluation metrics for thorough analysis
5. **Production Ready**: Easy-to-use prediction interface

## 📈 Performance Optimization

- **Batch Processing**: Efficient handling of multiple reviews
- **Sequence Truncation**: Optimal balance between context and computation
- **Mixed Precision**: Potential for FP16 training on supported hardware
- **Gradient Accumulation**: Stable training with effective batch sizes

## 🔮 Future Enhancements

- [ ] Hyperparameter tuning with Optuna
- [ ] Ensemble methods with multiple models
- [ ] Deploy as REST API with FastAPI
- [ ] Add multi-language support
- [ ] Implement model interpretability (LIME/SHAP)

## 📚 References

1. [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
2. [Hugging Face Transformers](https://huggingface.co/docs/transformers)
3. [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## 👥 Contributors

- Developed as an educational project for NLP and sentiment analysis

## 📄 License

This project is intended for educational purposes. Please check the original licenses for DistilBERT and the IMDB dataset for commercial use.
```
