# Patent Phrase Similarity with Transformer

This project is an NLP semantic similarity model built with **Hugging Face Transformers**.
The model predicts how similar two patent-related phrases are within a specific technical context.

## Project Overview

The goal of this project is to estimate the semantic similarity between an `anchor` phrase and a `target` phrase using a Transformer-based model.

The model receives structured text input in this format:

```text
Anchor: <anchor phrase> [SEP] Target: <target phrase> [SEP] Context: <technical context>
```

and predicts a similarity score between **0.0** and **1.0**.

## Example

```text
Anchor: battery cell
Target: power storage unit
Context: H01
Predicted Similarity Score: 0.78
```

## Dataset

The dataset contains patent phrase pairs with similarity scores.

Main columns used:

* `anchor`
* `target`
* `context`
* `score`

The `score` column is used as the target label.

Score values:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

## Task Type

This project is a **regression-like NLP task**.

Instead of predicting a class label, the model predicts a numerical similarity score.

## Model

The project uses a pretrained Transformer model:

```text
microsoft/deberta-v3-small
```

The model was fine-tuned using:

```python
AutoModelForSequenceClassification
```

with:

```python
num_labels=1
problem_type="regression"
```

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Hugging Face Datasets
* Hugging Face Transformers
* PyTorch
* Google Colab

## Project Pipeline

```text
Dataset
→ Data analysis
→ Input text construction
→ Train/validation split
→ Tokenization
→ Transformer model
→ Training
→ Evaluation
→ Prediction
```

## Key NLP Concepts Used

* Tokenization
* Input IDs
* Attention Mask
* Transformer Fine-Tuning
* Sequence Regression
* Semantic Similarity
* Pearson Correlation
* Train/Validation Split

## Evaluation

The model was evaluated using:

* Validation loss
* Pearson correlation
* Prediction error analysis

Pearson correlation is useful in this project because the goal is to measure how well the predicted scores follow the real similarity scores.

## Prediction Function

```python
def predict_similarity(anchor, target, context):
    text = f"Anchor: {anchor} [SEP] Target: {target} [SEP] Context: {context}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model(**inputs)
    pred = output.logits.item()

    pred = max(0, min(1, pred))
    return pred
```

## What I Learned

Through this project, I learned how to:

* Work with NLP datasets
* Build structured text input from multiple columns
* Use Hugging Face tokenizers
* Convert text into model-ready features
* Fine-tune a Transformer model
* Handle a regression-like NLP problem
* Evaluate model predictions using Pearson correlation
* Build an end-to-end NLP pipeline

## Future Improvements

* Try larger Transformer models
* Add better context descriptions instead of raw context codes
* Compare regression vs classification approaches
* Tune learning rate and batch size
* Add a Gradio demo interface
* Deploy the model on Hugging Face Spaces

## Author

This project was built as part of my Machine Learning and NLP learning journey.
