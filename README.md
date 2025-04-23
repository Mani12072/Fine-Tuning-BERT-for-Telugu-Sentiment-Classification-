## Fine-Tuning-BERT-for-Telugu-Sentiment-Classification-
A deep learning project that fine-tunes a multilingual BERT model to classify Telugu text into sentiment categories. Built using Hugging Face Transformers and trained on publicly available datasets.

## Telugu Sentiment Analysis using BERT

This project involves **fine-tuning a BERT model for sentiment classification** on Telugu language text data. It demonstrates the full pipeline from data loading and preprocessing to model training and evaluation using Hugging Face's Transformers and Datasets libraries.

---

## 📌 Project Highlights

- **Language**: Telugu
- **Task**: Sentiment Classification
- **Model**: `google-bert/bert-base-multilingual-cased`
- **Dataset**: `Sanath369/Telugu_sentiment_sentences` (via Hugging Face Hub)
- **Evaluation Metrics**: Accuracy, F1 Score
- **Libraries**: Transformers, Datasets, Evaluate, Scikit-learn, PyTorch
I have written all the code in a single file, providing a step-by-step guide to create the model. The file covers everything from data preprocessing to model training and evaluation, with detailed comments explaining each step.

While I created a Streamlit app for deployment, the focus is on the coding file, which includes the model architecture, training pipeline, and evaluation procedures. The file serves as the core resource, and the Streamlit deployment is an extension of the logic within it.

This file is essential for understanding and replicating the entire model-building process, with the deployment handled through Hugging Face.

Link for Hugging Face:https://huggingface.co/spaces/Mpavan45/Telugu_Sentiment_Finetuning

# Telugu Sentiment Analysis with BERT (Streamlit Web Application)

Welcome to the **Telugu Sentiment Analysis** project – a complete end-to-end sentiment classification application built with **Streamlit** and powered by a fine-tuned **BERT** model hosted on **Hugging Face**.

> 🚀 I developed this Streamlit application from scratch, starting from model fine-tuning, uploading it to Hugging Face, and finally creating a fully interactive and visually appealing web application. All steps – from data processing, training, model deployment, and app design – were meticulously crafted to deliver an efficient Telugu sentiment classifier.

---

## 🎯 Overview

This web app classifies Telugu text input as **Positive**, **Neutral**, or **Negative** using a BERT-based model fine-tuned specifically for Telugu sentiment analysis. The interface supports user input, displays example sentences, validates for Telugu characters, and presents results with styled UI components including emojis and a beautiful background image.

---

## 💡 What This App Can Do

- 💬 Accepts Telugu text input manually or via preloaded examples.
- 🔎 Validates the input to ensure only Telugu script is processed.
- 📊 Displays sentiment result with user-friendly labels and emojis:
  - **Positive** 😊
  - **Neutral** 😐
  - **Negative** 😞
- 🎨 Offers a stylish UI with glowing (radium) effect titles and result tags.
- 🌄 Features a full-screen background image to enhance the user experience.
- 💾 Maintains session state to retain user input and output across sessions.

---

## 🧠 Model Details

- **Model Name**: [`Mpavan45/Telugu_Sentimental_Analysis`](https://huggingface.co/Mpavan45/Telugu_Sentimental_Analysis)
- **Model Type**: Fine-tuned BERT for text classification.
- **Hosted On**: Hugging Face Model Hub
- **Labels**:
  - `LABEL_0` – Negative 😞
  - `LABEL_1` – Neutral 😐
  - `LABEL_2` – Positive 😊
- **Fine-Tuning Details**: The model was fine-tuned on a labeled Telugu dataset, specifically optimized to handle the unique syntax, sentiment expressions, and contextual nuances of the Telugu language.

---

## 🔍 How It Works

1. **Input Processing**: The app tokenizes Telugu text and converts it into numerical embeddings.
2. **BERT Encoding**: Text is passed through multiple transformer layers in BERT to generate deep contextualized representations.
3. **Classification Head**: The final embedding is fed to a classification head that outputs one of the sentiment categories.
4. **Output Display**: The result is interpreted, mapped to emojis, and shown with stylish formatting.

---

## 🚀 Why BERT?

- 🧭 **Bidirectional Understanding**: Captures the meaning of a word based on both previous and next words.
- 🗣️ **Language Context Mastery**: Essential for interpreting Telugu tone and meaning.
- 💪 **Superior Performance**: Outperforms older RNN/LSTM models on NLP tasks like sentiment analysis.

---

## ⚠️ Limitations

- 🧠 High resource usage due to BERT architecture.
- 🈲 Only Telugu text accepted – enforced by regex-based character validation.
- 🤷 May misclassify sarcasm or highly ambiguous statements.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** – Web framework
- **Transformers (Hugging Face)** – For BERT model inference
- **Torch** – PyTorch backend
- **Regex** – Input validation

---

