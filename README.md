<h1 align="center">🧠 NLP Text Classification & Spam Detection</h1>

<p align="center">
  Academic project developed in <strong>Python</strong> focused on natural language processing,
  text representation, supervised classification, and evaluation under class imbalance.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NLP-Text%20Classification-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Models-Naive%20Bayes%20%7C%20SVM%20%7C%20RF-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Features-BoW%20%7C%20TF--IDF-lightgrey?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Evaluation-F1--Macro%20%7C%20Imbalance-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Tasks-Spam%20Detection%20%7C%20News%20Classification-blueviolet?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://deepwiki.com/maria2332/nlp-text-classification-and-spam-detection" target="_blank">
    <img src="https://img.shields.io/badge/DeepWiki-Documentation-purple?style=for-the-badge"/>
  </a>
</p>

---

# 🧠 NLP Text Classification & Spam Detection

Academic project focused on **text classification and spam detection** using classical Natural Language Processing (NLP) techniques and machine learning models in Python.

The project explores how different **text representations**, **classification algorithms**, and **dataset balance strategies** affect performance, with a strong emphasis on **statistical rigor and interpretability**.

---

## 🎓 Academic Context

This project was developed as part of a **Natural Language Processing (NLP)** course within the **Degree in Mathematical Engineering**.

The objective is not only to build accurate classifiers, but to **analyze model behavior**, compare approaches, and understand the impact of **class imbalance** and **data preprocessing**.

---

## 📌 Tasks Addressed

### 1️⃣ Spam Detection
Binary text classification task:
- **Spam**
- **Ham (non-spam)**

### 2️⃣ News Classification
Multiclass text classification task with the following categories:
- **Deportes**
- **Internacional**
- **Nacional**
- **Sociedad**

---

## 🗂️ Dataset Configurations

The experiments were conducted under different data scenarios:

- **Balanced dataset**  
  Classes artificially balanced by duplicating samples.

- **Unbalanced dataset**  
  Duplicates removed, preserving the original (slightly imbalanced) class distribution.

- **Imbalance-aware evaluation**  
  Special focus on **macro-averaged metrics** to ensure fair comparison across classes.

---

## 🧪 Models Evaluated

The following models and representations were compared:

### 🔤 Text Representations
- **Bag of Words (BoW)**
- **TF-IDF**

### 🤖 Classification Models
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Linear SVM (LinearSVC)**
- **Random Forest**

All models share the same preprocessing pipeline to ensure comparability.

---

## 📊 Results on the Unbalanced News Dataset

| Model                     | Representation | Accuracy | Precision (macro) | Recall (macro) | F1-macro |
|---------------------------|----------------|----------|-------------------|----------------|----------|
| Naive Bayes (Multinomial) | BoW            | 0.9859   | 0.9875            | 0.9875         | 0.9872   |
| Random Forest             | TF-IDF         | 0.9718   | 0.9742            | 0.9737         | 0.9729   |
| Linear SVM (LinearSVC)    | TF-IDF         | 0.9718   | 0.9737            | 0.9743         | 0.9729   |
| Logistic Regression       | TF-IDF         | 0.9577   | 0.9597            | 0.9612         | 0.9601   |
| Naive Bayes (Multinomial) | TF-IDF         | 0.9577   | 0.9637            | 0.9581         | 0.9597   |

---

## 🧠 Interpretation of Results

After removing duplicated news and working with the real class distribution, **all models maintain excellent performance**, with accuracies between **0.95 and 0.99** and **F1-macro scores above 0.95**.

The best-performing model is:

- **Multinomial Naive Bayes with Bag of Words**
  - Accuracy: **0.9859**
  - F1-macro: **0.9872**
  - Only **1 misclassification out of 71 news articles**

Random Forest and Linear SVM with TF-IDF follow very closely, showing near-perfect classification in *Deportes*, *Internacional* and *Sociedad*, with a slight decrease in the *Nacional* category.

---

## ⚖️ Effect of Class Imbalance

Comparing results between artificially balanced datasets and the real unbalanced dataset shows that:

- Removing duplicated samples **does not degrade performance**
- Naive Bayes with BoW slightly **improves** its F1-macro score
- The **ranking of models remains stable** across experiments

This highlights the robustness of classical NLP models when evaluated properly using macro-averaged metrics.

---

## 🆕 Generalization to New, Unseen News

To evaluate real-world applicability, the trained models were tested on **completely new news articles**, including:

- Clear-cut cases (sports, international summits)
- Ambiguous or borderline topics (national vs. society)
- Topics not dominant in training data (technology, economy)

All new articles were processed using the **same preprocessing pipeline** as the training data.

Predictions across models were largely **consistent**, especially for well-defined categories, while ambiguous cases revealed meaningful differences between probabilistic and linear models.

---

## 🔮 Future Work: Transfer Learning

As a theoretical extension, future work could explore:

- Transfer learning with pretrained language models (e.g. **BERT**, **RoBERTa**)
- Comparison between classical ML models and transformer-based approaches
- Robustness under stronger class imbalance and domain shift

This would allow studying how **contextual embeddings** improve performance in complex or ambiguous classification scenarios.

---

## 🛠️ Technologies Used

- Python
- scikit-learn
- NumPy / pandas
- NLP preprocessing techniques
- Classical Machine Learning models

---

## 📌 Final Remarks

This project demonstrates that **classical NLP techniques**, when carefully implemented and evaluated, can achieve **excellent performance** on real-world text classification tasks.

Beyond raw accuracy, the work emphasizes **methodology, interpretability, and statistical soundness**, making it a solid foundation for more advanced NLP approaches.

---

## 📚 External Documentation

An automatically generated documentation view of this repository is available via DeepWiki:
 
👉 https://deepwiki.com/maria2332/nlp-text-classification-and-spam-detection
