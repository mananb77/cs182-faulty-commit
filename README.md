# Faulty Commit Classification with Binary Classifiers

This project implements and evaluates binary classification models to predict whether a commit is faulty or not. It includes data preprocessing, baseline and neural models, evaluation using precision-recall curves and average precision (AP), and support for trivial and advanced baselines.

This repository was developed as part of a machine learning project, and contains an end-to-end pipeline from data loading and model training to offline evaluation and submission packaging. This machine learning project was developed for Data C182 Fall 2024: Deep Neural Networks at UC Berkeley.

**Project spec**: https://docs.google.com/document/d/1erKfrMXIY_JkPB_648wyaTXP89Llo4MyD68l_pZRwZo/edit?tab=t.0#heading=h.i522cvq4jmw

---

## 🚀 Features

- Offline evaluation using precision-recall curves, F1 scores, and Average Precision (AP)
- Implementation of trivial baseline classifiers:
  - AlwaysPositiveBinaryClassifier
  - AlwaysNegativeBinaryClassifier
  - RandomBinaryClassifier
- Neural models including:
  - Single-layer feedforward neural network (SingleLayerNN)
  - Custom multilayer architectures to beat baseline performance
- Utilities for loading and preprocessing commit-level and developer-level metadata
- Colab integration and Gradescope-compatible submission pipeline

---

## 🗂️ Project Structure
.
├── consts.py                            # Project-wide constants

├── final_project.ipynb                  # Main Colab notebook for training, evaluation, and submission

├── evaluation/

│   └── offline_eval.py                  # Offline evaluation pipeline implementation (predict_samples, compute_eval_metrics)

├── modeling/

│   ├── model_interface.py              # Defines the base model interface

│   ├── model_random.py                 # Contains RandomBinaryClassifier

│   ├── model_single_layer.py          # Implementation of SingleLayerNN

│   └── trivial_models.py              # AlwaysPositive and AlwaysNegative classifiers

├── utils/

│   └── utils.py                        # Utility functions (e.g., dataloader creation, preprocessing helpers)

├── dataloader/

│   ├── fault_csv_dataset.py           # Custom PyTorch dataset class for loading commit data

│   └── preprocess.py                  # Data preprocessing logic (fit/transform for features)

├── data/

│   ├── raw/                           # Contains raw datasets (e.g., user_meta.csv)

│   └── split/                         # Pre-split train/val/test datasets

├── submission/

│   └── generate_submission.py         # Script to zip files and generate Gradescope-compatible submission

└── requirements.txt                   # Python dependencies

---

## 📈 Offline Evaluation Details

The evaluation pipeline computes:

- Precision-Recall Curve using `sklearn.metrics.precision_recall_curve`
- Average Precision using `sklearn.metrics.average_precision_score`
- Operating point (threshold) that maximizes F1 score
- Precision/Recall/F1 metrics at arbitrary thresholds

All results are wrapped in structured data classes (`PredictionMetadata`, `OperatingPointMetrics`, etc.).

---

## 🧪 Trivial Baselines

Implemented in `trivial_models.py` and `model_random.py`:

- **AlwaysPositiveBinaryClassifier**: always predicts the positive class
- **AlwaysNegativeBinaryClassifier**: always predicts the negative class
- **RandomBinaryClassifier**: predicts randomly based on a biased coin

Use these to establish performance baselines and debug the evaluation pipeline.

---

## 🧠 Neural Models

- **SingleLayerNN**: simple feedforward model with a single linear layer
- **Improved Model**: optional multi-layer architecture that aims to exceed `0.025` AP on the test set

---

## 📌 Acknowledgements
This project was developed for educational purposes as part of a machine learning course. Special thanks to the course staff for designing the pipeline and providing a great learning experience.
