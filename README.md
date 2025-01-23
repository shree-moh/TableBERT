# TableBERT for Software Defect Prediction
This repository contains the implementation of TableBERT, a pretrained language model fine-tuned for the Software Defect Prediction (SDP) task. TableBERT bridges the gap between textual and structured tabular data, providing a robust solution for defect prediction tasks in software engineering.
Overview

Software Defect Prediction (SDP) is a challenging task that involves analyzing both structured software metrics and textual descriptions to predict software defects. This implementation leverages TableBERT, a model designed to jointly learn representations for natural language and tabular data. Additionally, the model's hyperparameters are tuned using Particle Swarm Optimization (PSO), and Local Interpretable Model-Agnostic Explanations (LIME) are integrated for enhanced model interpretability

Features

Model: Fine-tuned TableBERT for SDP tasks.

Comparison: Performance comparison with Vanilla TableBERT (baseline).

Optimization: Hyperparameter tuning using PSO.

Interpretability: Integration of LIME for explaining model predictions.

Metrics: Evaluation using key metrics:

(PD)

(PF)

Balance

(FIR)


Installation

Prerequisites

Python 3.8+

PyTorch

Transformers library (Hugging Face)

LIME for interpretability
