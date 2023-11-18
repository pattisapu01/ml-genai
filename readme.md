# Credit Card Transaction Analysis with Gen-AI Techniques

## Overview
This repository contains two main Python scripts: `cc-data-generator.py` and `ml-genai.py`. The `cc-data-generator.py` script is used for generating simulated credit card transaction data, while `ml-genai.py` applies machine learning and generative AI techniques to analyze this data, focusing on anomaly detection and other insights.

## Files
- `cc-data-generator.py`: Generates a dataset of simulated credit card transactions. This script creates a realistic set of transaction data, including attributes like transaction type, amount, balances, and a flag indicating fraudulent activities.
- `ml-genai.py`: Applies various machine learning models and techniques to the generated dataset to identify patterns, anomalies, and potential fraudulent transactions. It includes methods like Random Forest, SMOTE for handling imbalanced data, and advanced techniques involving embeddings and distance metrics.

## Getting Started

### Prerequisites
- Python 3.x
- Relevant Python libraries as listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/pattisapu01/ml-genai
2. ## Install the required Python packages

```bash
pip install -r requirements.txt

3. run cc generator

```bash
   python cc-data-generator.py
4. run ml-genai.py

```bash
   python ml-genai.py
 