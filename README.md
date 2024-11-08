# VaxxiPred

**VaxxiPred** is a repository for vaccine candidate prediction research using various machine learning and deep learning models. The dataset contains protein sequences labeled as vaccine or non-vaccine candidates, with data from species including *Aeromonas hydrophila*, *Edwardsiella ictaluri*, and *Flavobacterium columnare*. This repository includes data preprocessing and four model implementations for binary classification of vaccine candidates.

## Repository Structure

The repository has the following folder structure:

- **Data**
  - **`Data_Preprocessing.ipynb`**: Notebook for preprocessing and preparing protein sequence data.
  - **`df_final_sequence.csv`**: Main dataset containing protein sequences and labels for vaccine candidate prediction.
  - **`uniprotkb_Aeromonas_hydrophila_AND_anno_2024_10_30.fasta`**: FASTA file for *Aeromonas hydrophila* protein sequences.
  - **`uniprotkb_Edwardsiella_ictaluri_AND_anno_2024_10_30.fasta`**: FASTA file for *Edwardsiella ictaluri* protein sequences.
  - **`uniprotkb_Flavobacterium_columnare_AND_2024_10_30.fasta`**: FASTA file for *Flavobacterium columnare* protein sequences.

- **Models**
  - **`Deep_NN_Vaccine_Prediction.ipynb`**: Deep Neural Network model for vaccine candidate prediction.
  - **`Finetuned_ESM2_Vaccine_Prediction.ipynb`**: Fine-tuned ESM-2 transformer model for protein sequence classification.
  - **`GNN_Vaccine_Prediction_3.ipynb`**: Graph Neural Network model utilizing GAT (Graph Attention Network) for candidate prediction based on sequence similarity.
  - **`XGBoost_Vaccine_Prediction.ipynb`**: XGBoost model with ESM-2 embeddings and BioPython-induced features.

## Project Overview

This project aims to predict potential vaccine candidates from protein sequences by training and evaluating various models. Four models are explored in this repository:

- **Deep Neural Network (DNN)**: A fully connected neural network trained on protein features.
- **Fine-tuned ESM-2**: A transformer model specifically designed for protein sequences, fine-tuned on our dataset.
- **Graph Neural Network (GNN)**: A GAT-based model that uses sequence similarity to create a graph and classify nodes (sequences) as vaccine candidates.
- **XGBoost**: A gradient boosting model trained on ESM-2 embeddings and physiological features.

## Usage

1. **Data Preprocessing**: Use `Data_Preprocessing.ipynb` to prepare the data for training and testing.
2. **Model Training and Evaluation**: Each model can be run from its respective notebook in the `Models` folder. The notebooks contain instructions for training, evaluating, and visualizing model performance.

## Results

Each model's performance is evaluated based on accuracy, precision, recall, F1-score, ROC-AUC score, and log loss. Confusion matrices and metrics are provided in the notebooks for detailed evaluation.
