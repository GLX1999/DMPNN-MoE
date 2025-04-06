# DMPNN-MoE: Temperature-Aware Small Molecule Solubility Prediction via Directed Message Passing Neural Networks with Mixture-of-Experts

## Introduction

This repository provides code for predicting the solubility of solutes in organic solvents using a novel **Directed Message-Passing Neural Network (DMPNN)** combined with a **Mixture of Experts (MoE)** model. The model is designed to enhance solubility predictions by capturing complex molecular structures and solute-solvent interactions. This work combines large-scale experimental solubility data and advanced machine learning techniques to offer a more accurate and robust tool for solubility prediction across various solvents and temperatures.

The proposed model has been evaluated using a comprehensive dataset containing over 56,000 experimental data points and demonstrated superior performance over existing models.

## Libraries and Dependencies

To run the code, ensure the following libraries are installed:

- **torch**: 2.2.1+cu118
- **sklearn**: 1.5.2
- **dgl**: 2.2.1+cu118
- **rdkit**: 2022.09.5
- **pytorch-lightning**: 2.2.5
- **numpy**: 1.24.4
- **pandas**: 1.5.3

You can install these dependencies via pip:

```
pip install torch==2.2.1+cu118
pip install scikit-learn==1.5.2
pip install dgl==2.2.1+cu118
pip install rdkit==2022.09.5
pip install pytorch-lightning==2.2.5
pip install numpy==1.24.4
pip install pandas==1.5.3
```

## Dataset

The dataset used for training and evaluation includes **56,945 solubility data points**, encompassing **791 solutes** and **140 solvents** across a **temperature range from 243.15 K to 403.15 K**. The solubility values are expressed in **log(S, mol/mol)**, with higher values indicating greater solubility.

- **Solutes**: Represented by SMILES notation.
- **Solvents**: Represented by SMILES notation.
- **Solubility data**: Collected from experimental datasets such as **BigSolDB** and **CombiSolu-Exp**, supplemented by literature data.

## Model Architecture

### Directed Message-Passing Neural Network (DMPNN)

The DMPNN leverages the graph structure of molecules where atoms are represented as nodes and bonds as edges. This allows the model to capture the local and global chemical relationships within molecules efficiently.

- **Node Features**: Each atom is represented by a 111-dimensional feature vector (e.g., atom type, valence electrons, hybridization).
- **Edge Features**: Each bond is represented by a 13-dimensional feature vector (e.g., bond order, conjugation).

In the message-passing phase, the DMPNN updates the node representations by iterating over directed edges, ensuring that the molecular topology is captured effectively.

### Mixture of Experts (MoE)

MoE is integrated with DMPNN to enhance the model’s adaptability across different solute-solvent systems. It consists of multiple expert networks that process different aspects of the input data. A gating network dynamically allocates weights to the outputs of these expert networks, ensuring that the model leverages the best-performing experts based on the input characteristics.

- **Experts**: Each expert specializes in different molecular features or solute-solvent interactions.
- **Gating Network**: Allocates dynamic weights to expert outputs.

### Temperature and Molecular Descriptors

The model also incorporates temperature and additional molecular-level descriptors (e.g., octanol-water partition coefficient, topological polar surface area), which are concatenated with the molecular feature vectors to further enrich the model's learning.

## Model Training and Evaluation

The model is trained using **PyTorch** and **PyTorch Lightning**, with optimization performed via the **Adam optimizer**. Cross-validation is performed to evaluate the model's generalization ability, and multiple performance metrics (e.g., MSE, MAE, R²) are used to assess the model’s predictive accuracy.

## Results

The **DMPNN-MoE** model achieves an **MSE of 0.1815 ± 0.0256**, **R² of 0.8634 ± 0.0165**, and **MAE of 0.2557 ± 0.0096** in 10-fold cross-validation. When validated on **ethylene sulfate (DTD)**—a solute not included in the training set—it attained an **R² of 0.8740**, an **MAE of 0.0736****, and an **RMSE of 0.0923.** These results underscore the model’s exceptional generalization ability and its robustness in predicting solubility for entirely unseen solutes.

## Example Usage

### Model Inference

To use `prediction.py` for solubility prediction, follow these steps:

1. Prepare a CSV file with three columns: `solute_smiles`, `solvent_smiles`, and `temperature`.
2. Input the file path for the data, the model paths (for 10-folds), and set the output file path.
3. The script will load the model, perform predictions, and save the results in the output CSV.

```
    input_csv = r'.csv'
    save_path = r'.csv'
    checkpoint_path = r'.ckpt'
```

## Conclusion

This repository offers a state-of-the-art tool for predicting solubility using machine learning techniques, combining graph neural networks with a mixture of experts framework. The model is versatile and can be applied across various solute-solvent systems, offering valuable insights for drug discovery, materials science, and chemical engineering.

## References

1. Guo, L., Zhao, Y., Liu, Q., Meng, Q. (2025). "DMPNN-MoE: Temperature-Aware Small Molecule Solubility Prediction via Directed Message Passing Neural Networks with Mixture-of-Experts". 