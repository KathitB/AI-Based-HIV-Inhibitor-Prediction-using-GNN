# AI-Based-HIV-Inhibitor-Prediction-using-GNN

Here’s a `README.md` file tailored for your **AI-based HIV Inhibitor Prediction** project using **DeepChem**, **SMILES strings**, and **Streamlit**:

---

````markdown
#  AI-Based HIV Inhibitor Prediction

This project uses deep learning models and cheminformatics to predict the **HIV activity of molecules** based on their SMILES representation. Built with **DeepChem** for molecular graph processing and **Streamlit** for an interactive web interface.

---

##  Features

-  Upload a CSV file containing **SMILES** strings.
-  Automatically featurizes molecules using **ConvMolFeaturizer**.
-  Predicts HIV inhibition activity using a pretrained **Graph-based model**.
-  Displays prediction results in an interactive table.
-  Download the results as a CSV file.

---

##  Tech Stack

- Python 3.10
- [DeepChem](https://deepchem.io/)
- Streamlit
- NumPy / Pandas
- RDKit (optional for SMILES validation)



##  Setup Instructions

1. **Clone the repo**:

   ```bash
   git clone https://github.com/your-username/hiv-inhibitor-predictor.git
   cd hiv-inhibitor-predictor
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:

   ```bash
   streamlit run app/app.py
   ```

---

##  CSV Format

Upload a CSV file with the following format:

```csv
smiles
CCO
CC(C)Cl
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

* The file must have a column named **`smiles`**.
* Rows with invalid SMILES will be skipped with a warning.

---

## Sample Output

| smiles                       | Predicted\_HIV\_Activity |
| ---------------------------- | ------------------------ |
| CCO                          | 0.489                    |
| CC(C)Cl                      | 0.672                    |
| CN1C=NC2=C1C(=O)N(C(=O)N2C)C | 0.825                    |

---

## 🛠 Troubleshooting

* **`ValueError: All arrays must be of the same length`**
  → This happens if invalid SMILES are skipped. The code now truncates mismatched lists.

* **`AttributeError: 'numpy.ndarray' object has no attribute 'atom_features'`**
  → This can happen if incorrect feature format is passed to the model. Ensure `ConvMolFeaturizer` is used and only valid molecules are included.


##  License

MIT License. Open to academic and non-commercial use.


##  Author

Developed by Kathit Bhongale as part of an internship project on AI for Drug Discovery.

