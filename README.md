# Machine Learning Practical Assignment
## Data Preprocessing & Feature Selection — Adult Census Income Dataset

### Group Members
| Student | Roll Number | Contribution |
|---|---:|---|
| Sachin Kumar | CSJMA23001390037 | Dataset, EDA, coordination |
| Rohit Singh | CSJMA23001390036 | Data preprocessing |
| Shubham Kumar Gupta | CSJMA23001390043 | Feature selection |
| Jay Singh | CSJMA23001390018 | Visualization, documentation, GitHub |

### Objective
This project studies data preprocessing and feature selection using the Adult Census Income dataset.
The notebook follows the assignment philosophy:

**Understand → Calculate → Code → Verify → Interpret**

### Dataset
UCI Adult / Census Income dataset.

### Repository Structure
- `dataset/` — dataset files (the notebook can download the data automatically)
- `notebooks/` — main Google Colab/Jupyter notebook
- `src/` — reusable from-scratch implementations
- `results/graphs
   - Generated plots are saved automatically when the notebook is run./` — generated visualizations
- `results/outputs/` — generated tables/results
- `report/` — final report

### How to Run
1. Open `notebooks/ML_Adult_Income_Assignment.ipynb` in Google Colab.
2. Run cells from top to bottom.
3. The notebook downloads the UCI Adult data if the CSV is not already present.
4. Review and personalize the interpretations and final feature decisions.
5. Commit meaningful changes to GitHub.

### Important Academic Note
This repository is an original project scaffold and implementation for this group. External repositories may be used only as references. Do not submit another group's repository unchanged.


## Final notebook workflow
The notebook uses a leakage-free final modelling pipeline: the dataset is split before training-dependent imputation and supervised feature selection. Feature-selection evidence is learned from the training set and the test set is kept separate.
