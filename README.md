# Machine Learning Practicals — T.E. AI & DS (SPPU)

Ten practical examination programs (25 marks each) covering all 11 syllabus experiments, each in its own folder with a detailed **README containing the theory, a code walkthrough, real output and viva questions**.

> The 10 original programs are **unmodified** copies of the code from the source notebook (verified byte-for-byte). The only added file is `practical_10b_q_learning.py` (see below).

## Practicals

| # | Topic | Dataset | Folder |
|---|---|---|---|
| 1 | Feature engineering & preprocessing | Wine | [practical-01](practical-01-feature-engineering-preprocessing/) |
| 2 | PCA and LDA | Wine | [practical-02](practical-02-pca-lda-dimensionality-reduction/) |
| 3 | Multiple Linear Regression | Diabetes | [practical-03](practical-03-multiple-linear-regression/) |
| 4 | Polynomial Regression & bias–variance | Diabetes | [practical-04](practical-04-polynomial-regression-bias-variance/) |
| 5 | Ridge and Lasso regression | Diabetes | [practical-05](practical-05-ridge-lasso-regularization/) |
| 6 | Logistic Regression | Breast Cancer | [practical-06](practical-06-logistic-regression/) |
| 7 | SVM (linear / RBF / polynomial) | Breast Cancer | [practical-07](practical-07-svm-kernels/) |
| 8 | K-Means, Elbow & Silhouette | Wine | [practical-08](practical-08-kmeans-clustering/) |
| 9 | DBSCAN | Wine | [practical-09](practical-09-dbscan-clustering/) |
| 10A | Ensemble learning (DT / RF / AdaBoost) | Breast Cancer | [practical-10](practical-10-ensemble-learning-and-q-learning/) |
| 10B | Q-Learning, 4×4 Grid World *(added)* | self-defined | [practical-10](practical-10-ensemble-learning-and-q-learning/) |

All datasets ship with scikit-learn — **no downloads needed**.

## Quick start

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python practical-03-multiple-linear-regression/practical_03_multiple_linear_regression.py
```

Plots open in a window with `plt.show()`. On a headless machine run with `MPLBACKEND=Agg python <file>`.

## Repository layout

```
├── README.md                  ← this file
├── DEBUG_REPORT.md            ← syntax / run check results for every program
├── requirements.txt
├── notebook/                  ← original source notebook
├── practical-XX-<topic>/
│   ├── practical_XX_<topic>.py   ← code (unchanged from notebook)
│   ├── README.md                 ← theory + code explanation + output + viva Q&A
│   └── images/                   ← plots (practicals 4, 8, 9)
└── tools/
    ├── build_readmes.py       ← re-inserts code + real output into the READMEs
    └── generate_figures.py    ← regenerates the plots without editing the code
```

## Debugging summary
Every program passes the Python syntax check and runs end-to-end (tested on Python 3 with scikit-learn 1.9.1, NumPy 2.5, pandas, matplotlib). Details, version notes and observations are in **[DEBUG_REPORT.md](DEBUG_REPORT.md)**.

## About Practical 10B
The notebook's Question 10 offers *Option A (Ensemble)* **or** *Option B (Q-Learning)*, but only Option A has code. `practical_10b_q_learning.py` was written to fill that gap, tested, and is labelled as added in its header comment. Delete it if you only want the notebook's original content.
