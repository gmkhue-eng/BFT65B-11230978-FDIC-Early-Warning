# BFT65B: A Big-Data Pipeline for Bank Failure Early Warning

**Course:** BFT65B — Big Data Analytics in Finance (National Economics University)  
**Author:** Giang Minh Khue — 11230978  
**Repository:** https://github.com/gmkhue-eng/BFT65B-11230978-FDIC-Early-Warning.git  
**Data Source:** US Federal Deposit Insurance Corporation (FDIC) BankFind Public API (`1984Q1–2026Q2`)  
**Download Date:** 2026-11-08

\---

## 1\. Project Overview

This repository implements an end-to-end, reproducible, and leak-free data engineering and machine learning pipeline for **1-year-ahead bank failure early warning**. Designed from the perspective of a central bank supervisory data engineer, the pipeline processes over four decades of quarterly FDIC Call Reports (`1,682,616` raw bank-quarter rows across `170` quarters) on a single workstation using **Polars**, **PyArrow**, **scikit-learn**, and **LightGBM**, with local distributed reproduction and Catalyst execution plan verification in **PySpark**.

### Key Architectural Highlights

* **Checkpoint-Based Directed Acyclic Pipeline:** Intermediate Parquet checkpoints (`clean\_panel.parquet`, `feature\_table.parquet`, `oof\_predictions.parquet`) and `joblib.Memory` caching eliminate redundant API calls, I/O overhead, and model refits across runs.
* **Canonical Schema \& Storage Optimization:** Enforces an explicit PyArrow `CANONICAL\_FIN\_SCHEMA` across all 170 Hive-partitioned Snappy Parquet files (`REPDTE=YYYYMMDD/data.parquet`), eliminating cross-partition schema drift (`Int64` vs. `Float64`).
* **Strict Zero-Leakage Guardrails:** Enforces within-bank chronological forward-filling, automated future-corruption unit testing (`assert\_no\_lookahead\_bias`), a **Golden Out-of-Time Split** (`1984Q1–2006Q4` development vs. `2008Q1–2025Q2` holdout test separated by a **4-quarter embargo gap** in `2007Q1–2007Q4`), and `PurgedExpandingWindowCV`.
* **Cost-Sensitive Supervisory Calibration:** Calibrates the alarm probability threshold strictly on development Out-of-Fold (OOF) predictions by minimizing asymmetric regulatory loss ($\\text{FN}\\cdot C\_{\\text{FN}} + \\text{FP}\\cdot C\_{\\text{FP}}$), where missed-failure resolution losses ($C\_{\\text{FN}}$) are estimated empirically from pre-2007 FDIC failure resolutions.

\---

## 2\. Dataset Scope \& Partition Summary

* **Raw Financials (`/financials`, `1984Q1–2026Q2`):** `1,682,616` bank-quarter rows, `33` CAMELS-aligned fields.
* **Raw Failures (`/failures`):** Complete FDIC historical failed institutions table (`CERT`, `FAILDATE`, `COST`, `QBFASSET`, `RESTYPE`).
* **Raw Institutions (`/institutions`):** Complete FDIC institutional directory (`ACTIVE`, `ENDEFYMD`, `CHANGEC1`) used to reconcile regulatory failures against non-failure exits (mergers and voluntary liquidations).
* **Clean Modelling Panel (`1984Q1–2025Q2`):** `1,653,966` rows (`11,577` positive bank-quarters; `3,015` unique failed banks contributing at least one positive label).
* **Golden Out-of-Time Split:**

  * **Development Set (`1984Q1–2006Q4`):** `1,185,025` rows (`9,351` positive bank-quarters; captures the Savings \& Loan crisis).
  * **Embargo Gap (`2007Q1–2007Q4`):** `4` quarters excluded to prevent overlapping `(q, q+4]` target leakage.
  * **Holdout Test Set (`2008Q1–2025Q2`):** `434,255` rows (`2,177` positive bank-quarters; evaluates out-of-time generalization across the 2008 Global Financial Crisis and 2023 regional bank failures).

\---

## 3\. Repository Structure

> \*\*Note:\*\* Raw datasets, processed Parquet checkpoints, trained `.joblib` artifacts, and local caches are excluded via `.gitignore` in compliance with submission guidelines (`Do not commit data files`) and are automatically regenerated when running the notebook.

```text
BFT65B-11230978-FDIC-Early-Warning/
├── fdic\_download\_starter.ipynb   # Main end-to-end pipeline notebook (Parts 1–6)
├── requirements.txt              # Pinned Python dependencies
├── .gitignore                    # Excludes raw/processed data and cache artifacts
└── README.md                     # Project documentation \& reproduction guide
```

\---

## 4\. Environment Setup \& Installation

### Prerequisites

* **Python:** `3.11` (recommended on Windows for optimal PySpark compatibility) or `3.10–3.13` on Linux / macOS.
* **Java (JDK):** `JDK 11`, `17`, or `21` installed and available on `PATH` (required for local PySpark execution).

### Local Installation (Windows / Linux / macOS)

1. Clone the repository and enter the project directory:

```bash
   git clone https://github.com/gmkhue-eng/BFT65B-11230978-FDIC-Early-Warning.git
   cd BFT65B-11230978-FDIC-Early-Warning
   ```

2. Create and activate a virtual environment:

```bash
   python -m venv .venv
   # Windows:
   .venv\\Scripts\\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. Install pinned dependencies:

```bash
   pip install -r requirements.txt
   ```

> \*\*Windows PySpark Guardrails:\*\* Cell 0 automatically binds Spark to `127.0.0.1`, sets `PYSPARK\_PYTHON` to the active interpreter, limits shuffle partitions (`8`), and uses `C:/spark\_temp` as `spark.local.dir` to prevent Windows firewall/socket hangs. All Spark reads use PyArrow-standardized Parquet files directly, avoiding `winutils.exe` write dependencies.

\---

## 5\. How to Reproduce End-to-End

1. Open `fdic\_download\_starter.ipynb` in VS Code or Jupyter Lab and select the `.venv` Python kernel.
2. Verify the execution control flags in **Cell 0**:

```python
   SMOKE\_TEST = False          # False: full 170-quarter scope (1984Q1-2026Q2)
   FORCE\_DOWNLOAD = False      # Auto-downloads if data\_raw/ is absent; skips if present
   FORCE\_CLEAN = False         # Auto-builds clean\_panel.parquet if absent
   FORCE\_FEATURES = False      # Auto-builds feature\_table.parquet if absent
   RUN\_OPTIONAL\_RFECV = False  # Excluded from main fast run
   RANDOM\_SEED = 42            # Fixed deterministic seed across all splits \& models
   ```

3. Run **`Kernel -> Restart Kernel and Run All Cells`**.

   * **First clean run:** Downloads all 170 quarters + `/failures` + `/institutions` from the FDIC API (\~15–25 minutes with 3 concurrent workers), constructs checkpoints, runs Polars/PySpark benchmarks, trains/tunes models, and saves tables/figures to `results/` (\~3 minutes compute time).
   * **Subsequent runs:** Automatically reuses Parquet checkpoints and completes end-to-end in **\~2–3 minutes**.

\---

## 6\. References \& Academic Integrity Disclosure

### References

\[1] Federal Deposit Insurance Corporation (FDIC). (n.d.). *BankFind Suite: Public API Documentation and Bulk Data Portal*. https://api.fdic.gov/banks/docs/ and https://banks.data.fdic.gov/bankfind-suite/bulkData

\[2] Nguyen, T. L. (2026). *BFT65B: Big Data Analytics in Finance — Course Lectures and Starter Ingestion Script (`fdic\_download\_starter.py`)*. National Economics University.

\[3] Polars Developers. (n.d.). *Polars Documentation* (https://docs.pola.rs); Apache Software Foundation. *Apache Parquet* (https://parquet.apache.org) and *Apache Arrow / PyArrow Documentation* (https://arrow.apache.org).

\[4] Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., Meng, X., Rosen, J., Venkataraman, S., Franklin, M. J., Ghodsi, A., Gonzalez, J., Shenker, S., \& Stoica, I. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM*, 59(11), 56–65.

\[5] Armbrust, M., Xin, R. S., Lian, C., Huai, Y., Liu, D., Bradley, J. K., Meng, X., Kaftan, T., Franklin, M. J., Ghodsi, A., \& Zaharia, M. (2015). Spark SQL: Relational data processing in Spark. In *Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data* (pp. 1383–1394).

\[6] Meng, X., Bradley, J., Yavuz, B., Sparks, E., Venkataraman, S., Liu, D., Freeman, J., Tsai, D. B., Amde, M., Owen, S., Xin, D., Xin, R., Franklin, M. J., Zadeh, R., Zaharia, M., \& Talwalkar, A. (2016). MLlib: Machine learning in Apache Spark. *Journal of Machine Learning Research*, 17(34), 1–7.

\[7] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., \& Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

\[8] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., \& Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.

\[9] López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley \& Sons.

\[10] Saito, T., \& Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.

\[11] Cole, R. A., \& White, L. J. (2012). Déjà vu all over again: The causes of U.S. commercial bank failures this time around. *Journal of Financial Services Research*, 42(1–2), 5–29.

### AI Tool Disclosure

\[12] Claude (Anthropic) was used to help draft and structure the written report from the author's notebook outputs, while Google Gemini was used as a coding and debugging assistant to optimize Polars lazy expressions, resolve Windows PySpark/scikit-learn compatibility issues, and organize the modular checkpoint pipeline. All numerical results originate directly from the executed notebook, and all architectural decisions, empirical validations, and written interpretations were independently verified and authored by the author, who assumes full responsibility for the content.

