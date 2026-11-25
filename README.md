\# BFT65B: A Big-Data Pipeline for Bank Failure Early Warning



\*\*Course:\*\* BFT65B — Big Data Analytics in Finance (National Economics University)  

\*\*Author:\*\* Giang Minh Khue — 11230978  

\*\*Repository:\*\* https://github.com/gmkhue-eng/BFT65B-11230978-FDIC-Early-Warning.git  

\*\*Data Source:\*\* US Federal Deposit Insurance Corporation (FDIC) BankFind Public API (`1984Q1–2026Q2`)  

\*\*Download Date:\*\* 2026-11-08  



\---



\## 1. Project Overview



This repository implements an end-to-end, reproducible, and leak-free data engineering and machine learning pipeline for \*\*1-year-ahead bank failure early warning\*\*. Designed from the perspective of a central bank supervisory data engineer, the pipeline processes over four decades of quarterly FDIC Call Reports (`1,682,616` raw bank-quarter rows across `170` quarters) on a single workstation using \*\*Polars\*\*, \*\*PyArrow\*\*, \*\*scikit-learn\*\*, and \*\*LightGBM\*\*, with distributed reproduction and Catalyst execution plan verification in \*\*PySpark\*\*.



\### Key Architectural Highlights

\* \*\*Checkpoint-Based Directed Acyclic Pipeline:\*\* Intermediate Parquet checkpoints (`clean\_panel.parquet`, `feature\_table.parquet`, `oof\_predictions.parquet`) and `joblib.Memory` caching eliminate redundant API calls, I/O overhead, and model refits across runs.

\* \*\*Canonical Schema \& Storage Optimization:\*\* Enforces an explicit PyArrow `CANONICAL\_FIN\_SCHEMA` across all 170 Hive-partitioned Snappy Parquet files (`REPDTE=YYYYMMDD/data.parquet`), eliminating cross-partition schema drift (`Int64` vs. `Float64`).

\* \*\*Strict Zero-Leakage Guardrails:\*\* Enforces within-bank chronological forward-filling, automated future-corruption unit testing (`assert\_no\_lookahead\_bias`), a \*\*Golden Out-of-Time Split\*\* (`1984Q1–2006Q4` development vs. `2008Q1–2025Q2` holdout test separated by a \*\*4-quarter embargo gap\*\* in `2007Q1–2007Q4`), and `PurgedExpandingWindowCV`.

\* \*\*Cost-Sensitive Supervisory Calibration:\*\* Calibrates the alarm probability threshold strictly on development Out-of-Fold (OOF) predictions by minimizing asymmetric regulatory loss ($\\text{FN}\\cdot C\_{\\text{FN}} + \\text{FP}\\cdot C\_{\\text{FP}}$), where missed-failure resolution losses ($C\_{\\text{FN}}$) are estimated empirically from pre-2007 FDIC failure resolutions.



\---



\## 2. Dataset Scope \& Partition Summary



\* \*\*Raw Financials (`/financials`, `1984Q1–2026Q2`):\*\* `1,682,616` bank-quarter rows, `33` CAMELS-aligned fields.

\* \*\*Raw Failures (`/failures`):\*\* Complete FDIC historical failed institutions table (`CERT`, `FAILDATE`, `COST`, `QBFASSET`, `RESTYPE`).

\* \*\*Raw Institutions (`/institutions`):\*\* Complete FDIC institutional directory (`ACTIVE`, `ENDEFYMD`, `CHANGEC1`) used to reconcile regulatory failures against non-failure exits (mergers and voluntary liquidations).

\* \*\*Clean Modelling Panel (`1984Q1–2025Q2`):\*\* `1,653,966` rows (`11,577` positive bank-quarters; `3,015` unique failed banks contributing at least one positive label).

\* \*\*Golden Out-of-Time Split:\*\*

&#x20; \* \*\*Development Set (`1984Q1–2006Q4`):\*\* `1,185,025` rows (`9,351` positive bank-quarters; captures the Savings \& Loan crisis).

&#x20; \* \*\*Embargo Gap (`2007Q1–2007Q4`):\*\* `4` quarters excluded to prevent overlapping `(q, q+4]` target leakage.

&#x20; \* \*\*Holdout Test Set (`2008Q1–2025Q2`):\*\* `434,255` rows (`2,177` positive bank-quarters; evaluates out-of-time generalization across the 2008 Global Financial Crisis and 2023 regional bank failures).



\---



\## 3. Repository Structure



> \*\*Note:\*\* Raw datasets, processed Parquet checkpoints, trained `.joblib` artifacts, and local caches are excluded via `.gitignore` in compliance with submission guidelines (`Do not commit data files`) and are automatically regenerated when running the notebook.



```text

FDIC\_EWS/

├── fdic\_download\_starter.ipynb   # Main end-to-end pipeline notebook (Parts 1–6)

├── requirements.txt              # Pinned Python dependencies

├── .gitignore                    # Excludes raw/processed data and cache artifacts

└── README.md                     # Project documentation \& reproduction guide

```



\---



\## 4. Environment Setup \& Installation



\### Prerequisites

\* \*\*Python:\*\* `3.11` (recommended on Windows for optimal PySpark compatibility) or `3.10–3.13` on Linux / Google Colab.

\* \*\*Java (JDK):\*\* `JDK 11`, `17`, or `21` installed and available on `PATH` (required for local PySpark execution).



\### Local Installation (Windows / Linux / macOS)

1\. Clone the repository and enter the project directory:

&#x20;  ```bash

&#x20;  git clone <your-repo-url>

&#x20;  cd FDIC\_EWS

&#x20;  ```

2\. Create and activate a virtual environment:

&#x20;  ```bash

&#x20;  python -m venv .venv

&#x20;  # Windows:

&#x20;  .venv\\Scripts\\activate

&#x20;  # Linux / macOS:

&#x20;  source .venv/bin/activate

&#x20;  ```

3\. Install pinned dependencies:

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



> \*\*Windows PySpark Guardrails:\*\* Cell 0 automatically binds Spark to `127.0.0.1`, sets `PYSPARK\_PYTHON` to the active interpreter, limits shuffle partitions (`8`), and uses `C:/spark\_temp` as `spark.local.dir` to prevent Windows firewall/socket hangs. All Spark reads use PyArrow-standardized Parquet files directly, avoiding `winutils.exe` write dependencies.



\### Google Colab Execution

1\. Upload `fdic\_download\_starter.ipynb` to Google Colab.

2\. Optionally mount Google Drive (checkpoints save to `/content/drive/MyDrive/FDIC\_EWS` if mounted, or `/content/FDIC\_EWS` otherwise).

3\. Run all cells sequentially; PySpark executes natively on Colab's Linux JVM.



\---



\## 5. How to Reproduce End-to-End



1\. Open `fdic\_download\_starter.ipynb` in VS Code, Jupyter Lab, or Google Colab.

2\. Verify the execution control flags in \*\*Cell 0\*\*:

&#x20;  ```python

&#x20;  SMOKE\_TEST = False          # False: full 170-quarter scope (1984Q1-2026Q2)

&#x20;  FORCE\_DOWNLOAD = False      # Auto-downloads if data\_raw/ is absent; skips if present

&#x20;  FORCE\_CLEAN = False         # Auto-builds clean\_panel.parquet if absent

&#x20;  FORCE\_FEATURES = False      # Auto-builds feature\_table.parquet if absent

&#x20;  RUN\_OPTIONAL\_RFECV = False  # Excluded from main fast run

&#x20;  RANDOM\_SEED = 42            # Fixed deterministic seed across all splits \& models

&#x20;  ```

3\. Run \*\*`Kernel -> Restart Kernel and Run All Cells`\*\*.

&#x20;  \* \*\*First clean run:\*\* Downloads all 170 quarters + `/failures` + `/institutions` from the FDIC API (\~15–25 minutes with 3 concurrent workers), constructs checkpoints, runs Polars/PySpark benchmarks, trains/tunes models, and saves tables/figures to `results/` (\~3 minutes compute time).

&#x20;  \* \*\*Subsequent runs:\*\* Automatically reuses Parquet checkpoints and completes end-to-end in \*\*\~2–3 minutes\*\*.



\---



\## 6. Citations \& Academic Integrity Disclosure



1\. \*\*Data Source:\*\* Federal Deposit Insurance Corporation (FDIC), \*BankFind Suite: Public API for Financial Institutions\* (`https://api.fdic.gov/banks`).

2\. \*\*Course Materials:\*\* Dr. Nguyen Tuan Long (National Economics University), \*BFT65B Big Data Analytics in Finance\* — starter ingestion script (`fdic\_download\_starter.py`) and Lecture 2 modules (\*Data Transform\*, \*Feature Selection\*, and \*Machine Learning Pipeline\*, 2026).

3\. \*\*Methodology References:\*\*

&#x20;  \* López de Prado, M. (2018). \*Advances in Financial Machine Learning\*. John Wiley \& Sons. (Chapter 7: Cross-Validation in Finance — Purged K-Fold and Embargo mechanisms).

&#x20;  \* Strobl, C., Boulesteix, A.-L., Zeileis, A., \& Hothorn, T. (2007). \*Bias in random forest variable importance measures: Illustrations, sources and a solution\*. BMC Bioinformatics, 8(1), 25.

4\. \*\*AI Tool Disclosure:\*\* Google Gemini was used as a collaborative coding and debugging assistant to optimize Polars lazy expressions, resolve Windows PySpark/Scikit-Learn compatibility issues, and structure the modular checkpoint pipeline. All design decisions, empirical validations, and written analyses were verified and authored by the student.

