# Flight Fare Forecasting – Machine Learning for Cost-Effective Travel

Welcome to our Flight Fare Forecasting project! This repository explores predictive modeling for airline ticket pricing using a large dataset from [Kaggle](https://www.kaggle.com/datasets/dilwong/flightprices). Our primary goal is to develop models that accurately predict flight fares using a variety of machine learning techniques—ranging from classical regression approaches to more advanced gradient-boosting algorithms and distributed training strategies.

---

## Repository Structure

```plaintext
.
|
+---Multiple_Model_Framework
|   |   Abstract_Model_Training_Template.ipynb
|   |   Weak_Baselines.ipynb
|   |
|   +---p1_pipeline_outputs
|   |       Partition1_First_Run.ipynb
|   |       Partition1_LinearRegression.ipynb
|   |       Partition1_RandomForestRegression.ipynb
|   |       Partition1_XGBoost.ipynb
|   |       PCA_part_1.tar.gz
|   |       prePCA_part_1.tar.gz
|   |       stats_1.npy
|   |
|   +---p2_pipeline_outputs
|   |       Partition2_LinearRegression.ipynb
|   |       Partition2_RandomForestRegression.ipynb
|   |       PCA_part_2.tar.gz
|   |       prePCA_part_2.tar.gz
|   |       stats_2.npy
|   |
|   \---p3_pipeline_outputs
|           Partition3_LinearRegression.ipynb
|           Partition3_RandomForestRegression.ipynb
|           PCA_part_3.tar.gz
|           prePCA_part_3.tar.gz
|           stats_3.npy
|
+---Single_Model_Framework
|       01_Data_Ingestion_and_Preprocessing.ipynb
|       02_EDA_and_Visualization.ipynb
|       03_Modeling_and_Evaluation.ipynb
|       04_XGBoost_with_Dask_(GPU).ipynb
|       05_CatBoost.ipynb
\---milestones
        Milestone 1 - Jan 20 - Project Abstract.pdf
        Milestone 2 - Feb 2 - 1st Progress Report.pdf
        Milestone 3 Slides - Feb 24 - 2nd Progress Report.pptx
```

### Folders and Contents

1. **Multiple_Model_Framework/**  
   - Contains multiple modeling approaches using various machine learning techniques.
   - Includes an **Abstract Model Training Template** and **Weak Baselines** notebook.
   - Stores partitioned pipeline outputs (**p1_pipeline_outputs, p2_pipeline_outputs, p3_pipeline_outputs**) for different subsets of the data.

2. **Single_Model_Framework/**  
   - **01_Data_Ingestion_and_Preprocessing.ipynb** – Loads the dataset, handles missing values, and prepares it for modeling.
   - **02_EDA_and_Visualization.ipynb** – Conducts exploratory data analysis and visualizes key trends.
   - **03_Modeling_and_Evaluation.ipynb** – Implements baseline models (Linear Regression, Random Forest) and evaluates their performance.
   - **04_XGBoost_with_Dask_(GPU).ipynb** – Trains XGBoost using Dask for scalable, distributed learning.
   - **05_CatBoost.ipynb** – Implements CatBoost for additional performance comparisons.

3. **milestones/**  
   - Contains milestone reports detailing the project’s progress:
     - **Milestone 1:** Project Abstract with background and motivation.
     - **Milestone 2:** Progress report on data pipeline and early modeling.
     - **Milestone 3:** Presentation slides summarizing updates and findings.

---

## Project Overview

### Motivation

- Airline ticket prices vary due to multiple complex factors (seasonality, demand, flight distance, class, etc.).  
- By applying machine learning to a large flight-pricing dataset, we aim to help travelers find cost-effective fares and assist the airline industry in optimizing pricing.

### Dataset

- **Source:** [Kaggle – Flight Prices](https://www.kaggle.com/datasets/dilwong/flightprices)  
- **Size & Scope:** Approximately 82+ million rows of itineraries from major U.S. airports within a specific 2022 timeframe.  
- **Key Features:** Departure/arrival airports, flight dates, base fare, total fare (including taxes/fees), distance, travel duration, airline codes, seat availability, and more.

### Methods & Models

- **Data Pipeline:**  
  - PySpark used for efficient ingestion and cleaning due to the dataset’s large size.  
  - Feature engineering included extracting time-related features (day of week, month), handling missing distances, boolean feature encoding, and route-specific transformations.
- **Exploratory Data Analysis (EDA):**  
  - Distribution of base fares vs. total fares, flight counts across different airports, seat availability, and ratio analysis.
- **Models & Techniques:**  
  - **Linear Regression & Random Forest** for baseline modeling.  
  - **Gradient Boosting (XGBoost and CatBoost)** for improved accuracy with structured data.  
  - **Distributed Training** (Dask + XGBoost on GPU) for scalability on large datasets.

### Current Findings

- **Base vs. Total Fare:** The total fare is typically a small markup (~15%) above the base fare, but some variability remains.  
- **Performance:**  
  - Random Forest and XGBoost outperform simpler baselines (like linear regression).  
  - Distributed GPU training significantly reduces computation time for large-scale experiments.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nikopastore/flight-fare-forecasting.git
cd flight-fare-forecasting
```

### 2. Download the Dataset

```bash
kaggle datasets download -d dilwong/flightprices
unzip flightprices.zip
```
- Place the `itineraries.csv` file in a suitable location (the notebooks reference it directly).

### 3. Environment Setup

- **Recommended:** Use a Conda or virtual environment.
- **Key Libraries:**
  - `pyspark`
  - `xgboost` (with GPU support if available)
  - `dask[complete]` (for distributed operations)
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **GPU Acceleration:** Ensure you have CUDA-compatible drivers and the correct version of XGBoost.

### 4. Running the Notebooks

- **Single_Model_Framework/** notebooks provide a step-by-step approach to data preparation, visualization, and model training.
  - Update file paths as needed before running.

- **Multiple_Model_Framework/** notebooks contain additional model frameworks with partitioned training data.

---

## Milestone Reports

You can find our detailed milestone documents in the **milestones** folder:

- **Milestone 1 - Project Abstract.pdf**  
  Contains the project overview, background, literature review, and initial approach.

- **Milestone 2 - 1st Progress Report.pdf**  
  Includes progress on data pipeline, EDA, feature engineering, and initial modeling strategies, along with team contributions and risk mitigation plans.

- **Milestone 3 Slides - 2nd Progress Report.pptx**  
  Summarizes key updates, modeling improvements, and next steps.

---