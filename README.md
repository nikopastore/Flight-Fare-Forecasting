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

1. **milestones/**  
   - **Milestone 1 - Project Abstract.pdf**  
     A high-level overview of the project’s background, motivation, and initial approach.  
   - **Milestone 2 - 1st Progress Report.pdf**  
     More in-depth details on the data pipeline, early exploratory analyses, and the first modeling attempts.  

2. **notebooks/**  
   - **01 Data Ingestion and Preprocessing.ipynb**  
     Demonstrates how to ingest the dataset (using both Pandas and PySpark), handle missing values, clean and format columns, and save the preprocessed data for downstream tasks.  
   - **02 EDA and Visualization.ipynb**  
     Explores key features of the dataset: airport distributions, fare breakdowns, and initial univariate and bivariate trends. Showcases plotting flight counts, analyzing base fare vs. total fare, and investigating boolean features such as basic economy and non-stop.  
   - **03 Modeling and Evaluation.ipynb**  
     Builds upon the processed dataset to train and evaluate initial models (Linear Regression, Random Forest). Demonstrates a feature-engineering pipeline, splitting data into training and test sets, and calculating performance metrics (e.g., MSE).  
   - **04 XGBoost with Dask (GPU).ipynb**  
     Leverages GPU-accelerated XGBoost and Dask to handle large-scale data. Shows how to train gradient-boosted trees in a distributed environment, enabling faster experimentation on big datasets.

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
  - **Gradient Boosting (XGBoost, LightGBM)** for improved accuracy with structured data.  
  - **Distributed Training** (Dask + XGBoost on GPU) for scalability on large datasets.

### Current Findings

- **Base vs. Total Fare:** The total fare is typically a small markup (~15%) above the base fare, but some variability remains.  
- **Performance:**  
  - Random Forest and XGBoost outperform simpler baselines (like linear regression).  
  - Distributed GPU training significantly reduces computation time for large-scale experiments.

---

## Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/nikopastore/flight-fare-forecasting.git
   cd flight-fare-forecasting
2. **Download the Dataset**

    ```bash
    kaggle datasets download -d dilwong/flightprices
    unzip flightprices.zip
  - Place the itineraries.csv file in a suitable location (the notebooks reference it directly).

3. **Environment Setup**
  - **Recommended**: Use a Conda or virtual environment.
  - **Key Libraries**:
    - pyspark
    - xgboost (with GPU support if available)
    - dask[complete] (for distributed operations)
    - pandas, numpy, matplotlib, seaborn, scikit-learn
  - For GPU acceleration, ensure you have CUDA-compatible drivers and the correct version of XGBoost.

## 4. Running the Notebooks

- **01 Data Ingestion and Preprocessing.ipynb**  
  - Update file paths as needed.  
  - Run cells sequentially; a cleaned CSV will be generated if you follow the pipeline.

- **02 EDA and Visualization.ipynb**  
  - Loads the same CSV.  
  - Generates histograms, bar plots, and other EDA visuals.

- **03 Modeling and Evaluation.ipynb**  
  - Demonstrates feature engineering, train/test splitting, and training baseline models (Linear Regression, Random Forest).  
  - Outputs MSE metrics for each approach.

- **04 XGBoost with Dask (GPU).ipynb**  
  - Requires GPU environment (e.g., Google Colab with GPU runtime or a local machine with CUDA).  
  - Trains an XGBoost regressor in distributed fashion to handle large subsets of the data efficiently.
 
---

## Milestone Reports

You can find our detailed milestone documents in the **milestones** folder:

- **Milestone 1 - Project Abstract.pdf**  
  Contains the project overview, background, literature review, and initial approach.

- **Milestone 2 - 1st Progress Report.pdf**  
  Includes progress on data pipeline, EDA, feature engineering, and initial modeling strategies, along with team contributions and risk mitigation plans.

