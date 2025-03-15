# Flight Fare Forecasting – Machine Learning for Cost-Effective Travel

Welcome to our Flight Fare Forecasting project! This repository explores predictive modeling for airline ticket pricing using a large dataset from [Kaggle](https://www.kaggle.com/datasets/dilwong/flightprices). Our primary goal is to develop models that accurately predict flight fares using a variety of machine learning techniques—ranging from classical regression approaches to more advanced gradient-boosting algorithms and distributed training strategies.

---

## Repository Structure


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


### Folders and Contents

1. Multiple_Model_Framework:  
   - Contains multiple modeling approaches using various machine learning techniques.
   - Includes an Abstract Model Training Template and Weak Baselines notebook.
   - Stores partitioned pipeline outputs (p1_pipeline_outputs, p2_pipeline_outputs, p3_pipeline_outputs) for different subsets of the data.

2. Single_Model_Framework:  
   - 01_Data_Ingestion_and_Preprocessing.ipynb – Loads the dataset, handles missing values, and prepares it for modeling.
   - 02_EDA_and_Visualization.ipynb – Conducts exploratory data analysis and visualizes key trends.
   - 03_Modeling_and_Evaluation.ipynb – Implements baseline models (Linear Regression, Random Forest) and evaluates their performance.
   - 04_XGBoost_with_Dask_(GPU).ipynb – Trains XGBoost using Dask for scalable, distributed learning.
   - 05_CatBoost.ipynb – Implements CatBoost for additional performance comparisons.

3. Milestones:  
   - Contains milestone reports detailing the project’s progress:
     - Milestone 1: Project Abstract with background and motivation.
     - Milestone 2: Progress report on data pipeline and early modeling.
     - Milestone 3: Presentation slides summarizing updates and findings.

---

## Project Overview

### Motivation

- Airline ticket prices vary due to multiple complex factors (seasonality, demand, flight distance, class, etc.).  
- By applying machine learning to a large flight-pricing dataset, we aim to help travelers find cost-effective fares and assist the airline industry in optimizing pricing.

### Dataset

- Source: [Kaggle – Flight Prices](https://www.kaggle.com/datasets/dilwong/flightprices)  
- Size & Scope: Approximately 82+ million rows of itineraries from major U.S. airports within a specific 2022 timeframe.  
- Key Features: Departure/arrival airports, flight dates, base fare, total fare (including taxes/fees), distance, travel duration, airline codes, seat availability, and more.

### Methods & Models

- Data Pipeline:  
  - PySpark used for efficient ingestion and cleaning due to the dataset’s large size.  
  - Feature engineering included extracting time-related features (day of week, month), handling missing distances, boolean feature encoding, and route-specific transformations.
- Exploratory Data Analysis (EDA):  
  - Distribution of base fares vs. total fares, flight counts across different airports, seat availability, and ratio analysis.
- Models & Techniques:  
  - Linear Regression & Random Forest for baseline modeling.  
  - Gradient Boosting (XGBoost, LightGBM, CatBoost) for improved accuracy with structured data.  
  - Distributed Training (Dask + XGBoost on GPU) for scalability on large datasets.

### Current Findings

- Base vs. Total Fare: The total fare is typically a small markup (~15%) above the base fare, but some variability remains.  
- Performance:  
  - Random Forest and XGBoost outperform simpler baselines (like linear regression).  
  - Distributed GPU training significantly reduces computation time for large-scale experiments.

---

## Getting Started

### 1. Clone the Repository


git clone https://github.com/nikopastore/flight-fare-forecasting.git
cd flight-fare-forecasting


### 2. Download the Dataset


kaggle datasets download -d dilwong/flightprices
unzip flightprices.zip

- Place the itineraries.csv file in a suitable location (the notebooks reference it directly).

### 3. Environment Setup

- Recommended: Use a Conda or virtual environment.
- Key Libraries:
  - pyspark
  - xgboost (with GPU support if available)
  - catboost
  - dask[complete] (for distributed operations)
  - pandas, numpy, matplotlib, seaborn, scikit-learn
- GPU Acceleration: Ensure you have CUDA-compatible drivers and the correct version of XGBoost.

> Note 1: If you prefer to work with Google Colab, most notebooks can be executed on a regular CPU (preferably with high RAM). However, the XGBoost and CatBoost notebooks require a TPU or GPU environment for proper execution.
> 
> Note 2: Running the notebooks locally in Jupyter may result in faster runtimes, depending on your device specifications.

---
## Instructions for Producing Test Results

Test results are generated on a per-partition basis (reflected in "Table 1" in our report). To produce the results for a given partition, follow these steps:

1. Open our repo's "Multiple_Model_Framework" folder.
2. To generate weak baseline results across all partitions, open the "Weak_Baselines.ipynb" notebook in Google Colab with a CPU runtime. This may take around 3-4 hours. To reduce runtime, one may generate weak baseline results for only a single data partition. To do this, navigate to the "Partition Selection" -> "Procedure" section, which contains only a single code cell. In this cell, manually adjust the partitions to be considered using the list variable named "pnos." E.g., to only evaluate data partition 1, set pnos = [1].
3. To generate other model results for a given partition, open one of the pipeline output folders (e.g., "p1_pipeline_outputs") and download the relevant files.
4. In the following example for partition 1, the following files would need to be downloaded: "PCA_part_1.tar.gz", "prePCA_part_1.tar.gz", "stats_1.npy". nalogous files would need to be downloaded if evaluating other partitions. These files would need to be uploaded to the "content" folder of Google Colab. A new folder named "PCA" would also need to be created in the "content" folder in Colab, inside which the "stats_1.np" file would be placed.
5. With this setup, proceed to download one of the available models under the "Multiple_Model_Framework" -> "p1_pipeline_outputs" folder in our GitHub repo. E.g., if evaluation of RandomForestRegression on partition 1 is desired, download "Partition1_RandomForestRegressor.ipynb" and upload it to the "content" folder in Colab.
6. Proceed to run the notebook on Colab to generate test results; this will likely take several hours. Afterwards, basic statstics for prediction squared errors are output towards the end of the notebook. This reveals the mean squared error of predictions on the partitionn.

---
## Milestone Reports

You can find our detailed milestone documents in the milestones folder:

- Milestone 1 - Project Abstract.pdf  
  Contains the project overview, background, literature review, and initial approach.

- Milestone 2 - 1st Progress Report.pdf  
  Includes progress on data pipeline, EDA, feature engineering, and initial modeling strategies, along with team contributions and risk mitigation plans.

- Milestone 3 Slides - 2nd Progress Report.pptx  
  Summarizes key updates, modeling improvements, and next steps.

---
