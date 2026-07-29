# How the Recommendation System Works — Step by Step
**HYM Product Recommender**  
*July 2026*

---

## Table of Contents
- [Introduction](#introduction)
- [Step 0 — Data Preparation (identical for all 4 models)](#step-0--data-preparation-identical-for-all-4-models)
- [Model 1 — Random](#model-1--random)
- [Model 2 — Popular](#model-2--popular)
- [Model 3 — Cluster (Similarity-based Grouping)](#model-3--cluster-similarity-based-grouping)
- [Model 4 — XGBoost (The Main Model)](#model-4--xgboost-the-main-model)
  - [4.1 — Feature Engineering](#41--feature-engineering)
  - [4.2 — Collaborative Signal (BPR)](#42--collaborative-signal-bpr)
  - [4.3 — Generating Training Examples: Positives and Negatives](#43--generating-training-examples-positives-and-negatives)
  - [4.4 — Dataset Construction and Training](#44--dataset-construction-and-training)
  - [4.5 — Selecting Candidates for Recommendation](#45--selecting-candidates-for-recommendation)
  - [4.6 — Scoring and Ranking](#46--scoring-and-ranking)
  - [4.7 — Hyperparameter Tuning (Optuna, Optional)](#47--hyperparameter-tuning-optuna-optional)
- [Evaluating Model Performance (identical for all 4)](#evaluating-model-performance-identical-for-all-4)
- [Visual Summary](#visual-summary)

---

## Introduction

This document provides a clear, step-by-step explanation of how each of the four recommendation models in this project operates—from raw data loading to generating and evaluating recommendation quality.

The 4 models are:
1. **Random**: Recommends items completely at random (serves as a baseline floor).
2. **Popular**: Recommends top-selling items universally to all users.
3. **Cluster**: Groups similar items and recommends products based on the cluster of the user's historical purchases.
4. **XGBoost**: The core advanced model. It learns from historical data to score and rank which items a specific user is most likely to purchase.

> **Note:** All 4 models are trained and evaluated under identical conditions (same user sample, same train/test split, and identical evaluation metrics) to ensure a fair comparison.

---

## Step 0 — Data Preparation (identical for all 4 models)

Before training any model, the pipeline executes the following pipeline setup:
1. **Load Data:** Import customers, items, and transaction logs from the source files.
2. **Sample Customers (e.g., 1,000 users):** Downsample the dataset to accelerate iteration speed during development.
3. **Filter Inactive Users:** Exclude users with minimal transaction history or long periods of inactivity to retain users with sufficient interaction data.
4. **Train/Test Split (*“leave-one-out”*):** For each customer, the most recent purchase is set aside as the test target (what the model needs to predict), while the remaining history is used for training.
5. **Build Auxiliary Resources:**
   * Customer purchase history mappings.
   * **Co-visitation Matrix:** Captures items frequently bought together (if many customers who bought item A also bought item B, A and B are co-visited).
   * Item categories, enabling coarse-grained evaluation when exact item predictions miss.

Once step 0 completes, any of the 4 models can be trained and evaluated.

---

## Model 1 — Random

The simplest baseline model, designed to establish the absolute performance floor (*“if a model cannot beat random chance, something is fundamentally wrong”*).

**Steps:**
1. Retrieve the entire item catalog.
2. For each test customer, randomly select 12 unique items.
3. Output these 12 items as recommendations and evaluate how many match the customer's ground truth purchase.

*Learns nothing from data; serves purely as a benchmark control.*

---

## Model 2 — Popular

A static non-personalized baseline that recommends identical items to every user.

**Steps:**
1. Aggregate sales counts for every item in the training split.
2. Identify the top 12 best-selling items.
3. Output these top 12 items to all test users without customization.

*Offers a stronger baseline than Random (popular items inherently carry higher prior probability of purchase), but lacks individual personalization.*

---

## Model 3 — Cluster (Similarity-based Grouping)

Groups items with shared attributes together and recommends unpurchased items from the clusters corresponding to a user's purchase history.

**Steps:**
1. **Feature Extraction:** Compute item-level features (average price, sales volume, mean buyer age, online vs. offline sales distribution, item tenure, etc.).
2. **Feature Scaling (Standardization):** Normalize features to a unified scale to ensure distance-based clustering algorithms treat all feature dimensions equitably.
3. **K-Means Clustering:** Partition items into $K$ distinct clusters (e.g., high-end seasonal apparel might form one cluster, budget essentials another).
4. **Customer Profiling:** Compute a user profile vector by averaging the feature representations of their past purchases.
5. **Candidate Retrieval:** Identify candidate items belonging to the same clusters as the user's past purchases (excluding already purchased items).
6. **Ranking:** Sort candidates by cosine similarity relative to the user profile, using mean buyer age as a secondary tie-breaker.
7. Output the top 12 ranked items.

*Incorporates personal purchase context, but relies purely on content/attribute similarity rather than collaborative user behavior.*

---

## Model 4 — XGBoost (The Main Model)

The flagship model. Instead of applying static heuristic rules, XGBoost trains a machine learning model to estimate a preference score for individual user-item candidate pairs and ranks them accordingly.

### 4.1 — Feature Engineering
Computes three distinct categories of numerical features:
* **Item Features:** Average price, total volume, past 30-day volume, catalog age, category popularity rank, etc.
* **Customer Features:** Transaction frequency, average spending, preferred product category, days since last purchase, age group, etc.
* **Interaction Features (Customer-Item):** Price affinity gap (difference between item price and user's typical spend), category match indicators, etc.

### 4.2 — Collaborative Signal (BPR)
Trains an auxiliary **Bayesian Personalized Ranking (BPR)** matrix factorization model to capture collaborative filtering dynamics (*"customers with similar purchase histories also bought this"*). The output score (`bpr_score`) is fed directly into the XGBoost feature matrix.

### 4.3 — Generating Training Examples: Positives and Negatives
XGBoost requires labeled positive and negative pairs to learn pairwise ranking bounds:
* **Positives:** Observed customer purchases.
* **Negatives:** Items the customer did not purchase. To prevent training trivial classifiers, negatives are constructed using a mixture of 4 strategies:
  * *Popularity Negatives (Easy):* Randomly sampled popular items not bought by the user.
  * *Cluster Negatives (Hard):* Unpurchased items from the user's preferred item clusters.
  * *Co-visitation Negatives (Hard):* Items frequently co-purchased with the user's history, but not bought by this user.
  * *BPR Negatives (Hard):* Unpurchased items assigned high score vectors by the collaborative BPR model.

Combining easy and hard negative samples forces the model to learn fine-grained preferences rather than basic popularity patterns.

### 4.4 — Dataset Construction and Training
1. Assemble all positive and negative pairs into a unified feature matrix containing domain features (4.1) and `bpr_score` (4.2).
2. Carve out an internal validation set to monitor for overfitting.
3. Train an **XGBRanker** model optimized specifically for pairwise or listwise ranking objectives.

### 4.5 — Selecting Candidates for Recommendation
During inference, candidate selection follows one of two configurable strategies:
* **Generic Pool:** Top $N$ global best-sellers sent to all users. Simple, but requires scoring large candidate sets (high computational cost).
* **Hybrid Personalized Candidates:** A blend of popular items, BPR nearest neighbors, cluster items, and co-visited items tailored per user. Achieves higher accuracy while scoring significantly fewer candidate items (lower computational cost).

### 4.6 — Scoring and Ranking
1. The trained XGBRanker scores all generated candidate pairs for a given user.
2. Candidates are sorted in descending order of predicted score.
3. The top 12 ranked items are served as recommendations.

### 4.7 — Hyperparameter Tuning (Optuna, Optional)
A standalone optimization script (`optuna_search.py`) runs automated Bayesian search loops over hyperparameter spaces (tree depth, regularization coefficients, negative sampling ratios, etc.) on a reduced dataset to maximize MAP@12. Optimal hyperparameter configurations are subsequently ported back to full-scale training pipelines.

---

## Evaluating Model Performance (identical for all 4)

After generating top-12 recommendation vectors across all models, output lists are benchmarked against ground truth purchases using the following metrics:

* **MAP@12 (Primary Metric):** Evaluates overall accuracy while heavily penalizing models for pushing relevant items down the ranking list.

$$\text{MAP@12} = \frac{1}{U} \sum_{u=1}^{U} \left( \frac{1}{\min(m_u, 12)} \sum_{k=1}^{12} P_u(k) \cdot rel_u(k) \right)$$

Where:
* $P_u(k) = \frac{\text{Hits in Top-}k}{k} \quad \text{(Precision at rank } k\text{)}$
* $rel_u(k) \in \{0, 1\} \quad \text{(Binary relevance: 1 if item at rank } k \text{ was purchased, 0 otherwise)}$
* $m_u = \text{Total ground truth purchases for user } u$

* **Hit Count / Hit Rate (%):** Measures the absolute number and percentage of test users who received at least one correct item recommendation within their top 12 list.
* **Category Hit Rate:** A relaxed precision metric evaluating whether a recommended item matches the product category of the actual purchase, even if the exact SKU differs.
* **Candidate Recall:** The ratio of target items captured within the candidate retrieval pool vs. total target items. Represents the theoretical upper bound (ceiling) achievable by the ranker.
* **Relative Hit Rate (%):** Ratio of hit items to ground truth purchases present *inside* the candidate pool. Isolates the ranker's sorting efficiency from candidate retrieval bottlenecks.

> **MLflow Tracking:** All evaluation runs log parameters, candidate configuration settings, and metrics automatically to MLflow for systematic model comparison.

---

## Visual Summary

| Model | Learns from Data? | Personalized per User? | Core Idea |
| :--- | :---: | :---: | :--- |
| **Random** | No | No | Select items completely at random |
| **Popular** | No | No | Recommend global best-sellers universally |
| **Cluster** | Partially (Unsupervised) | Yes | Recommend items from clusters matching past purchases |
| **XGBoost** | Yes | Yes | Predict scores and rank candidates using item features, user features, and collaborative BPR signals |
