# AM10_Group_Project
Group Project Repository for Study Group 11

Online News Popularity – Machine Learning Analysis

Predicting whether an online news article becomes Popular or remains Regular, using Random Forests and interpretable decision trees.

Project Overview

This project analyzes the Online News Popularity dataset and builds a machine learning model to classify news articles into:
- Popular (shares > 1400)
- Regular (shares ≤ 1400)

We explore feature engineering, visualization, clustering, dimensionality reduction, and supervised learning, with a focus on Random Forest classification and interpretable decision trees.

Dataset

The dataset contains 39,000+ online news articles and includes:
- Keyword statistics
- Sentiment metrics
- Engagement indicators
- Timing variables
- Topic categories

Preprocessing steps included:
- Dropping irrelevant LDA topic columns
- Converting weekday & topic dummy variables into categorical columns
- Creating the binary target: popular = 1 if shares > 1400 else 0

Methods Used

Exploratory Data Analysis
- Topic distribution by popularity
- Sentiment KDE plots
- Keyword/content statistics
- Correlation heatmaps

Unsupervised Learning
- K-Means clustering
- Hierarchical clustering
- PCA

Supervised Learning
Random Forest Classifier with:
n_estimators=300, max_depth=20, min_samples_split=20, min_samples_leaf=10, max_features='sqrt'

Why Random Forest?

Random Forest is ideal because it:
- Handles many numeric and categorical features
- Captures complex nonlinear relationships
- Avoids overfitting via bagging and random feature selection
- Allows interpretability through feature importances and surrogate trees

Interpreting the Decision Trees

Surrogate Tree
Shows dataset-level patterns:
- Keyword averages are the strongest predictor
- Topic + weekday interactions matter
- Weekend articles are less likely to be popular

Simplified RF Tree
Shows internal Random Forest logic:
- Low negativity + high vocabulary richness → more popular
- Keyword metrics strongly influence outcomes

Node Fields
- gini = class impurity (0 = pure)
- samples = rows reaching node
- value = [Regular, Popular]
- class = predicted label
- color = prediction (orange = Regular, blue = Popular)

Key Findings

- Keyword metrics are the top predictors
- Topic × weekday interactions strongly affect popularity
- Sentiment and lexical richness add predictive power
- Weekend posts underperform on average

Model Performance

The Random Forest achieved high accuracy with strong generalization thanks to optimized hyperparameters.

Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, SciPy


Data Source - https://archive.ics.uci.edu/dataset/332/online+news+popularity

Conclusion

Random Forest provided the best balance of performance, flexibility, and interpretability for predicting article popularity. This project demonstrates end-to-end ML: preprocessing, EDA, clustering, modeling, and interpretability.


