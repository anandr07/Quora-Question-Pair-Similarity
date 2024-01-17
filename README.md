# Quora-Question-Pair-Similarity

Quora Question Pair Similarity project aims to identify duplicate questions using natural language processing. Leveraging machine learning algorithms like Logistic Regression, SGD Classifier, and XGBoost, the system achieves accurate classification, enhancing user experience by reducing redundancy in question content on the Quora platform.
![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/d7ef0672-2561-4b55-900d-e52f70217373)

# Problem Statement
Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world. Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term. Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

Data is taken from kaggle competition (2017) : https://kaggle.com/competitions/quora-question-pairs

# Dataset Description
The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset. Please note: as an anti-cheating measure, Kaggle has supplemented the test set with computer-generated question pairs. Those rows do not come from Quora, and are not counted in the scoring. All of the questions in the training set are genuine examples from Quora.

# Project Architecture
![Block_diagram](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/7fcedb9a-90b4-476c-801f-cde038c29863)

# File structure
![file_structure](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/f349632e-e24f-48b5-983b-5da69fe5f77c)

# Data Details
id - the id of a training set question pair
qid1, qid2 - unique ids of each question (only available in train.csv)
question1, question2 - the full text of each question
is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.
# Businesss Constraints and Objectives
Missclassification should be reduced.
No Strict Low Latency requirements.
Adaptable threshold for probability of classification.
# How this is a ML problem ?
The objective is to classify whether two given questions are having same intention, typical classification problem. Create a application that takes two questions as input and in return tells if the questions have same meaning or not.

# Performance Metric
Metric(s): Log-Loss and Binary Confusion Matrix
