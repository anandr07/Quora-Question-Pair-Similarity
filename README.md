# Quora-Question-Pair-Similarity

Quora Question Pair Similarity project aims to identify duplicate questions using natural language processing. Leveraging machine learning algorithms like Logistic Regression, SGD Classifier, and XGBoost, the system achieves accurate classification, enhancing user experience by reducing redundancy in question content on the Quora platform.
![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/d7ef0672-2561-4b55-900d-e52f70217373)

# Problem Statement
Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world. Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term. Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

Data is taken from kaggle competition (2017) : https://kaggle.com/competitions/quora-question-pairs

# Dataset Description
The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset. Please note: as an anti-cheating measure, Kaggle has supplemented the test set with computer-generated question pairs. Those rows do not come from Quora, and are not counted in the scoring. All of the questions in the training set are genuine examples from Quora.

# Project Architecture
![Block_diagram_2](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/2ced8206-def6-458a-b623-169ed238ef27)

# File structure
![file_structure](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/bf2ab3e0-5268-4418-9982-0a4544bc7016)

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
- Random Model:
   - Log Loss for Training Data: 4.27141
   - Log Loss for Test Data: 3.95542
- Logistic Regression:
   - Train Log Loss: 0.46723
   - Test Log Loss: 0.47019
- SGDClassifier:
   - Train Log Loss: 0.44927
   - Test Log Loss: 0.45210
- NaiveBayesClassifier
   - Train Log Loss: 11.47686
   - Test Log Loss: 11.49861
- XGBoost
   - Train Log Loss: 0.23361
   - Test Log Loss: 0.35239
     
Log loss metrics reveal model performance. Random Model shows high log loss (4.27 train, 3.96 test). Logistic Regression and SGDClassifier perform well, while NaiveBayesClassifier indicates poor performance. XGBoost demonstrates effective generalization (0.23 train, 0.35 test).

# Importing Needed Libraries and accessing other py files(feature-extraction)
The project initiates data analysis and machine learning by importing essential Python libraries, including feature extraction, data visualization, and algorithms. It accesses specific functionalities from the 'feature_extraction' and 'ml_algorithms' modules for further use.

# Load the Data and Perform Data Analysis
Read CSV file into a Pandas DataFrame, display the first five rows and provide information about the dataset. It identifies missing values, visualizes them with a bar plot, and then drops the rows with null values, resulting in three rows being removed. The dataset initially has 404,290 entries, and after dropping rows with missing values, it has 404,287 entries. There are two null values in question 2 and one null value in question 1, dropping those rows.
![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/2b53febd-486d-4097-9bc8-c62308a77fe3)

# Distribution of data points among output classes (Similar and Non Similar Questions
- <b> Distribution of Duplicate and Non-duplicate Questions: </b>
  The bar plot illustrates the percentage distribution of questions categorized as duplicate and non-duplicate, checking for balance in the 'is_duplicate' column.

  ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/f7b9d6f6-a832-4443-83aa-27bdcd239590)

- <b>Number of Unique and Repeated Questions:</b>
  Analyzing the dataset reveals 537,929 unique questions. About 20.78% of questions appear more than once, with the maximum repetition being 157 times.
  
  ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/7ebc3018-e117-4ee5-8903-fdd4a06277e6)

- <b>Checking for Duplicates:</b>
  No rows are found where 'qid1' and 'qid2' are the same or interchanged, indicating no duplicate question pairs in the dataset.

- <b>Number of Occurrences of Each Question:</b>
  The histogram shows the log-scale distribution of the number of occurrences for each question, highlighting the maximum occurrence with a red dashed line.
  
  ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/fde24825-f2ba-4079-b084-838138cd5d96)
  The plot is close to a power-law distribution not exactly power-law but close to it.
