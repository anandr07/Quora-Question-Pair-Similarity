# Quora-Question-Pair-Similarity

Quora Question Pair Similarity project aims to identify duplicate questions using natural language processing. Leveraging machine learning algorithms like Logistic Regression, SGD Classifier, and XGBoost, the system achieves accurate classification, enhancing user experience by reducing redundancy in question content on the Quora platform.
![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/d7ef0672-2561-4b55-900d-e52f70217373)

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset Description](#dataset-description)
3. [Project Architecture](#project-architecture)
4. [File Structure](#file-structure)
5. [Data Details](#data-details)
6. [Performance Metric](#performance-metric)
7. [Load the Data and Perform Data Analysis](#load-the-data-and-perform-data-analysis)
8. [Top 10 Most Asked Questions on Quora](#top-10-most-asked-questions-on-quora)
9. [Distribution of Question Lengths](#distribution-of-question-lengths)
10. [Feature Engineering](#feature-engineering)
   - [Feature Extraction](#feature-extraction)
     
   - [Processing and Extracting Features](#processing-and-extracting-features)
    
   - [Pre-processing of Text](#pre-processing-of-text)
     
   - [Extracting Features](#extracting-features) 

   - [Visualizing in Lower Dimension using t-SNE](#visualizing-in-lower-dimension-using-t-sne)
   - [Featurizing Text Data with Tf-Idf Weighted Word-Vectors](#featurizing-text-data-with-tf-idf-weighted-word-vectors)
     
11. [Splitting into Train and Test Data](#splitting-into-train-and-test-data)
  
     
12. [Distribution of Output Variable in Train and Test Data](#distribution-of-output-variable-in-train-and-test-data)
  
     
13. [Results](#results)
   

# Problem Statement
Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world. Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term. Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

Data is taken from kaggle competition (2017) : https://kaggle.com/competitions/quora-question-pairs

# Dataset Description
The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset. Please note: as an anti-cheating measure, Kaggle has supplemented the test set with computer-generated question pairs. Those rows do not come from Quora, and are not counted in the scoring. All of the questions in the training set are genuine examples from Quora.

# Project Architecture
![Block_diagram_2](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/2ced8206-def6-458a-b623-169ed238ef27)

# File structure
![file_structure](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/04a2efa7-b48e-4a4e-9cab-3eeed961975d)

# Data Details
- id - the id of a training set question pair
- qid1, qid2 - unique ids of each question (only available in train.csv)
- question1, question2 - the full text of each question
- is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

# Businesss Constraints and Objectives
- Missclassification should be reduced.
- No Strict Low Latency requirements.
- Adaptable threshold for probability of classification.

# How this is a ML problem ?
- The objective is to classify whether two given questions are having same intention, typical classification problem.
- Create a application that takes two questions as input and in return tells if the questions have same meaning or not.

# Performance Metric
Metric(s): 
- Log-Loss
- Binary Confusion Matrix

# Importing Needed Libraries and accessing other py files(feature-extraction)
The project initiates data analysis and machine learning by importing essential Python libraries, including feature extraction, data visualization, and algorithms. It accesses specific functionalities from the 'feature_extraction' and 'ml_algorithms' modules for further use.

# Load the Data and Perform Data Analysis
Read CSV file into a Pandas DataFrame, display the first five rows and provide information about the dataset. It identifies missing values, visualizes them with a bar plot, and then drops the rows with null values, resulting in three rows being removed. The dataset initially has 404,290 entries, and after dropping rows with missing values, it has 404,287 entries. There are two null values in question 2 and one null value in question 1, dropping those rows.

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/7514279f-fb07-40b4-80dd-95ad61d42351)

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

# Top 10 Most asked questions on Quora:
- What are the best ways to lose weight?                                                                161
- How can you look at someone's private Instagram account without following them?                       120
- How can I lose weight quickly?                                                                        111
- What's the easiest way to make money online?                                                           88
- Can you see who views your Instagram?                                                                  79
- What are some things new employees should know going into their first day at AT&T?                     77
- What do you think of the decision by the Indian Government to demonetize 500 and 1000 rupee notes?     68
- Which is the best digital marketing course?                                                            66
- How can you increase your height?                                                                      63
- How do l see who viewed my videos on Instagram?                                                        61

# Distribution of Question Lengths:
Function determines the number of words in a sentence. It then applies this function to both 'question1' and 'question2' columns in a DataFrame. The resulting word counts are visualized using histograms for each question, allowing for a comparison of the distribution of question lengths.

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/d7c99aa6-f34d-4a1e-903f-60c54cf68516)

References for feature extraction:

- Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
- Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
- Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

# Feature Engineering
## Feature Extraction
____freq_qid1____ = Frequency of qid1's
____freq_qid2____ = Frequency of qid2's
____q1len____ = Length of q1
____q2len____ = Length of q2
____q1_n_words____ = Number of words in Question 1
____q2_n_words____ = Number of words in Question 2
____word_Common____ = (Number of common unique words in Question 1 and Question 2)
____word_Total____ =(Total num of words in Question 1 + Total num of words in Question 2)
____word_share____ = (word_common)/(word_Total)
____freq_q1+freq_q2____ = sum total of frequency of qid1 and qid2
____freq_q1-freq_q2____ = absolute difference of frequency of qid1 and qid2
## Feature Extraction after pre-processing.
Featurization (NLP and Fuzzy Features) Definition:

- <b>Token:</b> You get a token by splitting sentence a space
- <b>Stop_Word:</b> stop words as per NLTK.
- <b>Word:</b> A token that is not a stop_word


__Features__: - __cwc_min__ : Ratio of common_word_count to min lenghth of word count of Q1 and Q2
cwc_min = common_word_count / (min(len(q1_words), len(q2_words))

- __cwc_max__ : Ratio of common_word_count to max lenghth of word count of Q1 and Q2
cwc_max = common_word_count / (max(len(q1_words), len(q2_words))

- __csc_min__ : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2
csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))

- __csc_max__ : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))

- __ctc_min__ : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))

- __ctc_max__ : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))

- __last_word_eq__ : Check if First word of both questions is equal or not
last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])

- __first_word_eq__ : Check if First word of both questions is equal or not
first_word_eq = int(q1_tokens[0] == q2_tokens[0])

- __abs_len_diff__ : Abs. length difference
abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))

- __mean_len__ : Average Token Length of both Questions
mean_len = (len(q1_tokens) + len(q2_tokens))/2

- __fuzz_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

- __fuzz_partial_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

- __token_sort_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

- __token_set_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

- __longest_substr_ratio__ : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2
longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))

## Some additional features
- **ratio_q_lengths**: Calculates the ratio of the lengths of the two questions.
- **common_prefix**: Computes the length of the common prefix (the initial common sequence of characters) between the two questions.
- **common_suffix**: Calculates the length of the common suffix (the final common sequence of characters) between the two questions.
- **diff_words**: Calculates the absolute difference in the number of words between the two questions.
- **diff_chars**: Computes the absolute difference in the number of characters between the two questions.
- **jaccard_similarity**: Calculates the Jaccard similarity coefficient between the sets of words in the two questions.
- **longest_common_subsequence**: Computes the length of the longest common subsequence (LCS) between the two questions.

## Processing and Extracting Features
Sets the file path for a CSV file named "data_with_features.csv." It also specifies the number of rows to be used for training the model, with the variable rows_to_train set to 100,000. This number can be adjusted based on specific needs or dataset sizes. 

## Pre-processing of Text
Preprocessing:
- Removing html tags
- Removing Punctuations
- Performing stemming
- Removing Stopwords
- Expanding contractions etc.
  
## Extracting Features
Function generates features for the specified number of rows (rows_to_train), and then saves the data to the file. The resulting DataFrame is displayed, showing the first five rows with additional features extracted from the original dataset. The added features include various characteristics like frequency, length, and similarity ratios between the questions.

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/b8a6ff92-26e2-4a66-8019-58ce0e51b7dd)

### Check for questions with 2 words or less than 2 words
Filters sentences from the DataFrame based on the condition that either 'q1' or 'q2' should have two words or fewer. The filtered data is stored in a new DataFrame called filtered_data. Then prints details for the first 10 filtered sentences and the total number of sentences meeting the criteria. This filtering process helps inspect and understand specific characteristics of sentences with a low word count in either 'q1' or 'q2'.

Provide insights into the distribution of question lengths, highlighting the minimum lengths and the number of questions with the minimum length in both 'question1' and 'question2'.

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/87392c85-36a2-4c62-a868-edcb5bf4a676)

- Kullback-Leibler (KL) Divergence helps analyze the discriminatory power of <b> 33 different features </b> used, in distinguishing between duplicate and non-duplicate pairs in a dataset.
- This visualization allows us to compare the distribution of each feature for duplicate and non-duplicate pairs, providing insights into the characteristics that might differentiate between the two categories.
- Violin plots show the distribution shape, while Density plots provide a smooth estimate of the probability density function for each class.

  ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/d9b122c2-5a1c-49f0-b68f-bd66a3f60eda)

  ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/41b186b2-ba08-494e-8f4d-4ad9b284b529)

  ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/5294e682-eb95-4cd7-8851-abb455e20179)

  ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/56f68d71-7417-4e5f-a26c-eb39c74055e9)

This visualization helps identify features with high inverted KL Divergence, highlighting those that exhibit significant differences between duplicate and non-duplicate pairs. Higher values indicate features that are more discriminative in distinguishing between the two classes.

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/5badaa27-b2ea-4b7f-9cd6-668d87983f02)

Identifies the bottom (least discriminative) 5 features based on their calculated KL Divergence and then creates a pair plot for these features along with the target variable 'is_duplicate'. 

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/bac5f6f1-f8d3-40c8-a9d5-fe6ecae495bd)

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/e830345b-2eab-4943-8241-5548cba168ab)

### Important features in differentiating Duplicate(Similar) and Non-Duplicate(Dissimilar) Questions.
- Distribution of q1len for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
- Distribution of q2len for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
- Distribution of q1_n_words for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
- Distribution of q2_n_words for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
- Distribution of word_Total for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
- Distribution of word_share for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.

### Visualizing in lower dimension using t-SNE

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/068bb8cd-3415-40a7-adfd-ff7d5aaa30d5)

### Featurizing text data with Tf-Idf weighted word-vectors
- Extracts features for each question in the dataset using spaCy, considering the semantic meaning of words and their TF-IDF weights. These features are then added to the DataFrame for further analysis.
- Loads processed features, drops unnecessary columns, extracts features for Question 1 and Question 2, and displays information about the features in separate DataFrames.

-  Consolidates the features from different DataFrames into a single DataFrame and saves it to the specified CSV file for further use.
-  The code replaces non-numeric values in the DataFrame with NaN, checks for the presence of NaN values, and prints the count of NaN values in each column after replacement.

- Converts all features to numeric format, handling any errors by coercing non-numeric values to NaN.

### Due to lack of Computation Power the models are trained on 100,000 Rows.

- Checks if there are any NA (missing) values in the DataFrame after converting features to numeric format. If present, it prints "NA Values Present"; otherwise, it prints "No NA Values Present." It then displays the number of NaN values in each column after the conversion. Additionally, it converts the target variable y_true to a list of integers and shows the first few rows of the DataFrame.

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/09414d7e-e0aa-4770-9c15-25bb1c3cc201)

## Splitting into Train and Test Data
Train Data : 70%
Test Data : 30%

## Distribution of Output Variable in Train and Test Data
The left subplot shows the distribution in the training data, while the right subplot shows the distribution in the testing data. This helps to understand the balance or imbalance in the classes of the output variable.

![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/fb0a692b-6bba-4356-ac6d-81f838fa7b89)

# Results
- <b> Random Model </b>:
   - Log Loss for Training Data: 4.27141
   - Log Loss for Test Data: 3.95542
- <b> Logistic Regression </b>:
   - Train Log Loss: 0.46723
   - Test Log Loss: 0.47019

   ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/373166de-e3bb-448c-a98f-be2dc9ab9a00)

- <b> SGDClassifier </b>:
   - Train Log Loss: 0.44927
   - Test Log Loss: 0.45210
  
   ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/2063189e-f6ed-48fe-a355-f9d8761bbf90)

- <b> NaiveBayesClassifier </b>:
   - Train Log Loss: 11.47686
   - Test Log Loss: 11.49861

   ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/cde92940-3723-4659-ab08-df0126109c90)

- <b> XGBoost </b>:
   - Train Log Loss: 0.23361
   - Test Log Loss: 0.35239

   ![image](https://github.com/anandr07/Quora-Question-Pair-Similarity/assets/66896800/67c8502c-1a5c-4819-ab68-5d3315f5af25)

     
Log loss metrics reveal model performance. Random Model shows high log loss (4.27 train, 3.96 test). Logistic Regression and SGDClassifier perform well, while NaiveBayesClassifier indicates poor performance. XGBoost demonstrates effective generalization (0.23 train, 0.35 test).
