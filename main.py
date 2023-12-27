#%%[markdown]
## Quora Question Pairs 
# Identifying questions on quora with same intent 

#%%[markdown]
### Problem Statement

# Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. 
# It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

# Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. 
# Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. 
# Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

# Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. 
# Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

# %%[markdown]

# Data is taken from kaggle competition (2017) : https://kaggle.com/competitions/quora-question-pairs 

#%%[markdown]

### Dataset Description

# The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. 
# The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. 
# Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. 
# We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.

# Please note: as an anti-cheating measure, Kaggle has supplemented the test set with computer-generated question pairs. Those rows do not come from Quora, and are not counted in the scoring. 
# All of the questions in the training set are genuine examples from Quora. 

#%%[markdown]

### Data Details

# 1. id - the id of a training set question pair
# 2. qid1, qid2 - unique ids of each question (only available in train.csv)
# 3. question1, question2 - the full text of each question
# 4. is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

#%%[markdown]

### Businesss Constraints and Objectives

# 1. Missclassification should be reduced.
# 2. No Strict Low Latency requirements.
# 3. Adaptable threshold for probability of classification.

#%%[markdown]

### How this is a ML problem ? 

# The objective is to classify whether two given questions are having same intention, typical classification problem. 
# Create a application that takes two questions as input and in return tells if the questions have same meaning or not.

#%%[markdown]

### Performance Metric

# Metric(s):
# Log-Loss and Binary Confusion Matrix

#%%[markdown]

### Importing Needed Libraries and accessing other py files(feature-extraction).

#%%

import os 
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from feature_extraction import process_data
import seaborn as sns

# # Getting the current script's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Adding the parent directory to the Python path
# sys.path.append(os.path.dirname(current_dir))

# # Import other python files.
#

#%%[markdown]

### Load the Data and Perform Data Analysis

#%%
data = pd.read_csv('data/train.csv')

#%%[markdown]
# First 5 rows of data

# %%
data.head(5)

# %%
print(f"Number of Observations in data are {data.shape[0]}")

#%%
print(data.info())

missing_values_count = data.isnull().sum()

# Create a bar plot of missing values
plt.figure(figsize=(10, 6))
missing_values_count.plot(kind='bar', color='skyblue')

# Add labels and title
plt.title('Number of Missing Values per Column', fontsize=16)
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Number of Missing Values', fontsize=14)

# Show the plot
plt.show()

#%%[markdown]
# Note: There are two null values in question 2 and one null value in question 1, dropping those rows.

#%%
# Count the number of rows before dropping
rows_before_drop = len(data)

# Drop rows with missing values
data = data.dropna()

# Count the number of rows after dropping
rows_after_drop = len(data)

# Calculate the number of rows dropped
rows_dropped = rows_before_drop - rows_after_drop

# Display the number of rows dropped
print("Number of rows dropped:", rows_dropped)

#%%[markdown]
### Distribution of data points among output classes (Similar and Non Similar Questions)
# Check for Balance of Data (Ouput Column: is_duplicate)

#%%
# Group by 'is_duplicate' and count the number of observations for each group
grouped_data = data.groupby("is_duplicate")['id'].count()

total_questions = grouped_data.sum()
percentages = (grouped_data / total_questions) * 100

colors = ['lightblue', 'lightcoral']  
plt.figure(figsize=(10, 8))  
ax = percentages.plot(kind='bar', color=colors, edgecolor='black')

plt.title('Distribution of Duplicate and Non-duplicate Questions', fontsize=16)
plt.xlabel('is_duplicate', fontsize=14)
plt.ylabel('Percentage of Questions', fontsize=14)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)

ax.set_xticklabels(['Non-duplicate', 'Duplicate'], rotation=0)

plt.show()

#%%[markdown]

# Number of Unique Questions, Number of Unique Question appearing more than one time and number of times a single question is repeated.

#%%
qids = pd.Series(data['qid1'].tolist() + data['qid2'].tolist())
unique_qs = len(np.unique(qids))
qs_morethan_onetime = np.sum(qids.value_counts() > 1)

print('Total number of Unique Questions: {}\n'.format(unique_qs))
print('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime, round(qs_morethan_onetime/unique_qs*100,2)))
print('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts())))

# %%
# Plot the number of unique questions and repeated questions
plt.figure(figsize=(8, 6))
colors = ['yellow', 'lightgreen']
plt.bar(['Unique Questions', 'Repeated Questions'], [unique_qs, qs_morethan_onetime], color=colors, edgecolor='black')

plt.title('Number of Unique and Repeated Questions', fontsize=16)
plt.ylabel('Number of Questions', fontsize=14)

# Add text annotations
for i, count in enumerate([unique_qs, qs_morethan_onetime]):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=12)

plt.show()

# %%[markdown]

# Checking for Duplicates

#%%
# Check for rows where qid1 and qid2 are the same
same_qid_rows = data[data['qid1'] == data['qid2']]

# Check for rows where qid1 and qid2 are interchanged
interchanged_qid_rows = data[data.apply(lambda row: row['qid1'] == row['qid2'] or (row['qid1'] == row['qid2'] and row['qid1'] is not None and row['qid2'] is not None), axis=1)]

# Display the results
print("Rows where qid1 and qid2 are the same:")
print(same_qid_rows)

print("\nRows where qid1 and qid2 are interchanged:")
print(interchanged_qid_rows)

# Count the total number of duplicate pairs
total_duplicates = len(same_qid_rows) + len(interchanged_qid_rows)
print("Total number of duplicate pairs:", total_duplicates)

# %%[markdown]

# Number of Occurances of each question.

#%%
plt.figure(figsize=(20, 10))

counts, bins, _ = plt.hist(qids.value_counts(), bins=160, color='skyblue', edgecolor='black')

plt.yscale('log', nonpositive='clip')

plt.title('Log-Histogram of question appearance counts', fontsize=16)
plt.xlabel('Number of occurrences of question', fontsize=14)
plt.ylabel('Number of questions', fontsize=14)

max_occurrence = max(qids.value_counts())
plt.axvline(x=max_occurrence, color='red', linestyle='--', label=f'Max Occurrence: {max_occurrence}')

plt.legend()

plt.show()

# %%[markdown]

# The plot is close to a power-law distribution not exactly power-law but close to it.

#%%[markdown]

### Top 10 Most asked questions on Quora:

#%%
all_questions = pd.concat([data['question1'], data['question2']], ignore_index=True)

# Display the top 10 most common questions
top_10_common_questions = all_questions.value_counts().head(10)

print("Top 10 Most Common Questions:")
print(top_10_common_questions)

#%%[markdown]

### Distribution of Question Lengths:

#%%
# Function to count the number of words in a sentence
def count_words(sentence):
    # Handle the case where the sentence is NaN (missing value)
    if pd.isnull(sentence):
        return 0
    # Count the number of words by splitting the sentence
    return len(str(sentence).split())

# Plot histograms for question lengths
plt.figure(figsize=(12, 6))
plt.hist(data['question1'].apply(lambda x: count_words(x)), bins=50, alpha=0.5, label='Question 1', color='blue')
plt.hist(data['question2'].apply(lambda x: count_words(x)), bins=50, alpha=0.5, label='Question 2', color='orange')

# Title and labels
plt.title('Distribution of Question Lengths', fontsize=16)
plt.xlabel('Number of words', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Display legend
plt.legend()

plt.show()

#%%[markdown]

# References for feature extraction:
# - Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
# - Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
# - Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

#%%[markdown]

### Feature Extraction 

# - ____freq_qid1____ = Frequency of qid1's
# - ____freq_qid2____ = Frequency of qid2's 
# - ____q1len____ = Length of q1
# - ____q2len____ = Length of q2
# - ____q1_n_words____ = Number of words in Question 1
# - ____q2_n_words____ = Number of words in Question 2
# - ____word_Common____ = (Number of common unique words in Question 1 and Question 2)
# - ____word_Total____ =(Total num of words in Question 1 + Total num of words in Question 2)
# - ____word_share____ = (word_common)/(word_Total)
# - ____freq_q1+freq_q2____ = sum total of frequency of qid1 and qid2 
# - ____freq_q1-freq_q2____ = absolute difference of frequency of qid1 and qid2 

#%%

# %%
file_path = "df_fe_without_preprocessing_train.csv"
data = process_data(file_path)

#%%
data.head(5)

# %%[markdown]

### Check for questions with 2 words or less than 2 words

#%%
# Filter sentences with 2 words or less in either q1 or q2
filtered_data = data[(data['q1_n_words'] <= 2) | (data['q2_n_words'] <= 2)]

# Print the filtered sentences along with is_duplicate column and the number of sentences
num_sentences = len(filtered_data)
print(f"Number of Sentences: {num_sentences}\n")

for index, row in filtered_data.head(10).iterrows():
    print(f"Q1: {row['question1']}")
    print(f"Q2: {row['question2']}")
    print(f"Is Duplicate: {row['is_duplicate']}")
    print("-" * 50)

# %%
print ("Minimum length of the questions in question1 : " , min(data['q1_n_words']))

print ("Minimum length of the questions in question2 : " , min(data['q2_n_words']))

print ("Number of Questions with minimum length [question1] :", data[data['q1_n_words']== 1].shape[0])
print ("Number of Questions with minimum length [question2] :", data[data['q2_n_words']== 1].shape[0])

#%%[markdown]

# Univariate Analysis : 'freq_qid1', 'freq_qid2', 'q1len', 'q2len', 'q1_n_words', 'q2_n_words','word_Common', 'word_Total', 'word_share', 'freq_q1+q2', 'freq_q1-q2'.

#%%
# List of columns to plot
columns_to_plot = ['freq_qid1', 'freq_qid2', 'q1len', 'q2len', 'q1_n_words', 'q2_n_words',
                   'word_Common', 'word_Total', 'word_share', 'freq_q1+q2', 'freq_q1-q2']

# Loop through each column and create plots
for column in columns_to_plot:
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    sns.violinplot(x='is_duplicate', y=column, data=data)

    plt.subplot(1, 2, 2)
    sns.histplot(data[data['is_duplicate'] == 1.0][column], label="1", color='red', kde=True)
    sns.histplot(data[data['is_duplicate'] == 0.0][column], label="0", color='blue', kde=True)
    
    # Add legend and labels
    plt.legend()
    
    # Set the title at the center for the entire figure
    plt.suptitle(f'Distribution of {column} for Duplicate and Non-duplicate Questions', fontsize=16, ha='center')
    
    plt.xlabel(column)
    plt.ylabel('Density')
    
    plt.show()

#%%[markdown]

### Important features in differentiating Duplicate(Similar) and Non-Duplicate(Dissimilar) Questions.

# 1. Distribution of q1len for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 2. Distribution of q2len for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 3. Distribution of q1_n_words for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 4. Distribution of q2_n_words for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 5. Distribution of word_Total for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 6. Distribution of word_share for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.

#%%[markdown]
    
### Pre-processing of Text 
    
# - Preprocessing:
# 1. Removing html tags 
# 2. Removing Punctuations
# 3. Performing stemming
# 4. Removing Stopwords
# 5. Expanding contractions etc.

# %%
