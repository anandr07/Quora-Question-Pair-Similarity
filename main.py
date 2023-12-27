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

### Importing Needed Libraries

#%%

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

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
def count_words(sentence):
    # Handle the case where the sentence is NaN (missing value)
    if pd.isnull(sentence):
        return 0
    # Count the number of words by splitting the sentence
    return len(str(sentence).split())

# Apply the count_words function to 'question1' and 'question2' for each row
data['q1_len'] = data['question1'].apply(lambda x: count_words(x))
data['q2_len'] = data['question2'].apply(lambda x: count_words(x))

# Plot histograms for question lengths
plt.figure(figsize=(12, 6))
plt.hist(data['q1_len'], bins=50, alpha=0.5, label='Question 1', color='blue')
plt.hist(data['q2_len'], bins=50, alpha=0.5, label='Question 2', color='orange')

# Title and labels
plt.title('Distribution of Question Lengths', fontsize=16)
plt.xlabel('Number of words', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Display legend
plt.legend()

plt.show()

#%%