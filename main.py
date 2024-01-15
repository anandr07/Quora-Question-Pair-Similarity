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
# from feature_extraction import process_data, extract_features
from feature_extraction import process_and_extract_features
from ml_algorithms.tSNE_for_data_visualization import plot_tsne_visualization
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



#%%
# Importing other files from ml_algorithms folder:

import sys
sys.path.append(os.getcwd()+'\ml_algorithms')

#%%
from Logistic_Regression import logistic_regression_function
from SGDClassfier_RandomSearch_V1 import sgd_random_search_v1
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

#### Top 10 Most asked questions on Quora:

#%%
all_questions = pd.concat([data['question1'], data['question2']], ignore_index=True)

# Display the top 10 most common questions
top_10_common_questions = all_questions.value_counts().head(10)

print("Top 10 Most Common Questions:")
print(top_10_common_questions)

#%%[markdown]

#### Distribution of Question Lengths:

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

#%%[markdown] 
### Pre-processing of Text 
    
#%%[markdown]  
# - Preprocessing:
# 1. Removing html tags 
# 2. Removing Punctuations
# 3. Performing stemming
# 4. Removing Stopwords
# 5. Expanding contractions etc.

#%%[markdown] 
#### Feature Extraction after pre-processing.

# Featurization (NLP and Fuzzy Features)

# Definition:
# - __Token__: You get a token by splitting sentence a space
# - __Stop_Word__ : stop words as per NLTK.
# - __Word__ : A token that is not a stop_word
# <br>
# <br>    
# __Features__:
# - __cwc_min__ :  Ratio of common_word_count to min lenghth of word count of Q1 and Q2 <br>cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
# <br>
# <br>
# - __cwc_max__ :  Ratio of common_word_count to max lenghth of word count of Q1 and Q2 <br>cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
# <br>
# <br>
# - __csc_min__ :  Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 <br> csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
# <br>
# <br>
# - __csc_max__ :  Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2<br>csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
# <br>
# <br>
# - __ctc_min__ :  Ratio of common_token_count to min lenghth of token count of Q1 and Q2<br>ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
# <br>
# <br>
# - __ctc_max__ :  Ratio of common_token_count to max lenghth of token count of Q1 and Q2<br>ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
# <br>
# <br>
# - __last_word_eq__ :  Check if First word of both questions is equal or not<br>last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
# <br>
# <br>
# - __first_word_eq__ :  Check if First word of both questions is equal or not<br>first_word_eq = int(q1_tokens[0] == q2_tokens[0])
# <br>
# <br>       
# - __abs_len_diff__ :  Abs. length difference<br>abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
# <br>
# <br>
# - __mean_len__ :  Average Token Length of both Questions<br>mean_len = (len(q1_tokens) + len(q2_tokens))/2
# <br>
# <br>

# - __fuzz_ratio__ :  https://github.com/seatgeek/fuzzywuzzy#usage
# http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# <br>
# <br>

# - __fuzz_partial_ratio__ :  https://github.com/seatgeek/fuzzywuzzy#usage
# http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# <br>
# <br>

# - __token_sort_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage
# http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# <br>
# <br>

# - __token_set_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage
# http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# <br>
# <br>

# - __longest_substr_ratio__ :  Ratio of length longest common substring to min lenghth of token count of Q1 and Q2<br>longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens)) <br>

### Some additional features - Adding some more features which I feel will add good information. <br>

# ratio_q_lengths:  This feature calculates the ratio of the lengths of the two questions. <br>

# common_prefix: This feature computes the length of the common prefix (the initial common sequence of characters) between the two questions. <br>

# common_suffix: This feature calculates the length of the common suffix (the final common sequence of characters) between the two questions. <br>

# diff_words: This feature calculates the absolute difference in the number of words between the two questions. <br>

# diff_chars: This feature computes the absolute difference in the number of characters between the two questions. <br>

# jaccard_similarity: This feature calculates the Jaccard similarity coefficient between the sets of words in the two questions. <br>

# longest_common_subsequence: This feature computes the length of the longest common subsequence (LCS) between the two questions. <br>

#%%[markdown]

### Processing and Extracting Features
# %%
# Processing and Extracting Features
file_path = "data_with_features.csv"

# *****************************************************Observations_to_Train*************************************************
rows_to_train = 200000 # Change as per Needs
print(f"TRAINING WITH {rows_to_train} OBSERVATIONS")
# ***************************************************************************************************************************

if os.path.isfile(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data.fillna('', inplace=True)  # Fill NaN values with empty string if needed
else:
    data = process_and_extract_features(file_path,rows_to_train)
#%%
data.head(5)

# %%[markdown]

#### Check for questions with 2 words or less than 2 words

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

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Function to calculate KL Divergence
def calculate_kl_divergence(duplicate_data, non_duplicate_data, feature):
    duplicate_dist = duplicate_data[feature].dropna()
    non_duplicate_dist = non_duplicate_data[feature].dropna()

    epsilon = 1e-10
    duplicate_dist += epsilon
    non_duplicate_dist += epsilon

    min_length = min(len(duplicate_dist), len(non_duplicate_dist))
    duplicate_dist = duplicate_dist.head(min_length)
    non_duplicate_dist = non_duplicate_dist.head(min_length)

    kl_divergence = entropy(duplicate_dist, non_duplicate_dist)
    return kl_divergence

features_to_plot = ['freq_qid1', 'freq_qid2', 'q1len', 'q2len', 'q1_n_words', 'q2_n_words',
                    'word_Common', 'word_Total', 'word_share', 'freq_q1+q2', 'freq_q1-q2',
                    'ratio_q_lengths', 'common_prefix', 'common_suffix', 'diff_words', 'diff_chars',
                    'jaccard_similarity', 'longest_common_subsequence', 'cwc_min', 'cwc_max', 'csc_min',
                    'csc_max', 'ctc_min', 'ctc_max', 'last_word_eq', 'first_word_eq', 'abs_len_diff',
                    'mean_len', 'token_set_ratio', 'token_sort_ratio', 'fuzz_ratio', 'fuzz_partial_ratio',
                    'longest_substr_ratio']

#%%
# Create one image with Violin plots and Density plots for each feature
num_features = len(features_to_plot)
plt.figure(figsize=(16, 2*num_features))

for i, feature in enumerate(features_to_plot):
    plt.subplot(num_features, 2, 2*i + 1)
    sns.violinplot(x='is_duplicate', y=feature, data=data)
    plt.title(f'Violin Plot for {feature}')

    plt.subplot(num_features, 2, 2*i + 2)
    sns.kdeplot(data[data['is_duplicate'] == 1][feature], label='Duplicate', shade=True)
    sns.kdeplot(data[data['is_duplicate'] == 0][feature], label='Not Duplicate', shade=True)
    plt.title(f'Density Plot for {feature}')

plt.tight_layout()
plt.show()

#%%
# Calculate and visualize inverted KL Divergence
kl_divergence_results = pd.DataFrame(columns=['Feature', 'KL_Divergence'])

for feature in features_to_plot:
    kl_divergence = calculate_kl_divergence(data[data['is_duplicate'] == 1], data[data['is_duplicate'] == 0], feature)
    kl_divergence_results = pd.concat([kl_divergence_results, pd.DataFrame({
        'Feature': [feature],
        'KL_Divergence': [kl_divergence]
    })], ignore_index=True)

# Display KL Divergence results in a table
print(kl_divergence_results)

# Create a bar plot to visualize inverted KL Divergence
kl_divergence_results['Inverted_KL_Divergence'] = 1 / (kl_divergence_results['KL_Divergence'] + 1e-10)

plt.figure(figsize=(15, 10))
sns.barplot(x='Feature', y='Inverted_KL_Divergence', data=kl_divergence_results.sort_values(by='Inverted_KL_Divergence', ascending=False))
plt.title('Inverted KL Divergence for Each Feature')
plt.xticks(rotation=45, ha='right')
plt.show()

#%%
bottom_5_features = kl_divergence_results.nsmallest(5, 'KL_Divergence')['Feature']

print("The best 5 features are:")
print(bottom_5_features)

# Pair plot for the top 10 features
n = data.shape[0]
sns.pairplot(data[bottom_5_features.tolist() + ['is_duplicate']][0:n], hue='is_duplicate', vars=bottom_5_features.tolist())
plt.show()

#%%[markdown]
#### Important features in differentiating Duplicate(Similar) and Non-Duplicate(Dissimilar) Questions.

# 1. Distribution of q1len for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 2. Distribution of q2len for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 3. Distribution of q1_n_words for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 4. Distribution of q2_n_words for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 5. Distribution of word_Total for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.
# 6. Distribution of word_share for Duplicate and Non-duplicate Questions overlap but not completely making it a good feature.

#%%[markdown]
### Visualizing in lower dimension using t-SNE

plot_tsne_visualization(data)

#%%[markdown]
### Featurizing text data with Tf-Idf weighted word-vectors

# %%
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
import sys
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy
from spacy.cli import download

# Load the dataset
df = pd.read_csv('data/train.csv')

# change this for final training of model
# ***************************************Observations_to_Train***************************************************************
df = df[:rows_to_train]
# ***************************************************************************************************************************
# Encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: str(x))
df['question2'] = df['question2'].apply(lambda x: str(x))

# Combine texts
questions = list(df['question1']) + list(df['question2'])

# TF-IDF vectorization
tfidf = TfidfVectorizer(lowercase=False)
tfidf.fit_transform(questions)
word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

def load_spacy_model(model_name='en_core_web_lg'):
    try:
        # Check if the spaCy model is already installed
        nlp = spacy.load(model_name)
        print(f"spaCy model '{model_name}' loaded successfully.")
    except OSError:
        print(f"Model '{model_name}' not found. Downloading...")
        # Download the spaCy model if not already installed
        download(model_name)
        nlp = spacy.load(model_name)
        print(f"spaCy model '{model_name}' downloaded and loaded successfully.")

    return nlp

# Load spaCy model
nlp = load_spacy_model()

# Extract features using spaCy
vecs1 = []
vecs2 = []

for qu1, qu2 in tqdm(zip(list(df['question1']), list(df['question2']))):
    doc1 = nlp(qu1)
    mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])

    for word1 in doc1:
        vec1 = word1.vector
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        mean_vec1 += vec1 * idf

    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)

    doc2 = nlp(qu2)
    mean_vec2 = np.zeros([len(doc2), len(doc2[0].vector)])

    for word2 in doc2:
        vec2 = word2.vector
        try:
            idf = word2tfidf[str(word2)]
        except:
            idf = 0
        mean_vec2 += vec2 * idf

    mean_vec2 = mean_vec2.mean(axis=0)
    vecs2.append(mean_vec2)

df['q1_feats_m'] = list(vecs1)
df['q2_feats_m'] = list(vecs2)

#%%
# Loading Processing and Extracting Features data
if os.path.isfile('data_with_features.csv'):
    dfnlp = pd.read_csv("data_with_features.csv", encoding='latin-1', nrows = rows_to_train)
else:
    print("Run the Processing and Extracting Features Cell")

# Drop unnecessary columns
df1 = dfnlp.drop(['qid1', 'qid2', 'question1', 'question2'], axis=1)
df3 = df.drop(['qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], axis=1)
df3_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index=df3.index)
df3_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index=df3.index)

# Display information about features
print("Number of features in nlp dataframe:", df1.shape[1])
print("Head(5) of nlp dataframe:")
print(df1.head(5))

print("\nNumber of features in question1 w2v dataframe:", df3_q1.shape[1])
print("Head(5) of question1 w2v dataframe:")
print(df3_q1.head(5))

print("\nNumber of features in question2 w2v dataframe:", df3_q2.shape[1])
print("Head(5) of question2 w2v dataframe:")
print(df3_q2.head(5))

print("\nNumber of features in the final dataframe:", df1.shape[1] + df3_q1.shape[1] + df3_q2.shape[1])

#%%
# Check if the final_features.csv file already exists
if not os.path.isfile('final_features.csv'):
    # Add id column to df3_q1 and df3_q2
    df3_q1['id'] = df1['id']
    df3_q2['id'] = df1['id']
    
    # Merge df1 with df3_q1 and df3_q2 on 'id'
    result = df1.merge(df3_q1, on='id', how='left').merge(df3_q2, on='id', how='left')
    
    # Save the result to final_features.csv
    result.to_csv('final_features.csv', index=False)

#%%
# Read the CSV file
df = pd.read_csv('final_features.csv')

# %%
# Replace non-numeric values with NaN
df.replace({col: {'_x': np.nan} for col in data.columns}, inplace=True)

# Check if there are any NA values in the DataFrame
if df.isna().any().any():
    print("NA Values Present")
else:
    print("No NA Values Present")

# Check the number of NaN values in each column after replacement
nan_counts = df.isna().sum()
print("Number of NaN values in each column after replacement:")
print(nan_counts)

#%%
# Remove the first row 
# df.drop(data.index[0], inplace=True)

# Get the target variable
y_true = df['is_duplicate']
df.drop(['id', 'is_duplicate'], axis=1, inplace=True)

print(df.shape)

# Convert all the features into numeric
cols = list(df.columns)
for i in cols:
    df[i] = pd.to_numeric(df[i], errors='coerce')

#%%
# Check for NA Values
if df.isna().any().any():
    print("NA Values Present")
else:
    print("No NA Values Present")
# Check the number of NaN values in each column after conversion
nan_counts_after_conversion = df.isna().sum()
print("Number of NaN values in each column after conversion:")
print(nan_counts_after_conversion)

# Convert y_true to a list of integers
y_true = list(map(int, y_true.values))

# Display the first few rows of the data
df.head()

#%%[markdown]
### Splitting into Train and Test Data

# %%
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(df, y_true, stratify=y_true, test_size=0.3)

# Convert lists to DataFrames if they are not already
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# Create a DataFrame to display the sizes of the splits
split_sizes = pd.DataFrame({
    'Data Split': ['X_train', 'X_test', 'y_train', 'y_test'],
    'Size': [X_train.shape[0], X_test.shape[0], y_train.shape[0], y_test.shape[0]]
})

# Display the split sizes in tabular format
print("Size of Data Splits:")
print(split_sizes)

# Print head(5) for each split
print("Head of X_train:")
print(X_train.head())

print("\nHead of X_test:")
print(X_test.head())

print("\nHead of y_train:")
print(y_train.head())

print("\nHead of y_test:")
print(y_test.head())

# %%[markdown]
### Distribution of Output Variable in Train and Test Data

#%%
# Plotting the distribution of the output variable in train data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train.iloc[:, 0])
plt.title('Distribution of Output Variable in Train Data')

# Plotting the distribution of the output variable in test data
plt.subplot(1, 2, 2)
sns.countplot(x=y_test.iloc[:, 0])
plt.title('Distribution of Output Variable in Test Data')

plt.show()

#%%[markdown]
### Using a Random Model to Predict and Noting Performance 

# Our Models should perform better than the Random Model

#%%
from sklearn.metrics import log_loss

# Generate random predictions for y_train
np.random.seed(42)  # Set seed for reproducibility
random_predictions_train = np.random.rand(len(y_train))

# Ensure the predictions sum up to 1 for each sample
random_predictions_train /= random_predictions_train.sum(keepdims=True)

# Calculate log loss for y_train
log_loss_train = log_loss(y_train, random_predictions_train)

# Display the log loss for the training data
print(f'Log Loss for Training Data: {log_loss_train:.5f}')

# Generate random predictions for y_test
random_predictions_test = np.random.rand(len(y_test))
random_predictions_test /= random_predictions_test.sum(keepdims=True)

# Calculate log loss for y_test
log_loss_test = log_loss(y_test, random_predictions_test)

# Display the log loss for the test data
print(f'Log Loss for Test Data: {log_loss_test:.5f}')

#%%[markdown]

### Logistic Regression ----> Baseline Model

#%%
logistic_regression_function(X_train, X_test, y_train, y_test)

#%%[markdown]

### SGDClassifier for predict </br>

# Performing hyperparameter tuning using RandomizedSearchCV for an SGDClassifier with elastic net penalty, followed by training, evaluation, and visualization of the best model. </br> 
# The evaluation includes log loss calculation and display of the confusion matrix.

#%%
sgd_random_search_v1(X_train, X_test, y_train, y_test)

#%%[markdown]
