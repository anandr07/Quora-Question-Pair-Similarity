#%%
import os
import pandas as pd
from nltk.corpus import stopwords
from pre_processing import preprocess
from fuzzywuzzy import fuzz
import distance
from nltk.metrics import jaccard_distance

SAFE_DIV = 0.0001
STOP_WORDS = stopwords.words("english")

def get_token_features(q1, q2):
    token_features = [0.0]*10
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

def process_and_extract_features(file_path,rows_to_train):
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, encoding='latin-1')
    else:
        data = pd.read_csv("data/train.csv")
        data = data[:rows_to_train]
        data.dropna(subset=['question1', 'question2'], inplace=True)
        data['freq_qid1'] = data.groupby('qid1')['qid1'].transform('count') 
        data['freq_qid2'] = data.groupby('qid2')['qid2'].transform('count')
        data['q1len'] = data['question1'].str.len() 
        data['q2len'] = data['question2'].str.len()
        data['q1_n_words'] = data['question1'].apply(lambda row: len(row.split(" ")))
        data['q2_n_words'] = data['question2'].apply(lambda row: len(row.split(" ")))

        def normalized_word_Common(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * len(w1 & w2)
        data['word_Common'] = data.apply(normalized_word_Common, axis=1)

        def normalized_word_Total(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * (len(w1) + len(w2))
        data['word_Total'] = data.apply(normalized_word_Total, axis=1)

        def normalized_word_share(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
        data['word_share'] = data.apply(normalized_word_share, axis=1)

        data['freq_q1+q2'] = data['freq_qid1']+data['freq_qid2']
        data['freq_q1-q2'] = abs(data['freq_qid1']-data['freq_qid2'])

        # preprocessing each question
        data["question1"] = data["question1"].fillna("").apply(preprocess)
        data["question2"] = data["question2"].fillna("").apply(preprocess)

        print("token features...")

        # Merging Features with dataset
        token_features = data.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)

        data["cwc_min"]       = list(map(lambda x: x[0], token_features))
        data["cwc_max"]       = list(map(lambda x: x[1], token_features))
        data["csc_min"]       = list(map(lambda x: x[2], token_features))
        data["csc_max"]       = list(map(lambda x: x[3], token_features))
        data["ctc_min"]       = list(map(lambda x: x[4], token_features))
        data["ctc_max"]       = list(map(lambda x: x[5], token_features))
        data["last_word_eq"]  = list(map(lambda x: x[6], token_features))
        data["first_word_eq"] = list(map(lambda x: x[7], token_features))
        data["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
        data["mean_len"]      = list(map(lambda x: x[9], token_features))

        print("fuzzy features..")

        data["token_set_ratio"]       = data.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
        data["token_sort_ratio"]      = data.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
        data["fuzz_ratio"]            = data.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
        data["fuzz_partial_ratio"]    = data.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
        data["longest_substr_ratio"]  = data.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)

        print("Adding Additional features.....")

        data['ratio_q_lengths'] = data.apply(lambda row: ratio_of_question_lengths(row['question1'], row['question2']), axis=1)
        data['common_prefix'] = data.apply(lambda row: common_prefix(row['question1'], row['question2']), axis=1)
        data['common_suffix'] = data.apply(lambda row: common_suffix(row['question1'], row['question2']), axis=1)
        data['diff_words'] = data.apply(lambda row: abs(row['q1_n_words'] - row['q2_n_words']), axis=1)
        data['diff_chars'] = data.apply(lambda row: abs(len(str(row['question1'])) - len(str(row['question2']))), axis=1)
        data['jaccard_similarity'] = data.apply(lambda row: jaccard_similarity(row['question1'], row['question2']), axis=1)
        data['longest_common_subsequence'] = data.apply(lambda row: longest_common_subsequence(row['question1'], row['question2']), axis=1)

        data.to_csv(file_path, index=False)

    return data

def ratio_of_question_lengths(q1, q2):
    # Function to calculate the ratio of question lengths
    len_q1 = len(str(q1))
    len_q2 = len(str(q2))
    
    if len_q2 == 0:
        return 0.0
    
    return len_q1 / len_q2

def common_prefix(q1, q2):
    # Function to find the length of the common prefix
    i = 0
    while i < min(len(q1), len(q2)) and q1[i] == q2[i]:
        i += 1
    return i

def common_suffix(q1, q2):
    # Function to find the length of the common suffix
    i, j = len(q1) - 1, len(q2) - 1
    while i >= 0 and j >= 0 and q1[i] == q2[j]:
        i -= 1
        j -= 1
    return len(q1) - i - 1

def jaccard_similarity(q1, q2):
    # Function to calculate Jaccard's Similarity
    q1_tokens = set(q1.split())
    q2_tokens = set(q2.split())
    
    if not q1_tokens or not q2_tokens:
        return 0.0
    
    return 1.0 - jaccard_distance(q1_tokens, q2_tokens)

def longest_common_subsequence(q1, q2):
    # Function to calculate Longest Common Subsequence
    seq1 = list(q1)
    seq2 = list(q2)
    
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

#%%