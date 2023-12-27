#%%
import os 
import pandas as pd 

def process_data(file_path):
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, encoding='latin-1')
    else:
        data = pd.read_csv(file_path)
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

        data.to_csv(file_path, index=False)

    return data