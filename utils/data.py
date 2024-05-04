import os
import pandas as pd

label_map = {'happy': 1, 'sad': 0}

# Download dataset from: https://github.com/mohummedalee/twitteraae-sentiment-data/
def load_twitter_aae(dir):
    sentences = []
    labels = []
    dialects = []
    for dial in ['aae', 'sae']:
        for lab in ['happy', 'sad']:
            # load dialect x sentiment combination
            fpath = os.path.join(dir, f'{dial}_{lab}')
            df = pd.read_csv(fpath, on_bad_lines='skip', encoding_errors='ignore', names=['text'])
            n = df.shape[0]  # number of correctly loaded sentences
            sentences.extend(df['text'].tolist())
            labels.extend([label_map[lab]] * n)
            dialects.extend([dial.upper()] * n)            

    return sentences, labels, dialects

if __name__ == '__main__':
    # test loading one file
    DATA_DIR = '../data/raw/sentiment_race'

    sentences, labels, dialects = load_twitter_aae(DATA_DIR)
    print(len(sentences), len(labels), len(dialects))