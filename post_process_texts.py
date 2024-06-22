"""
WORKS IN-PLACE
"""

import re
from utils import PATTERN, DESC_ROOT
import os
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ''
    
    text = text.lower()
    #punctuation_special_chars_pattern = r'[^\w\s]|_'
    #single_letter_words_pattern = r'\b[a-zA-Z]\b'
    #numbers_pattern = r'\b\d+\b'  
    #text = re.sub(punctuation_special_chars_pattern, ' ', text)
    #text = re.sub(numbers_pattern, ' ', text)  
    tokens = text.split()
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    clean_text_final = ' '.join(cleaned_tokens)
    return clean_text_final



def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


def main():
    description_csvs = [os.path.join(DESC_ROOT, f) for f in os.listdir(DESC_ROOT) if f.endswith("csv")]

    np.random.seed(13)
    for csv in description_csvs:
        df = pd.read_csv(csv)
        figures = [PATTERN.search(cap).group(0) for cap in df['Description']]
        descs = [cap.replace(fg, "") for fg, cap in zip(figures, df['Description'])]

        clean_sentences = [re.sub(r'\s+', ' ', sentence.strip()) for sentence in descs]
        # TODO remove all punctuations
        df['Description'] = clean_sentences

        # for idx, sentence in enumerate(clean_sentences, 1):
        #     print(f"{idx}. {sentence}")
        # time.sleep(5)

        refs = [eval(r) for r in df['Reference']]
        refs = [map(lambda x:re.sub(r'\s+', ' ', x.strip()), sent_list) for sent_list in refs]
        df['Reference'] = [list(map(lambda x:x.replace(PATTERN.search(x).group(0), "") ,group)) for group in refs]
        # TODO remove all punctuations
        
        df['Description'] = df['Description'].replace('', np.nan)
        df['Cleaned_Description'] = df['Description'].apply(clean_text)
        df['Lemmatized_Description'] = df['Cleaned_Description'].apply(lemmatize_text)
        df['Stemmed_Description'] = df['Cleaned_Description'].apply(stem_text)
        df = df.dropna(subset=['Description'])

        ## Adjust the splits. IT IS COMPLETE RANDOM SPLIT WE MAY WANT TO PARAMETERIZE HERE TO HAVE BOOK BY BOOK SPLIT
        df['split'] = [
                'valid' if (rand_num := np.random.random()) < 0.1 else
                'test' if rand_num < 0.2 else
                'train'
                for _ in range(len(df))]
        df.to_csv(csv, index=False)
        print(f"Done [{csv}]...")

        """
        Figure calls  are removed so far in this code.
        Remove all reference words such as see, refer to, etc.
        """