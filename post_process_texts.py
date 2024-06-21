"""
WORKS IN-PLACE
"""

import re
from utils import PATTERN, DESC_ROOT
import os
import pandas as pd
import numpy as np
def main():
    description_csvs = [os.path.join(DESC_ROOT, f) for f in os.listdir(DESC_ROOT) if f.endswith("csv")]

    np.random.seed(13)
    for csv in description_csvs:
        df = pd.read_csv(csv)
        figures = [PATTERN.search(cap).group(0) for cap in df['Description']]
        descs = [cap.replace(fg, "") for fg, cap in zip(figures, df['Description'])]

        clean_sentences = [re.sub(r'\s+', ' ', sentence.strip()) for sentence in descs]
        df['Description'] = clean_sentences

        # for idx, sentence in enumerate(clean_sentences, 1):
        #     print(f"{idx}. {sentence}")
        # time.sleep(5)

        refs = [eval(r) for r in df['Reference']]
        refs = [map(lambda x:re.sub(r'\s+', ' ', x.strip()), sent_list) for sent_list in refs]
        df['Reference'] = [list(map(lambda x:x.replace(PATTERN.search(x).group(0), "") ,group)) for group in refs]
        
        df['Description'] = df['Description'].replace('', np.nan)
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