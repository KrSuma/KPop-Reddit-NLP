import math
from IPython import display
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sns.set(style='darkgrid', context='talk', palette='Dark2')

reddit = praw.Reddit(client_id='AA7pptPLpjM-XA',
                     client_secret='YB6hIjE9no2YZ1h6eTyJqDCvvjPhAg',
                     user_agent='DaebakNLP')

head = set()
for subs in reddit.subreddit('kpop').new(limit=None):
    head.add(subs.title)
    display.clear_output()
    print(len(head))

sia = SIA()
results = []

for line in head:
    score = sia.polarity_scores(line)
    score['headline'] = line
    results.append(score)

pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
print(df.head())

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
print(df.head())

df2 = df[['headline', 'label']]
df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].headline)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].headline)[:5], width=200)

print(df.label.value_counts())
print(df.label.value_counts(normalize=True) * 100)