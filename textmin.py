import time
start_time = time.time()

# Problem Statement
# The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech 
# if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

# Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' 
# denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.


import pandas as pd
from pprint import pprint
import numpy as np
from nltk.corpus import stopwords

f=open("out.txt","w")

### Data importing.
train = pd.read_csv("train_E6oV3lV.csv")

##

## Word count in each tweet.
train["Word_count"] = train["tweet"].apply(lambda x: len(str(x).split(" ")))
# print(train.head())

## Number of characters. len of each tweet
train["len_tweet"] = train["tweet"].str.len()
# print(train.head())

## Average word length.
# print("Average word length is in data set",np.mean(train["len_tweet"]))

## Average word length of each tweet is. number of characters/number of word.
train["Avg_word_len"] = train["len_tweet"]/train["Word_count"]
# pprint(train.head(),stream=f)

## Number of Stop words. Stopwords are the commonly used words that a search engine has been programmed to ignore.
stop = stopwords.words("english")
train["len_stop"] = train["tweet"].apply(lambda x: len([x for x in x.split() if x in stop]))

## Number of special characters. (# -- tag)
train["hastags"] = train["tweet"].apply(lambda x: len([x for x in x.split() if x.startswith("#")]))

## number of numerics.
train["number_numerics"] = train["tweet"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

## Number of upper case words.
train["is_upper"] = train["tweet"].apply(lambda x: len([x for x in x.split() if x.isupper()]))

##################################################
## Basic Preprocessing.
##################################################
# Till now we learned how to extract basic feature from text data.
# Before diving into text and feature extraction. our first step should be cleaning the data, and 
# Converting everything into lowercases and removing white spaces.
train["tweet"] = train["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
# pprint(train.head(),stream=f)


# removing punctuation
train["tweet"] = train["tweet"].str.replace("[^\w\s]","")

# Removal of stopwords, -- are the commonly used words suggested by nltk.
train["tweet"] = train["tweet"].apply(lambda x: " ".join(x for x in  x.split() if x not in stop))
pprint(train.head(),stream=f)

## Removal of common words from data set.
frq = pd.Series(" ".join(train["tweet"]).split()).value_counts()[:10]
df_common_list = list(frq.index)
train["tweet"] = train["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in df_common_list))
pprint(train.head(),stream=f)

# removing rarely used words from the dataset, since they are very rare.
# freq = pd.Series(" ".join(train["tweet"]).split()).value_counts()[-20:-1]
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
print(freq)
train["tweet"] = train["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in list(freq.index)))
print("\n",pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:])



### Spelling Correction
# Spelling correction is an important problem in text mining.
# To achieve this we will use text blob library.

# from textblob import TextBlob

# print(train["tweet"][:5].apply(lambda x: str(TextBlob(x).correct())))
# before doing spelling correction, keep in mind that, most people use words in abbreviated form. So when we apply textblob.correct(),
# there is a chance that it may auto correct to false meaning.


# Tokenization.






print("\n")
# To get to know the execution time.
print(time.time() - start_time)