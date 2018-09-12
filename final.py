import pandas as pd
import numpy as np
from textblob import TextBlob

# Train the data with tfidfvectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# ML Package
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# metric
from sklearn.metrics import accuracy_score,classification_report

# INPUT

train = pd.read_csv("train_E6oV3lV.csv")
test = pd.read_csv("test_tweets_anuFYb8.csv")
from nltk.corpus import stopwords
from textblob import Word

# Cleaning -- Removing @+ and # tags.
def cleaning_extra_chara(data):
    data["tweet"] = data["tweet"].str.replace("(@[A-Za-z0-9]+)|#","")
    return data

# Removing Stop words.
def text_blob_stop_words(data,stop):
    data["tweet"] = data["tweet"].apply(lambda x: " ".join([x for x in x.split() if x not in stop]))
    return data

# Lemmatize the words, -- convering all the words to its root word.

def lemma(data):
    data["tweet"] = data["tweet"].apply(lambda  x: " ".join([Word(x).lemmatize() for x in x.split()]))
    return data

def vector_matrix(data):
    cv = TfidfVectorizer(stop_words="english")
    cv_matrix = cv.fit_transform(data["tweet"])
    return cv_matrix,cv

def test_vector_matrix(data,cv):
    # cv = TfidfVectorizer(stop_words="english")
    cv_matrix = cv.transform(data["tweet"])
    return cv_matrix

def ML_Algorithm(cv_matrix,data):
    mnb = MultinomialNB()
    model = mnb.fit(cv_matrix,data["label"])
    return model

# {"max_depth" :range(15,25,5),"min_samples_split":range(5,20,5),"min_samples_leaf":range(5,20,5),"max_features":range(10,30,10),"n_estimators":range(60,150,20)}

# def Random_Forest_tuning(cv_matrix,data):
#     rf = RandomForestClassifier(random_state=20)
#     param = {"min_samples_split":range(2,5,2),"min_samples_leaf":range(2,5,2),"max_features":range(2,5,1)}
#     model_tune = GridSearchCV(estimator=rf,param_grid=param,scoring="accuracy",cv=5,verbose=1)
#     model = model_tune.fit(cv_matrix,data["label"])
#     print(model.best_params_)
#     return model.best_params_

# max_features = best_param["max_features"],
#                                 min_samples_leaf = best_param["min_samples_leaf"],
#                                 min_samples_split = best_param["min_samples_split"],
#                                 n_estimators = 100

def Random_Forest (cv_matrix,data):
    rf = RandomForestClassifier(random_state=20)
    model = rf.fit(cv_matrix,data["label"])
    return model

def count_vetcor_matrix(data):
    ct = CountVectorizer(max_features=1000,stop_words="english")
    ct_matrix = ct.fit_transform(data["tweet"])
    return ct_matrix,ct

def predict(cv_matrix,model):
    predicted = model.predict(cv_matrix)
    return predicted

def remove_common_words(data):
    freq_table = pd.Series(" ".join(data["tweet"]).split()).value_counts()
    length = len(pd.Series(" ".join(data["tweet"]).split()).value_counts())
    freq_table = freq_table[freq_table>900]
    data["tweet"] = data["tweet"].apply(lambda x: " ".join([x for x in x.split() if x not in freq_table]))
    return data

def testing(data,model,cv,label):
    stop = stopwords.words("english")
    data=cleaning_extra_chara(data)
    data = lemma(data)
    data=text_blob_stop_words(data,stop)
    cv_matrix = test_vector_matrix(data,cv)
    predicted = model.predict(cv_matrix)
    data[label] = predicted
    # print(data.head())
    return data

def main(data):
    stop = stopwords.words("english")
    data=cleaning_extra_chara(data)
    data = lemma(data)
    data=text_blob_stop_words(data,stop)
    data = remove_common_words(data)
    cv_matrix,cv = vector_matrix(data)
    ct_matrix,ct = count_vetcor_matrix(data)
    model_1 = ML_Algorithm(cv_matrix,data)
    model_2 = Random_Forest(cv_matrix,data)
    model_3 = ML_Algorithm(ct_matrix,data)
    model_4 = Random_Forest(ct_matrix,data)

    # best_param = Random_Forest_tuning(cv_matrix,data)
    # predicted = predict(cv_matrix,model)
    # data["predict"] = predicted
    # print(data.head())
    # print(np.mean(data["predict"] - data["label"] ))
    return model_1,model_2,cv,ct,model_3,model_4


# Splitting Data Sets.
# print(len(train["tweet"]))
X_train = train.iloc[0:24000,:]
X_test = train.iloc[24000:,:]

# Testing the model with available datset, 
train = X_train 
test =X_test
Y_actual = test["label"]

model_1,model_2,cv,ct,model_3,model_4 = main(train)
final_data = testing(test,model_1,cv,"label_1")
final_data = testing(final_data,model_2,cv,"label_2")
final_data = testing(final_data,model_3,ct,"label_3")
final_data = testing(final_data,model_4,ct,"label_4")

# print("tfidf NB \n",classification_report(test["label"],final_data["label_1"]))
# print("tfidf RF \n",classification_report(test["label"],final_data["label_2"]))
# print("CT NB \n",classification_report(test["label"],final_data["label_3"]))
# print("CT RF \n",classification_report(test["label"],final_data["label_4"]))

final_data["label"] = final_data["label_1"] + final_data["label_2"]+ final_data["label_3"]+ final_data["label_4"]
final_data["label"] = final_data["label"].apply(lambda  x: 1 if x>0 else 0)
# print(final_data.head())

final_data.drop(["tweet","label_1","label_2","label_3","label_4"],axis=1,inplace=True)
print(pd.Series(final_data["label"]).value_counts())
final_data.to_csv("test.csv",index=False)

print(accuracy_score(test["label"],final_data["label"]))
print(classification_report(test["label"],final_data["label"]))


