import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize 
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import ttk
# nltk.download('sentiwordnet')

df = pd.read_csv('data.csv')
# print(df.head(5))

df.replace({"positive":1,"negative":0,"neutral":2},inplace=True)

edited_sentence= df['sentence'].copy()
df['sentence_without_stopwords'] = edited_sentence
# print(df.head(5))

def preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','', text)
    
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    tokens = [item for item in tokens if item not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatization = [lemmatizer.lemmatize(item) for item in tokens]
    
    return ' '.join(lemmatization)

df['after preprocessing'] = df['sentence'].apply(preprocessing)
# print(df)

pos=neg=obj=count=0

postagging = []

for sentence in df['after preprocessing']:
    list = word_tokenize(sentence)
    postagging.append(nltk.pos_tag(list))

df['pos_tags'] = postagging

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

# Returns list of pos-neg and objective score. But returns empty list if not present in senti wordnet.

def get_sentiment(word,tag):
    wn_tag = penn_to_wn(tag)
    
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    # Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet.Synset instances are the groupings of synonymous words that express the same concept. 
    
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [synset.name(), swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

    pos=neg=obj=count=0
    
senti_score = []

for pos_val in df['pos_tags']:
    senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
    for score in senti_val:
        try:
            pos = pos + score[1]  
            neg = neg + score[2]  
        except:
            continue
    senti_score.append(pos - neg)
    pos=neg=0    
    
df['senti_score'] = senti_score
# print(df['senti_score'])

# print(df.head)

overall=[]
for i in range(len(df)):
    if df['senti_score'][i]>= 0.05:
        overall.append('Positive')
    elif df['senti_score'][i]<= -0.05:
        overall.append('Negative')
    else:
        overall.append('Neutral')
df['Overall Sentiment']=overall

# print(df.head(10))

df['new_sentence'] = df['after preprocessing'].copy()
# print(df.head(5))

bow_counts = CountVectorizer(tokenizer= word_tokenize, ngram_range=(1,3)) # number of n-grams
bow_data = bow_counts.fit_transform(df['new_sentence'])

X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_data, df['Overall Sentiment'], test_size = 0.2, random_state = 0)

# Training the model 
lr_model_all = LogisticRegression() # Logistic regression
lr_model_all.fit(X_train_bow, y_train_bow) # Fitting a logistic regression model

# Predicting the output
test_pred_lr_all = lr_model_all.predict(X_test_bow) # Class prediction

print("Accuracy of test data : ", accuracy_score(y_test_bow, test_pred_lr_all))

# Print a classification report
print(classification_report(y_test_bow,test_pred_lr_all))


# def predict_sentiment():
#     review_text = sentence.get()
#     cleaned_review = preprocess_text(review_text)  # Using preprocess_text instead of review_cleaning
#     sentiment = analyze_sentiment(cleaned_review)
#     result_label.config(text=f"Sentiment: {sentiment}")

# # GUI setup
# window = tk.Tk()
# window.title('Sentiment Analysis')
# window.geometry('400x200')

# label_review = ttk.Label(window, text='Enter a review:')
# label_review.pack()

# entry_review = ttk.Entry(window, width=50)
# entry_review.pack()

# btn_predict = ttk.Button(window, text='Predict Sentiment', command=predict_sentiment)
# btn_predict.pack()

# window.mainloop()
