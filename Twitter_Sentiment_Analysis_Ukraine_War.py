# import the required libraries
import numpy as np
import pandas as pd
import seaborn as sns              # for data visualization
import matplotlib.pyplot as plt   # for data visualization
import string                     # to handle the text (upper,lower,puncation,whitespace)
import re                         #to work with regular expression (pattern matching)
import nltk                       # it used for text preprocessing
from nltk.util import pr
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
stemmer = nltk.SnowballStemmer('english') # stemmining

from nltk.stem import WordNetLemmatizer
lemmatizer = nltk.SnowballStemmer('english')# lemmitization

# nltk.download('stopwords')
stopword = set(stopwords.words('english'))


data =pd.read_csv("war_tweets.xls - war_tweets.csv") # importing dataset 

data.head()

data.shape

data.info()

data.describe()

data.isnull().sum()

data.columns

data["tweet"].head()

data["language"].value_counts()

data.language.value_counts().sort_values().plot(kind = 'pie')

data["tweet"][0]

def hashtag_extract(text_list):
    hashtags = []
# Loop over the words in the tweet
    for text in text_list:
        ht = re.findall(r"#(\w+)", text)
        hashtags.append(ht)
    return hashtags

def generate_hashtag_freqdist(hashtags):
    a = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                     'Count': list(a.values())})
    # selecting top 15 most frequent hashtags
    d = d.nlargest(columns="Count", n = 25)
    plt.figure(figsize=(16,7))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    plt.xticks(rotation=80)
    ax.set(ylabel = 'Count')
    plt.show()

hashtags = hashtag_extract(data["tweet"])
hashtags = sum(hashtags, [])


hashtags 


generate_hashtag_freqdist(hashtags)


data['total_length_characters'] = data['tweet'].str.len()
print(data['total_length_characters'])
total_length_characters = data['total_length_characters'].sum()
print(total_length_characters)
count = 0
for y in data["tweet"]:
    count = count + 1
print(count)
average_length = total_length_characters / count
print (average_length)


data['total_count_words'] = data['tweet'].str.split().str.len()
print(data['total_count_words'])
total_words = data['total_count_words'].sum()
print(total_words)
count = 0
for y in data["tweet"]:
    count = count + 1
print(count)
average_words = total_words / count
print (average_words)


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [lemmatizer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)


data["tweet"]


data['total_length_characters'] = data['tweet'].str.len()
print(data['total_length_characters'])
total_length_characters = data['total_length_characters'].sum()
print(total_length_characters)
count = 0
for y in data["tweet"]:
    count = count + 1
print(count)
average_length = total_length_characters / count
print (average_length)


data['total_count_words'] = data['tweet'].str.split().str.len()
print(data['total_count_words'])
total_words = data['total_count_words'].sum()
print(total_words)
count = 0
for y in data["tweet"]:
    count = count + 1
print(count)
average_words = total_words / count
print (average_words)


# get_ipython().system(' pip install TextBlob')


from textblob import TextBlob


def analyze_sentiment(tweet):
    analysis = TextBlob(clean(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


data['Sentiment'] = data['tweet'].apply(lambda x:analyze_sentiment(x))
data['Source'] = 'random_user'
data['Length'] = data['tweet'].apply(len)
data['Word_counts'] = data['tweet'].apply(lambda x:len(str(x).split()))


data['Sentiment']

data['Length']

data['Word_counts'] 


data1=data[['tweet','retweets_counts', 'Sentiment', 'Source',
            'Length','Word_counts']]
data1.head()


data1['Clean tweet'] = data1['tweet'].apply(lambda x:clean(x))


data1[["Clean tweet","Sentiment"]].iloc[100]

sentiment = data1['Sentiment'].value_counts()
sentiment

plt.figure(figsize = (10,8))
sns.countplot(data = data1, x = 'Sentiment')
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (6, 6))
colors = ['b','r','y'] # Color of each section
sizes = [count for count in data1['Sentiment'].value_counts()]
labels = list(data['Sentiment'].value_counts().index)
explode = (0.1, 0, 0) # To slice the perticuler section
ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', explode = explode, textprops = {"fontsize":15},colors = colors)#Font size of text in pie chart
ax.set_title('Sentiment Polarity on invasion Tweets Data \n (total = 9127 tweets)', fontdict=None)
plt.show();


neutral = data1[data1['Sentiment'] == 0]
positive = data1[data1['Sentiment'] == 1]
negative = data1[data1['Sentiment'] == -1]


negative.iloc[1]



#neutral_text
print("Neutral tweet example :",neutral['tweet'].values[15])
# Positive tweet
print("Positive Tweet example :",positive['tweet'].values[37])
#negative_text
print("Negative Tweet example :",negative['tweet'].values[1])


# get_ipython().system('pip install wordcloud')

from wordcloud import WordCloud


txt = ' '.join(text for text in data1['Clean tweet'])
wordcloud = WordCloud(
            background_color = 'white',
            max_font_size = 100,
            max_words = 100,
            width = 800,
            height = 500
            ).generate(txt)
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis('off')
plt.show()


positive_words =' '.join([text for text in data1['Clean tweet'][data1['Sentiment'] == 1]])
wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

negative_words =' '.join([text for text in data1['Clean tweet'][data1['Sentiment'] == -1]])
wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


neutral_words =' '.join([text for text in data1['Clean tweet'][data1['Sentiment'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(neutral_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# So this is how you can analyze the sentiments of people over the Ukraine and Russia war

#from gensim.models import Word2Vec
from nltk.corpus import stopwords

corpus=[]
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [lemmatizer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)

data["tweet"] 


from nltk.corpus import stopwords
print(stopwords.words('english'))


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(data["tweet"]).toarray()


X.shape


y=pd.get_dummies(data1['Sentiment'])
y=y.iloc[:,1].values


y[0]


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# model 1 Nave Bayers:
#  Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

y_pred


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(classification_report(y_pred,y_test))


# model 2 : Logistic regression

# Import the logistic regression model from sklearn 
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)

y_predicted = log_reg.predict(X_test)

accuracy = accuracy_score(y_test,y_predicted)
accuracy

print(classification_report(y_predicted,y_test))


# model 3: Suport vector machine

from sklearn.svm import SVC
svc = SVC(random_state =0)
fitSVC = svc.fit(X_train,y_train)
y_predictedS = fitSVC.predict(X_test)

#print(confusion_matrix(y_test,y_y_predictedS))
print('SVM Accuracy:',accuracy_score(y_test,y_predictedS))
print(classification_report(y_test,y_predictedS))


