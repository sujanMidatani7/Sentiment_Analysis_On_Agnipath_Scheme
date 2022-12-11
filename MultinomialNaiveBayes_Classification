#We have tried to train the comment data with our own classifier methods - Multinomial- NaiveBayes. The Trainingf and test accuracy are normal(95 and 60 Approx.)
#The code to it is as follows..

pip install nltk

#To import DataFrames from Google Drive
import pandas as pd
#To pre-process the data , making it apt for the model
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#Importing Classifier, Splitting of Training and test data and Accuracy Score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
#Reading and storing correct column data
data = pd.read_csv('/content/drive/MyDrive/TCD_modified_850_1000.csv')
data=data.iloc[0:396,:]
# print(data)

#Fitting the data , suitable for the model
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(data['Opinion'])

#Division of data to train and test parts
X_train, X_test, Y_train, Y_test = train_test_split(text_counts, data['Target'], test_size=0.35, random_state=35)

#Fitting the model with the train data
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)

predicted = MNB.predict(X_test)

p_train=MNB.predict(X_train)
acc_p=metrics.accuracy_score(p_train,Y_train)
accuracy_score = metrics.accuracy_score(predicted, Y_test)
print("Acc: ",acc_p,"\n",accuracy_score)

a=[str(i) for i in data['Opinion']]
o = len(a.pop())
#Checking the model with our own data
x=input("enter comment with exactly ", o," letters")
dic={"Opinion ":[x]+a+["hummy "]}
dat=pd.DataFrame(dic)

text_counts = MNB.predict(cv.fit_transform(dat['Opinion ']))
print(text_counts[0])#<--- This is the prediction value
