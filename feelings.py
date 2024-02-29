import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import math
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB

CATEGORY = [
    'Sadness',  # 0
    'Joy',      # 1
    'Love',     # 2
    'Anger',    # 3
    'Fear',     # 4
    'Surprise'  # 5
]

'''
CATEGORY OF LABEL:

0 - Sadness
1 - Joy
2 - Love
3 - Anger
4 - Fear
5 - Surprise
'''

def label_to_name(label: np.float64):
    return CATEGORY[int(label)]

def name_to_label(name: str):
    return CATEGORY.index(name[0].upper() + name.lower()[1:])

# Loading in the dataset
data = pd.read_csv("data.csv")
print("Shape of the dataset: "+str(data.shape))

# Print the types of the data
print(data.dtypes)

# Print the first 3 rows of the data
#print(data.head(3))

# Graphing the dataset to visualize
ax = data["label"].value_counts().sort_index().plot(kind="bar",
                                                    title="Raw Data visualization",
                                                    figsize=(6, 4))
ax.set_xlabel("Emotions")
plt.show()

# Fix biased data
data_results = data["label"].value_counts().sort_index().values
min_val = [None, -1]
vals = {}

new_data = {
    "text": [],
    "label": []
}

for i, value in enumerate(data_results):
    if not min_val[0]:
        min_val = [value, i]
    elif min_val[0] > value:
        min_val = [value, i]

normalized_data = None

for i, label_id in enumerate(data["label"]):
    label_id = str(label_id)
    if not vals.get(label_id):
        vals[label_id] = []

    if len(vals[label_id]) < min_val[0]:
        vals[label_id].append(i)
        new_data["text"].append(data["text"][i])
        new_data["label"].append(int(data["label"][i]))

data["text"] = pd.Series(new_data["text"])
data["label"] = pd.Series(new_data["label"])

ax = data["label"].value_counts().sort_index().plot(kind="bar",
                                                    title="Normalized Data visualization",
                                                    figsize=(6, 5))
ax.set_xlabel("Emotions")
plt.show()

# Classifying emotions
data_classes = data[(data['label'] == 0) | (data['label'] == 1) | (data['label'] == 2) | (data['label'] == 3) |
                    (data['label'] == 4) | (data['label'] == 5)]
print(data_classes, end="\n")

data = data.head(3000)

# Separate the data into x and y
x = data["text"]
y = data["label"]

print("\nX: Axis", end="\n")
print(x.head())
print("\nY: Axis", end="\n")
print(y.head())

# Vectorizing the text
# vocab = CountVectorizer()
#
# vocab.fit_transform([x[50]])
#
nltk.download("stopwords")
nltk.download("words")

def text_process(text):
    # Create a list nopunc that contains all characters in the text string that are not in the string.punctuation set
    nopunc = [char for char in text if char not in string.punctuation]
    # Convert the `nopunc` list back into a string
    nopunc = ''.join(nopunc)
    # Split the `nopunc` string into a list of words
    # Create a new list that contains all words in
    # the `nopunc` list that are not in the `stopwords.words('english')` set
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


vocab = (CountVectorizer(analyzer=text_process))
vocab.fit(x)

x = vocab.transform(x)

#Shape of the matrix:
print("Shape of the sparse matrix: ", x.shape)

#Non-zero occurences:
print("Non-Zero occurences: ",x.nnz)

# DENSITY OF THE MATRIX
density = (x.nnz/(x.shape[0]*x.shape[1]))*100
print("Density of the matrix = ", density)

# Splitting the data set to training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=750)


'''
MultinomialNB classifier
'''

mnb = MultinomialNB()

# Training the model
mnb.fit(x_train, y_train)

# Predict the sentiment
predmb = mnb.predict(x_test)

# Computing the accuracy and rounding to the nearest 2 decimal places
score_nmb = round(accuracy_score(y_test, predmb) * 100, 2)

# Printing the confusion matrix of the classifier
print("Confusion Matrix of Multinomial Naive Bayes:")
print(confusion_matrix(y_test, predmb))

# Classification report
print("Classification report:")
print(classification_report(y_test, predmb, zero_division=0))

'''
Test cases
'''
# pr = data['text'][1]
# print(pr)
# print("Actual Rating:")
# print(data["label"][1])

# while True:
#     pr = input("Input sentence (e to exit): ")
#     if (pr.lower()) == "e": break
#
#     pr_t = vocab.transform([pr])
#     print("Predicted feeling: ")
#     print(label_to_name(mnb.predict(pr_t)[0]))

def calculate(pr: str):
    pr_t = vocab.transform([pr])
    return label_to_name(mnb.predict(pr_t)[0])