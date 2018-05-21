import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# build a dictionary {<word>: <number of word occurences in directory}
def build_counts(directory):
    cnt = Counter()
    # get files in the folder, omit system files
    files = [os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith(".")]
    for file in files:
        with open(file) as f:
            for line in f:
                for word in line.split():
                    cnt[word] += 1
    word_counts = dict(cnt.most_common(3000))
    return word_counts

# build a dictionary {<word>: <word_id}
# word_id is just an index of a word in the passed "words" list
def build_words2ids(words):
    words2ids = dict()
    for i, word in enumerate(words):
        words2ids[word] = i
    return words2ids

# builds 2 dictionaries:
# train_labels = {<doc_id>: <1 if spam, 0 if not spam>}
# features_matrix = {<doc_id>: {<word_id>: <number of occurences of word in doc>}}
def extract_features(word_counts, words2ids, directory):
    # get files in the folder, omit system files
    files = [os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith(".")]
    features_matrix = np.zeros((len(files), len(word_counts)))
    train_labels = np.zeros(len(files))
    doc_id = 0
    for file in files:
        last_token = file.split('/')[-1]
        if last_token.startswith("spmsg"):
            train_labels[doc_id] = 1
        with open(file) as f:
            f.readline() # skip email subject
            f.readline() # skip second empty line
            body = f.readline()
            for word in body.split():
                if word in words2ids:
                    word_id = words2ids[word]
                    features_matrix[doc_id][word_id] += 1
            
        doc_id += 1
    return features_matrix, train_labels

TEST_DIR = "../test-mails/"
TRAIN_DIR = "../train-mails/"

word_counts = build_counts(TRAIN_DIR)
words = list(word_counts.keys())
words2ids = build_words2ids(words)

train_features_matrix, train_labels = extract_features(word_counts, words2ids, TRAIN_DIR)
model = GaussianNB()
model.fit(train_features_matrix, train_labels)

## train accuracy 
predicted_labels = model.predict(train_features_matrix)
train_accuracy = accuracy_score(train_labels, predicted_labels)
print("accuracy on train set: " + str(train_accuracy))

## test accuracy
test_features_matrix, test_labels = extract_features(word_counts, words2ids, TEST_DIR)
predicted_labels = model.predict(test_features_matrix)
test_accuracy = accuracy_score(test_labels, predicted_labels)
print("accuracy on test set: " + str(test_accuracy))
