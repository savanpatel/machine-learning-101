import os
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.metrics import accuracy_score


import cPickle
import gzip

def load(file_name):
    # load the model
    stream = gzip.open(file_name, "rb")
    model = cPickle.load(stream)
    stream.close()
    return model


def save(file_name, model):
    # save the model
    stream = gzip.open(file_name, "wb")
    cPickle.dump(model, stream)
    stream.close()


def make_Dictionary(root_dir):
    all_words = []
    emails = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()

    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)

    return dictionary



def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    train_labels = np.zeros(len(files))
    count = 0;
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        train_labels[docID] = 0;
        filepathTokens = fil.split('/')
        lastToken = filepathTokens[len(filepathTokens) - 1]
        if lastToken.startswith("spmsg"):
            train_labels[docID] = 1;
            count = count + 1
        docID = docID + 1
    return features_matrix, train_labels



TRAIN_DIR = "../train-mails"
TEST_DIR = "../test-mails"

dictionary = make_Dictionary(TRAIN_DIR)

print "reading and processing emails from file."

features_matrix, labels = extract_features(TRAIN_DIR)
test_feature_matrix, test_labels = extract_features(TEST_DIR)

features_matrix = features_matrix[:len(features_matrix)/10]
labels = labels[:len(labels)/10]

save("/tmp/features_matrix", features_matrix)
save("/tmp/labels", labels)
save("/tmp/test_feature_matrix", test_feature_matrix)
save("/tmp/test_labels", test_labels)

model = svm.SVC()

print "Training model."
#train model
model.fit(features_matrix, labels)

predicted_labels = model.predict(test_feature_matrix)

print "FINISHED classifying. accuracy score : "
print accuracy_score(test_labels, predicted_labels)
