import numpy
import nltk
import tensorflow
import random
import tflearn
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
#nltk.download('punkt')


import json
with open('intents.json') as file:
    data = json.load(file)
    #print(data)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
print("list of all tags based on length of patterns")
print(docs_y)
print(len(docs_y))
print(" ")

print("list of all pattern combine to one ")
print(docs_x)
print(len(docs_x))
print(" ")

print("list of all pattern tokenize to one complete list")
print(words)
print(len(words))
print(" ")

print("list of all tags tokenize to one complete list")
print(labels)
print(len(labels))

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

print(" ")
print("list of all pattern stemmed to one complete list")
print(words)
print(len(words))
print(" ")

print("list of all tags stemmed to one complete list")
print(labels)
print(len(labels))

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)
print(" ")
print("BOW of each words in pattern (docs_x) (as sentence) vs individual stemmed pattern words (words) (as to check frequency) ")
print(training)
print(len(training))

print(" ")
print("BOW of each words in tags (docs_y) (as sentence) vs individual stemmed pattern words (labels) (as to check frequency) ")
print(output)
print(len(output))

training = numpy.array(training)
output = numpy.array(output)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load('./model.tflearn')
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")