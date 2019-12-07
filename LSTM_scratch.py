# skip-gram_hierarchical-softmax
import numpy as np
import random
import math
import time
from datetime import datetime
import operator
import sys
from sklearn.metrics.pairwise import cosine_similarity




def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


f = open("text9.txt", 'r')
words = f.read().split(' ')
f.close()
words=words[:34]

# text에 있는 전체 단어의 수를 센다.
total_word_number = np.array(words).shape[0]
print('단어 수 :', total_word_number)

# first_dic에는 (word: 그 단어수)의 dictionary를 부여한다.
prime_dic = {}

# dic에는 unknown이라는 단어를 포함한다.
dic = {}

# index_of_word에는 해당 단어의 index, 배열 내 위치 를 저장한다.
index_of_word = {}
unknown_number = 0

# 먼저 first_dic에 각 단어의 개수를 센다.
for word in words:
    if word != '':
        if word in prime_dic.keys():
            prime_dic[word] += 1
        else:
            prime_dic[word] = 1

# unknown_number이하로 등장하는 단어는 UNKNOWN으로 처리한다.
dic['UNKNOWN'] = 0
index_of_word['UNKNOWN'] = 0

index = 1
for x, y in prime_dic.items():
    if y < unknown_number:
        dic['UNKNOWN'] += y
    else:
        dic[x] = y
        index_of_word[x] = index
        index += 1

vocabulary_size = len(index_of_word)
print('vocabulary_size :', vocabulary_size)
word_of_index={y:x for x,y in index_of_word.items()}
# unknown으로 취급할 최소 개수
unknown_number = 0

#train data 만들기
X_train=[]
y_train=[]

i=0
for _ in range(1000):
    X_train.append([index_of_word[j]for j in words[i:i+10]])
    y_train.append([index_of_word[j]for j in words[i+1:i+11]])
    words.append(words.pop(0))

print(X_train)
print(y_train)



class LSTMumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (4*hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4*hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        c = np.zeros((T + 1, self.hidden_dim))
        o = np.zeros((T, self.hidden_dim))
        i_=np.zeros((T, self.hidden_dim))
        o=np.zeros((T, self.hidden_dim))
        f=np.zeros((T, self.hidden_dim))
        g=np.zeros((T, self.hidden_dim))
        pred=np.zeros((T, self.word_dim))
        #x, hs, c, is_, f, o, g, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}, {}
        s[-1] = np.zeros(self.hidden_dim)  # t=0일때 t-1 시점의 hidden state가 필요하므로
        c[-1] = np.zeros(self.hidden_dim)
        H = self.hidden_dim
        # forward pass
        for t in np.arange(T):
            temp = self.U[:, x[t]] + self.W.dot(s[t - 1])
            i_[t] = sigmoid(temp[:H])
            f[t] = sigmoid(temp[H:2*H])
            o[t] = sigmoid(temp[2*H:3*H])
            g[t] = np.tanh(temp[3*H:])
            c[t] = f[t]*(c[t - 1]) + i_[t]*(g[t])
            s[t] = o[t] * np.tanh(c[t])
            pred[t]=softmax(self.V.dot(s[t]))
        return pred, o, s, c, f, g, i_




    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        pred,  o, s, c, f, g, i_ = self.forward_propagation(x)
        return np.argmax(pred, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            pred, o, s, c, f, g, i_ = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = pred[np.arange(len(y[i])), y[i]]
            #print(len(y[i]),y[i])
            #print(pred[np.arange(len(y[i])), y[i]])
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L


    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N




    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        pred, o, s, c, f, g, i_  = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        #delta_o = o
        dhnext = np.zeros_like(s[0])
        dcnext = np.zeros_like(c[0])

        n = 1
        a = len(y) - 1
        for t in np.arange(T)[::-1]:
            if n > len(y):
                continue
            dy = np.copy(pred[t])  # shape (num_chars,1).  "dy" means "dloss/dy"
            dy[y[t]] -= 1
            #print('dy.shape :', dy.shape)
            #print('s[t].shape :', s[t].shape)
            dLdV += np.outer(dy, s[t])
            dh = np.dot(self.V.T, dy) + dhnext  # backprop into h.
            dc = dcnext + (1 - np.tanh(c[t]) * np.tanh(c[t])) * dh * o[t]  # backprop through tanh nonlinearity
            dcnext = dc * f[t]
            di = dc * g[t]
            df = dc * c[t - 1]
            do = dh * np.tanh(c[t])
            dg = dc * i_[t]
            ddi = (1 - i_[t]) * i_[t] * di
            ddf = (1 - f[t]) * f[t] * df
            ddo = (1 - o[t]) * o[t] * do
            ddg = (1 - g[t]**2) * dg
            #da = np.hstack((ddi.ravel(), ddf.ravel(), ddo.ravel(), ddg.ravel()))
            da = np.hstack((ddi.ravel(),ddf.ravel(),ddo.ravel(),ddg.ravel()))
            dLdU += np.outer(da, x[t])
            dLdW += np.outer(da, s[t - 1])
            dhnext = np.dot(self.W.T, da)
            n += 1
            a -= 1
        for dparam in [dLdV , dLdU, dLdW]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients.

        return [dLdU, dLdV, dLdW]


    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV


    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
                #print(X_train[i])
                #print(self.predict(X_train[i]))
                num_examples_seen += 1


np.random.seed(10)
# Train on a small subset of the data to see what happens
model = LSTMumpy(vocabulary_size)
losses = model.train_with_sgd(X_train[:100], y_train[:100], nepoch=500, evaluate_loss_after=50)

x=['the', 'english', 'revolution', 'and', 'the', 'sans', 'culottes', 'of', 'the', 'french']
y=[]
for word in x:
    if word not in index_of_word.keys():
        y.append(0)
    else:
        y.append(index_of_word[word])

print('y :', y)
#print('predict :', model.predict(y))
print('predict :', word_of_index[model.predict(y)[-1]])
