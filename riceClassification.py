import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#read data csv
from google.colab import files
data = files.upload()


#pre-process
data = pd.read_csv('riceClassification.csv')

data.head()

data.tail()

x=1
for col in data.columns:
    print(f'Kolom {x} :',col)
    x+=1
data.describe()

#drop 'id' attribute
data = data.drop(['id'], axis=1)

data.head()

data.info()

data.isnull().sum()

data.duplicated().sum()

data_hist = data.copy()
data_hist.hist(figsize=(15,15))
plt.show()

data_hist.plot.box(subplots=True, figsize=(14,14))
plt.show()

X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X[:10],y[:10]


#Adaline
import matplotlib.pyplot as plt
import numpy as np

def line(w, th=0):
    w2 = w[2] + .001 if w[2] == 0 else w[2]

    return lambda x: (th - w[1] * x - w[0]) / w2


def plot(func, X, target, padding=1, marker='o'):
    X = np.array(X)

    x_vals, y_vals = X[:, 1], X[:, 2]
    xmin, xmax, ymin, ymax = x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()

    markers = f'r{marker}', f'b{marker}'
    line_x = np.arange(xmin-padding-1, xmax+padding+1)

    for c, v in enumerate(np.unique(target)):
        p = X[np.where(target == v)]

        plt.plot(p[:,1], p[:,2], markers[c])

    plt.axis([xmin-padding, xmax+padding, ymin-padding, ymax+padding])
    plt.plot(line_x, func(line_x))
    plt.show()

#step bipolar
def bipstep(y, th=0):
  return 1 if y >= th else -1

#training Adaline
import sys

def adaline_fit(x, t, alpha=.5, max_err=.1, max_epoch=-1, verbose=False, draw=True):
  w = np.random.uniform(0, 1, len(x[0]) + 1)
  b = np.ones((len(x), 1))
  x = np.hstack((b, x))
  stop = False
  epoch = 0
  
  while not stop and (max_epoch == -1 or epoch < max_epoch):
    epoch += 1
    max_ch = -sys.maxsize
    
    if verbose:
      print('\nEpoch', epoch)
    
    for r, row in enumerate(x):
      y = np.dot(row, w)
      
      for i in range(len(row)):
        w_new = w[i] + alpha * (t[r] - y) * row[i]
        max_ch = max(abs(w[i] - w_new), max_ch)
        w[i] = w_new
        
        if verbose:
          print('Bobot:', w)
          
        if draw:
          plot(line(w), x, t)
    
    stop = max_ch < max_err
  
  return w, epoch

#testing Adaline
def adaline_predict(X, w):
  Y = []
  
  for x in X:
    y_in = w[0] + np.dot(x, w[1:])
    y = bipstep(y_in)
    Y.append(y)
    
  return Y

#prediction & evaluation
train = minmax_scale(data)
target = data.Class
w, epoch = adaline_fit(train, target, verbose=True, draw=True)
output = adaline_predict(train, w)
accuracy = accuracy_score(output, target)

print('Output:', output)
print('Epoch:', epoch)
print('Target:', target)
print('Accuracy:', accuracy)


#SOM
train = minmax_scale(data)
target = data.Class
w, epoch = adaline_fit(train, target, verbose=True, draw=True)
output = adaline_predict(train, w)
accuracy = accuracy_score(output, target)

print('Output:', output)
print('Epoch:', epoch)
print('Target:', target)
print('Accuracy:', accuracy)

#prediction & evaluation
data_som = data.copy()
data_som.drop(['Class'], axis=1)
X = minmax_scale(data_som)
target = data_som.Class
centroids, labels = som(X, lrate=.1, b=.05, max_epoch=100, n_cluster=2)
silhouette = silhouette_score(X, labels)
db_index = davies_bouldin_score(X, labels)

print('Silhouette score:', silhouette)
print('Davies Bouldin Index:', db_index)
draw(X, target, centroids)


#LVQ
def lvq_fit(train, target, lrate, b, max_epoch):
    start_time = time.time()
    label, train_idx = np.unique(target, return_index=True)
    weight = train[train_idx].astype(np.float64)
    train = np.array([e for i, e in enumerate(zip(train, target)) if i not in train_idx])
    train, target = train[:, 0], train[:, 1]
    epoch = 0

    while epoch < max_epoch:
        for i, x in enumerate(train):
            distance = [sum((w - x) ** 2) for w in weight]
            min = np.argmin(distance)
            sign = 1 if target[i] == label[min] else -1
            weight[min] += sign * lrate * (x - weight[min])
        lrate *= b
        epoch += 1
    
    execution = time.time() - start_time
    print("Waktu eksekusi: %s detik" % execution)

    return weight, label

def lvq_predict(X, model):
    center, label = model
    Y = []

    for x in X:
      d = [sum((c - x) ** 2) for c in center]
      Y.append(label[np.argmin(d)])
    
    return Y

X = minmax_scale(data)
Y = data.Class

#prediction & evaluation
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

model = lvq_fit(X_train, y_train, lrate=.1, b=.05, max_epoch=100)
output = lvq_predict(X_test, model)
accuracy = accuracy_score(output, y_test)
confusion = confusion_matrix(output, y_test)
colors = 'rgbcmyk'

print('Accuracy:', accuracy)
print('Confusion matrix:')
print(confusion)

for x, label in zip(X_train, y_train):
    plt.plot(x[0], x[1], colors[label] + '.')
    
for center, label in zip(model[0], model[1]):
    plt.plot(center[0], center[1], colors[label] + 'o')
for x, label in zip(X_test, output):
    plt.plot(x[0], x[1], colors[label] + 'x')

 
#Backpropagation
def bp_fit(X, target, layer_conf, max_epoch, max_error=.1, learn_rate=.1, print_per_epoch=100):
  start_time = time.time()
  np.random.seed(1)
  nin = [np.empty(i) for i in layer_conf]
  n = [np.empty(j + 1) 
  if i < len(layer_conf) - 1 else np.empty(j) for i, j in enumerate(layer_conf)]
  w = np.array([np.random.rand(layer_conf[i] + 1, layer_conf[i + 1]) for i in range(len(layer_conf) - 1)])
  dw = [np.empty((layer_conf[i] + 1, layer_conf[i + 1])) for i in range(len(layer_conf) - 1)]
  d = [np.empty(s) for s in layer_conf[1:]]
  din = [np.empty(s) for s in layer_conf[1:-1]]
  epoch = 0
  mse = 1

  for i in range(0, len(n)-1):
    n[i][-1] = 1
  
  while (max_epoch == -1 or epoch < max_epoch) and mse > max_error:
    epoch += 1
    mse = 0
    
    for r in range(len(X)):
        n[0][:-1] = X[r]
  
        for L in range(1, len(layer_conf)):
            nin[L] = np.dot(n[L-1], w[L-1])
            n[L][:len(nin[L])] = sig(nin[L])

        e = target[r] - n[-1]
        mse += sum(e ** 2)
        d[-1] = e * sigd(nin[-1])
        dw[-1] = learn_rate * d[-1] * n[-2].reshape((-1, 1))

        for L in range(len(layer_conf) - 1, 1, -1):
          din[L-2] = np.dot(d[L-1], np.transpose(w[L-1][:-1]))
          d[L-2] = din[L-2] * np.array(sigd(nin[L-1]))
          dw[L-2] = (learn_rate * d[L-2]) * n[L-2].reshape((-1, 1))
        
        w += dw
  
    mse /= len(X)
  
    if print_per_epoch > -1 and epoch % print_per_epoch == 0:
      print(f'Epoch {epoch}, MSE: {mse}')
    
    execution = time.time() - start_time
    print("Waktu eksekusi: %s detik" % execution)

  return w, epoch, mse

def bp_predict(X, w):
    n = [np.empty(len(i)) for i in w]
    nin = [np.empty(len(i[0])) for i in w]
    predict = []
    
    n.append(np.empty(len(w[-1][0])))

    for x in X:
        n[0][:-1] = x
       
        for L in range(0, len(w)):
            nin[L] = np.dot(n[L], w[L])
            n[L + 1][:len(nin[L])] = sig(nin[L])

        predict.append(n[-1].copy())

    return predict

X = minmax_scale(data)
Y = onehot_enc(data.Class)

#one hot encoding
import numpy as np

def onehot_enc(lbl, min_val=0):
  mi = min(lbl)
  enc = np.full((len(lbl), max(lbl) - mi + 1), min_val, np.int8)
  
  for i, x in enumerate(lbl):
    enc[i, x - mi] = 1
  
  return enc

def onehot_dec(enc, mi=0):
  return [np.argmax(e) + mi for e in enc]

#Sigmoid
def sig(X):
  return [1 / (1 + np.exp(-x)) for x in X]


def sigd(X):
  output = []
  
  for i, x in enumerate(X):
    s = sig([x])[0]
    
    output.append(s * (1 - s))
  
  return output

#prediction & evaluation
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3,random_state=1)
w, ep, mse = bp_fit(X_train, y_train, layer_conf=(11, 3, 2), learn_rate=.1, max_epoch=100, max_error=.5, print_per_epoch=25)

print(f'Epochs: {ep}, MSE: {mse}')

predict = bp_predict(X_test, w)
predict = onehot_dec(predict)
y_test = onehot_dec(y_test)
accuracy = accuracy_score(predict, y_test)
confusion = confusion_matrix(predict, y_test)

print('Output:', predict)
print('True :', y_test)
print('Accuracy:', accuracy)
print('Confusion matrix:')
print(confusion)
