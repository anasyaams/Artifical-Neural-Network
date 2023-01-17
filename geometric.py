#Fungsi Training LVQ
import numpy as np
def lvq_fit(train, target, lrate, b, max_epoch):
    label, train_idx = np.unique(target, return_index=True)
    weight = np.array(train[train_idx].astype(np.float64))
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

    return weight, label
  
#Fungsi Testing LVQ
def lvq_predict(X, model):
    center, label = model
    Y = []

    for x in X:
      d = [sum((c - x) ** 2) for c in center]
      Y.append(label[np.argmin(d)])
    
    return Y
 
#Fungsi Hitung Akurasi
def calc_accuracy(a, b):
    s = [1 if a[i] == b[i] else 0 for i in range(len(a))]
    
    return sum(s) / len(a)
  
#Input
input = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1], [1, 1],
         [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1], [1, 1],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0]]

target = [4, 4, 4, 4, 4, 2, 2, 2, 2,
          4, 4, 4, 4, 4, 2, 2, 2, 2,
          1, 1, 1, 1, 1, 1, 1, 2, 2,
          1, 1, 1, 1, 1, 1, 1, 2, 2,
          1, 1, 1, 1, 1, 1, 1, 3, 3,
          1, 1, 1, 1, 1, 1, 1, 3, 3,
          1, 1, 1, 1, 1, 1, 1, 3, 3,
          1, 1, 1, 1, 1, 1, 1, 3, 3,
          1, 1, 1, 1, 1, 1, 1, 3, 3]

model = lvq_fit(input, target, lrate=.1, b=.8, max_epoch=40)
output = lvq_predict(input, model)
accuracy = calc_accuracy(output, target)

print('Accuracy:', accuracy)
