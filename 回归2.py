import numpy as np
import matplotlib.pyplot as plt

train_x = [235,216,148,35,85,204,49,25,173,191,134,99,117,112,162,272,159,159,59,198]
train_y = [591,539,413,310,308,519,325,332,498,498,392,334,385,387,425,659,400,427,319,522]

theta = np.random.rand(3)

mu = np.mean(train_x)
std = np.std(train_x)
def standardize(x):
    return (x - mu)/std

train_z = standardize(train_x)


def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

X = to_matrix(train_z)
def f(x):
    return np.dot(x,theta)

def E(x,y):
    return 0.5 * np.sum((y-1f(x)) ** 2)

diff = 1
ETA = 1e-3
error = E(X,train_y)

while diff > 1e-2:
    theta = theta - ETA * np.dot(f(X)-train_y,X)
    cur_error = E(X,train_y)
    diff = error - cur_error
    error = cur_error
    
x = np.linspace(-3,3,100)
plt.plot(train_z,train_y,'o')
plt.plot(x,f(to_matrix(x)))
plt.show()
