import numpy as np
import matplotlib.pyplot as plt

train_x = [235,216,148,35,85,204,49,25,173,191,134,99,117,112,162,272,159,159,59,198]
train_y = [591,539,413,310,308,519,325,332,498,498,392,334,385,387,425,659,400,427,319,522]
# plt.plot(train_x,train_y,'o')
# plt.show()
theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
    return theta0 + theta1 * x
def E(x,y):
    return 0.5 * np.sum((y - f(x)) ** 2)

mu = np.mean(train_x)
std = np.std(train_x)
def standardize(x):
    return (x - mu)/std

train_z = standardize(train_x)
# plt.plot(train_z,train_y,'o')
# plt.show()
#学习率
ETA = 1e-3

#误差
differ = 1

#更新次数
count = 0

error = E(train_z,train_y)

while differ > 1e-2:
    temp0 = theta0 - ETA * np.sum(f(train_z) - train_y )
    temp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)

    theta0 = temp0
    theta1 = temp1

    cur_error = E(train_z,train_y)
    differ = error - cur_error
    error = cur_error
    count+=1
    log = ' 第 {} 次 : theta0 = {:.3f}, theta1 = {:.3f}, 差值 = {:.4f}'
    print(log.format(count, theta0, theta1, differ))


x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()
