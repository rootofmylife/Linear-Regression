import matplotlib.pyplot as plt
import numpy as np

# with open('./ex1data2.txt') as f:
#     data = f.read().splitlines()

# data = [s.split(',') for s in data]

# x_val = np.array([int(xv[1]) for xv in data])
# y_val = np.array([int(yv[2]) for yv in data])

x_val = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
y_val = np.array([49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# plt.plot(x_val, y_val, 'ro')
# plt.axis([140, 190, 45, 75])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()

b0, b1 = -33, 0.5

def computeCost(x_train, y_origin, b0, b1): # loss funtion
    sum = 0
    for i in range(len(x_train)):
        sum += ((b0 + b1 * x_train[i]) - y_origin[i])**2
        
    sum = sum / (2 * len(x_train))
    print('Loss: ' + str(sum))

def gradientDescent(x_train, y_origin, b0, b1):
    db0, db1 = 0, 0
    for i in range(len(x_train)):
        db0 += (b0 + b1 * x_train[i] - y_origin[i])
    for i in range(len(x_train)):
        db1 += (b0 + b1 * x_train[i] - y_origin[i]) * x_train[i]
    # computeGradient
    db0 = db0 / len(x_train)
    db1 = db1 / len(x_train)

    return normalizeFeature(b0, b1, db0, db1)

def normalizeFeature(b0, b1, db0, db1):
    learning_rate = 0.0001
    b_0 = b0 - learning_rate * b0
    b_1 = b1 - learning_rate * b1
    return (b_0, b_1)

for it in range(1000):
    if it % 25 == 0:
        computeCost(x_val, y_val, b0, b1)
        b0, b1 = gradientDescent(x_val, y_val, b0, b1)
        print(str(b0) + ' ' + str(b1))