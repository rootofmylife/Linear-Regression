import matplotlib.pyplot as plt
import numpy as np

# with open('./ex1data2.txt') as f:
#     data = f.read().splitlines()

# data = [s.split(',') for s in data]

# x_val = np.array([int(xv[1]) for xv in data])
# y_val = np.array([int(yv[2]) for yv in data])



def normalizeFeature(b0, b1, db0, db1):
    learning_rate = 0.001
    b_0 = b0 - learning_rate * db0
    b_1 = b1 - learning_rate * db1
    return (b_0, b_1)

def computeCost(x_train, y_origin, b00, b11): # loss funtion
    sum = 0
    for i in range(len(x_train)):
        sum += ((b00 + b11 * x_train[i]) - y_origin[i])**2
        
    return sum / float((2 * len(x_train)))

def gradientDescent(x_train, y_origin, b0, b1, learning_rate):
    db0, db1 = 0, 0
    for i in range(len(x_train)):
        db0 += (b0 + b1 * x_train[i] - y_origin[i])
        db1 += (b0 + b1 * x_train[i] - y_origin[i]) * x_train[i]

    # computeGradient
    db0 = db0 / float(len(x_train))
    db1 = db1 / float(len(x_train))
    
    b_0 = b0 - (learning_rate * db0)
    b_1 = b1 - (learning_rate * db1)

    return b_0, b_1
    # return normalizeFeature(b0, b1, db0, db1)

if __name__ == '__main__':
    b0, b1 = 0, 0
    lr = 0.00001

    x_val = np.array([147, 150, 153])
    y_val = np.array([49, 50, 51])
    print('Loss 1st: ' + str(computeCost(x_val, y_val, b0, b1)))
    for it in range(10):
        
        # print('Loss: ' + str(computeCost(x_val, y_val, b0, b1)))
        b0, b1 = gradientDescent(x_val, y_val, b0, b1, lr)
            
    print('Loss last: ' + str(computeCost(x_val, y_val, b0, b1)))
    print('B0: ' + str(b0) + ' ' + 'B1: ' + str(b1))

    x0 = np.linspace(145, 185, 2)
    y0 = b0 + b1 * x0

    plt.plot(x_val, y_val, 'ro')
    plt.plot(x0, y0)
    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()