import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == '__main__':
    b0, b1 = -34, 0.5999
    lr = 0.00001

    x_val = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
    y_val = np.array([49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68])
    print('1st Loss: ' + str(computeCost(x_val, y_val, b0, b1)))
    for it in range(100000):
        b0, b1 = gradientDescent(x_val, y_val, b0, b1, lr)
            
    print('Last Loss: ' + str(computeCost(x_val, y_val, b0, b1)))
    print('B0: ' + str(b0) + ' ' + 'B1: ' + str(b1))

    x0 = np.linspace(145, 185, 2)
    y0 = b0 + b1 * x0

    plt.plot(x_val, y_val, 'ro')
    plt.plot(x0, y0)
    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()