import numpy as np;


X = np.array([
    #1, x1, x2, x3, x4
    [1, 1, 2, 3, 4], #1st
    [1, 3, 4, 5, 6], #2nd
    [1, 4, 5, 6, 7], #3rd
    [1, 5, 6, 7, 8] #4th
])

W = np.array([1, 1, 1, 1, 1]).T

Y = np.array([10, 12, 14, 16]).T
print(X.shape)

learning_rate = 0.001

def several_variable_linear_regression():
    global W
    N  = X.shape[0]
    error = X @ W.T - Y
    loss = (error @ error.T) / N
    print(loss)
    gradient_loss = (2 / N) * X.T @ error
    W = W - learning_rate * gradient_loss

if __name__ == "__main__":
    for i in range(10000):
        several_variable_linear_regression()
    print(W)