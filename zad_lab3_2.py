import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# zad3_1_1 ---------------------------------------------
def and_func(x):
    return int(x[0] and x[1])

def not_func(x):
    return int(not x[0])

def step_func(x):
    return np.where(x >= 0, 1, 0)

def perceptron_learn(x, y, w, b, learning_rate):
    epochs = 100
    for epoch in range(epochs):
        errors = 0
        for i in range(len(x)):
            net = np.dot(x[i], w) + b
            y_hat = step_func(net)
            error = y[i] - y_hat
            if error != 0:
                errors += 1
                w = w + learning_rate * error * x[i]
                b = b + learning_rate * error
        if errors == 0:
            # print("Szkolenie zbieżne epoch", epoch+1)
            break
    return w, b

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_not = np.array([1, 0, 1, 0])

w_and = np.zeros(2)
b_and = 0
w_not = np.zeros(1)
b_not = 0

learning_rate = 0.1
w_and, b_and = perceptron_learn(x_train, y_and, w_and, b_and, learning_rate)
w_not, b_not = perceptron_learn(x_train[:, [0]], y_not, w_not, b_not, learning_rate)

# zad3_1_2 ------------------------------------------------------
def bool_func(x):
    return int(x[0] and not x[1])

def step_func(x):
    return np.where(x >= 0, 1, 0)

def perceptron(x, w, b):
    net = np.dot(x, w) + b
    y_hat = step_func(net)
    return y_hat

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 1, 0])

w = np.array([1, -1, -1])
b = 0

learning_rate = 0.1
epochs = 100
for epoch in range(epochs):
    errors = 0
    for i in range(len(x_train)):
        x = np.array([x_train[i][0], not x_train[i][1], 1])
        y = y_train[i]
        y_hat = perceptron(x, w, b)
        error = y - y_hat
        if error != 0:
            errors += 1
            w = w + learning_rate * error * x
            b = b + learning_rate * error
    if errors == 0:
        # print("Szkolenie zbieżne epoch", epoch+1)
        break

# zad3_1_3 ----------------------------------------------------
def bool_func(x):
    return int(x[0] != x[1])

def step_func(x):
    return np.where(x >= 0, 1, 0)

def neural_network(x, w1, w2, b1, b2):
    net1 = np.dot(x, w1) + b1
    y1 = step_func(net1)
    net2 = np.dot(y1, w2) + b2
    y_hat = step_func(net2)
    return y_hat

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])

w1 = np.random.rand(2, 2)
w2 = np.random.rand(2)
b1 = np.zeros(2)
b2 = 0

learning_rate = 0.1
epochs = 1000
for epoch in range(epochs):
    errors = 0
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        net1 = np.dot(x, w1) + b1
        y1 = step_func(net1)
        net2 = np.dot(y1, w2) + b2
        y_hat = step_func(net2)
        error = y - y_hat
        if error != 0:
            errors += 1
            delta2 = error * y_hat * (1 - y_hat)
            w2 = w2 + learning_rate * delta2 * y_hat
            b2 = b2 + learning_rate * delta2
            delta1 = delta2 * w2 * y1 * (1 - y1)
            w1 = w1 + learning_rate * delta1.reshape(-1, 1) * x.reshape(1, -1)
            b1 = b1 + learning_rate * delta1
    if errors == 0:
        # print("Szkolenie zbieżne epoch", epoch+1)
        break

# zad3_1_4 -----------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])

np.random.seed(1)
w1 = 2 * np.random.random((2, 4)) - 1
w2 = 2 * np.random.random((4, 1)) - 1
b1 = np.zeros((1, 4))
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    z1 = np.dot(x_train, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    y_hat = sigmoid(z2)

    error = y_train.reshape(-1, 1) - y_hat

    delta2 = error * sigmoid_derivative(z2)
    d_w2 = np.dot(a1.T, delta2)
    d_b2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = np.dot(delta2, w2.T) * sigmoid_derivative(z1)
    d_w1 = np.dot(x_train.T, delta1)
    d_b1 = np.sum(delta1, axis=0)

    w2 += learning_rate * d_w2
    b2 += learning_rate * d_b2
    w1 += learning_rate * d_w1
    b1 += learning_rate * d_b1

    if (epoch + 1) % 1000 == 0:
        print("Epoch:", epoch + 1, "Blad:", np.mean(np.abs(error)))

print("w1 =", w1)
print("b1 =", b1)
print("w2 =", w2)
print("b2 =", b2)

x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.array([0, 1, 1, 0])
z1 = np.dot(x_test, w1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
y_hat = sigmoid(z2)
print("Y =", np.round(y_hat, decimals=4))

#------------------------------------------------------------------------------------

years = np.array([2000, 2002, 2005, 2007, 2010]).reshape(-1, 1)
percentages = np.array([6.5, 7.0, 7.4, 8.2, 9.0])

model = LinearRegression().fit(years, percentages)

def predict_unemployment(year):
    return model.predict(np.array([year]).reshape(-1, 1))[0]

year = 2010
while predict_unemployment(year) < 12:
    year += 1

plt.scatter(years, percentages, color='blue', label='Dane historyczne')
plt.plot(years, model.predict(years), color='red', label='Regresja liniowa')

plt.xlabel('Rok')
plt.ylabel('Procent bezrobotnych')
plt.title('Model regresji liniowej dla procentu bezrobotnych')

# plt.legend()
# plt.show()

# print(f"Wynik modelu regresji liniowej dla roku 2015: {predict_unemployment(2015):.3f}")
# print(f"Procent bezrobotnych przekroczy 12% w roku {year}.")
# print("\n")


# zad3_1_1 show
for x in x_train:
    # print("AND({}, {}) = {}".format(x[0], x[1], and_func(x)))
    # print("NOT({}) = {}".format(x[0], not_func(x)))
    net_and = np.dot(x, w_and) + b_and
    y_hat_and = step_func(net_and)
    net_not = np.dot(x.reshape(-1, 1), w_not) + b_not
    y_hat_not = step_func(net_not)
    # print("AND({}, {}) = {}".format(x[0], x[1], y_hat_and))
    # print("\n")

# zad3_1_2 show
for x in x_train:
    y_hat = perceptron(np.array([x[0], not x[1], 1]), w, b)
#     print("{} ∧ ¬{} = {}".format(x[0], x[1], bool_func(x)))
#     print("\n")

# zad3_1_3 show
for x in x_train:
    net1 = np.dot(x, w1) + b1
    y1 = step_func(net1)
    net2 = np.dot(y1, w2) + b2
    y_hat = step_func(net2)
    # print("{} XOR {} = {}".format(x[0], x[1], bool_func(x)))
