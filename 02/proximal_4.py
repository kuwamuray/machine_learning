import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd

c = np.zeros((2,1))
d = np.zeros((2,1))
l = 4
u = np.array([[1],[2]])
w = rd.rand(2,1)
x = np.zeros(100)
y = np.zeros(100)
A = np.array([[3, 0.5],[0.5, 1]])
K = np.linalg.eig(2 * A)
B = np.max(K[0])

e = 1 / B
next_w = np.zeros((2,1))
w_hat =  np.array([[7/11],[2/11]])

def calculate_psi(w):
    psi_2 = w - u
    psi_1 = np.transpose(psi_2)
    psi = np.dot(np.dot(psi_1,A),psi_2)
    return psi

def calculate_Jw(w):
    Jw_1 = calculate_psi(w)
    Jw_2 = l * w[0]
    Jw_3 = l * w[1]
    if Jw_2 < 0 :
        Jw_2 = Jw_2 * (-1.0)
    if Jw_3 < 0 :
        Jw_3 = Jw_3 * (-1.0)
    Jw = Jw_1 + Jw_2 + Jw_3
    return Jw

def gradient(w):
    grac_1 = np.zeros(w.shape)
    grac_2 = np.zeros(w.shape)
    h = 0.0000001
    grad = np.zeros(w.shape)
    for i in range(2):
        w[i] += h
        grac_1 = calculate_psi(w)
        w[i] -= h
        w[i] -= h
        grac_2 = calculate_psi(w)
        w[i] += h
        grad[i] = (grac_1 - grac_2) / (2.0 * h)
    return grad

def gradient_Jw(w):
    grac_1 = np.zeros(w.shape)
    grac_2 = np.zeros(w.shape)
    h = 0.0000001
    grad = np.zeros(w.shape)
    for i in range(2):
        w[i] += h
        grac_1 = calculate_Jw(w)
        w[i] -= h
        w[i] -= h
        grac_2 = calculate_Jw(w)
        w[i] += h
        grad[i] = (grac_1 - grac_2) / (2.0 * h)
    return grad

for i in range(100):
    print()
    print("i = " + str(i))
    print("w = " + str(np.transpose(w)))
    print("Jw = " + str(np.transpose(calculate_Jw(w))))
    g = gradient(w)
    gjw = gradient_Jw(w)
    print("grad = " + str(np.transpose(gjw)))
    for j in range(2):
        c[j] = w[j] - e * g[j]
        if c[j] < - e * l :
            next_w[j] = c[j] + e * l
        elif c[j] > e * l :
            next_w[j] = c[j] - e * l
        else :
            next_w[j] = 0.0
    w = next_w
    print()
    x[i] = i
    for j in range(2):
        d[j] = w[j] - w_hat[j]
    y[i] = np.sqrt(d[0] * d[0] + d[1] * d[1])

plt.plot(x,y)
plt.yscale('log')
plt.show()
