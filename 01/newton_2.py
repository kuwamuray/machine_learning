import numpy as np
import numpy.random as rd

e = 0.1
l = 0.1

n = 40
omega = rd.rand(1,1)
noise = 0.8 * rd.rand(n,1)
x = rd.rand(n,2)
y = np.zeros([n,1])
for i in range(n):
    k = omega * x[i,0] + x[i,1] + noise[i]
    if k > 0 :
        y[i] = 1
y_3 = np.ones([n,1])
y = 2 * y - y_3
w = rd.rand(2,1)

def calculate_Jw(w):
    c_1 = np.dot(x,w)
    c_2 = - c_1 * y
    c_3 = np.exp(c_2)
    c_4 = c_3 + y_3
    c_5 = np.log(c_4)
    Jw_1 = np.sum(c_5)
    Jw_2 = l * (np.dot(np.transpose(w),w))
    Jw = Jw_1 + Jw_2
    return Jw

def gradient(w):
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

def gradient_Hessian(w):
    Hessian = np.zeros((2,2))
    h = 0.0000001
    for i in range(2):
        for j in range(2):
            w[i] += h
            w[j] += h
            H_1 = calculate_Jw(w)
            w[i] -= h
            w[i] -= h
            H_2 = calculate_Jw(w)
            w[j] -= h
            w[j] -= h
            H_3 = calculate_Jw(w)
            w[i] += h
            w[i] += h
            H_4 = calculate_Jw(w)
            w[i] -= h
            w[j] += h
            Hessian[i,j] = (H_1 - H_2 + H_3 - H_4) / (4 * h * h)
    return Hessian

for i in range(100):
    print()
    print("i = " + str(i))
    print("w = " + str(np.transpose(w)))
    print("Jw = " + str(np.transpose(calculate_Jw(w))))
    g = gradient(w)
    H = gradient_Hessian(w)
    print("H = " + str(H))
    print("grad = " + str(np.transpose(g)))
    w -= np.dot(np.linalg.inv(H),g)
    print()
