import numpy as np
import matplotlib.pyplot as plt

Lengh = np.linspace(0, 0.5, 10)
Width = np.linspace(0, 0.5, 10)
x_L, y_w = np.meshgrid(Lengh, Width)

[p, h] = A.shape


random_matrix = np.random.rand(p, h)*0.04
alfa = np.zeros((p, h))
for t in range(h):
    for itr in range(p):
         alfa[itr, t] = A[itr, t]+random_matrix[itr, t]

for i in range(h-5):
     #plt.clf()
     plt.figure(i)
     xx, yy = np.meshgrid(alfa[:, 2*i], alfa[:, 2*i+1])
     plt.plot(xx, yy, color='r',  marker='.', linestyle='none')
     plt.plot(x_L, y_w, color='k')
     plt.plot(y_w, x_L, color='k')
     plt.pause(0.05)
     plt.show()




Val=np.array(Best_Fitness)

K=201
alf = np.zeros((K))
random_matrix = np.random.rand(int(K))*0.05

for t in range(int(K)):
     alf[t] = Val[t]+random_matrix[t]


Val=np.array(alf)

theta = 2 * np.pi * Val
area=60*Val**2
colors = theta
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(theta, Val, c=colors, s=area, cmap='hsv', alpha=0.75)