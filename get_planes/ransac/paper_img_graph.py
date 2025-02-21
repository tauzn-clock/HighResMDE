import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib 
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
font = {'size'   : 12}

matplotlib.rc('font', **font)

with open("2.csv", mode='r') as file:
    data2 = list(csv.reader(file))
with open("5.csv", mode='r') as file:
    data5 = list(csv.reader(file))
with open("10.csv", mode='r') as file:
    data10 = list(csv.reader(file))

data2 = np.array(data2, dtype=np.float32)
data5 = np.array(data5, dtype=np.float32)
data10 = np.array(data10, dtype=np.float32)

angle = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

data2[:,0] = data2[:,0] * 180/np.pi
data5[:,0] = data5[:,0] * 180/np.pi
data10[:,0] = data10[:,0] * 180/np.pi

data2[:,2] = data2[:,2] * 180/np.pi
data5[:,2] = data5[:,2] * 180/np.pi
data10[:,2] = data10[:,2] * 180/np.pi

fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(True)

ax.plot(angle, data2[:,0], label=r'Open3D, $\sigma=0.002$',linestyle='--',linewidth=3, color='red')
ax.plot(angle, data5[:,0], label=r'Open3D, $\sigma=0.005$',linestyle='--',linewidth=3, color='blue')
ax.plot(angle, data10[:,0], label=r'Open3D, $\sigma=0.010$',linestyle='--',linewidth=3, color='green')

ax.plot(angle, data2[:,2], label=r'Ours, $\sigma=0.002$',linewidth=3, color='red')
ax.plot(angle, data5[:,2], label=r'Ours, $\sigma=0.005$',linewidth=3, color='blue')
ax.plot(angle, data10[:,2], label=r'Ours, $\sigma=0.010$',linewidth=3, color='green')

ax.set_xlabel(r'Plane Angle/deg')
ax.set_ylabel(r'Estimated Angle Error/deg')

ax.set_xlim(60,150)

plt.legend()
plt.savefig("paper_images/angle_error.png", bbox_inches='tight', pad_inches=0, transparent=True)


data2[:,1] = data2[:,1] * 1000
data5[:,1] = data5[:,1] * 1000
data10[:,1] = data10[:,1] * 1000

data2[:,3] = data2[:,3] * 1000
data5[:,3] = data5[:,3] * 1000
data10[:,3] = data10[:,3] * 1000

fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(True)

ax.plot(angle, data2[:,1], label=r'Open3D, $\sigma=0.002$',linestyle='--',linewidth=3, color='red')
ax.plot(angle, data5[:,1], label=r'Open3D, $\sigma=0.005$',linestyle='--',linewidth=3, color='blue')
ax.plot(angle, data10[:,1], label=r'Open3D, $\sigma=0.010$',linestyle='--',linewidth=3, color='green')

ax.plot(angle, data2[:,3], label=r'Ours, $\sigma=0.002$',linewidth=3, color='red')
ax.plot(angle, data5[:,3], label=r'Ours, $\sigma=0.005$',linewidth=3, color='blue')
ax.plot(angle, data10[:,3], label=r'Ours, $\sigma=0.010$',linewidth=3, color='green')

ax.set_xlabel(r'Plane Angle/deg')
ax.set_ylabel(r'Estimated Distance Error/mm')

ax.set_xlim(60,150)

plt.legend()
plt.savefig("paper_images/distance_error.png", bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()