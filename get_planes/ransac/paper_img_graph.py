import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib 
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 
font = {'size'   : 10}

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

data2[:,1] = data2[:,1] * 1000
data5[:,1] = data5[:,1] * 1000
data10[:,1] = data10[:,1] * 1000

data2[:,3] = data2[:,3] * 1000
data5[:,3] = data5[:,3] * 1000
data10[:,3] = data10[:,3] * 1000

fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(True)

line1 = ax.plot(angle, data2[:,0], label=r'Angle Error, $\sigma=0.002$',linewidth=3, color='red')
line2 = ax.plot(angle, data5[:,0], label=r'Angle Error, $\sigma=0.005$',linewidth=3, color='blue')
line3 = ax.plot(angle, data10[:,0], label=r'Angle Error, $\sigma=0.010$',linewidth=3, color='green')

ax.set_ylabel(r'Open3D Estimated Angle Error/deg')
ax.set_xlabel(r'Plane Angle/deg')
ax.set_xlim(60,150)
ax.set_ylim(0,20)
ax.legend(loc='upper left')

ax2 = ax.twinx()

line4 = ax2.plot(angle, data2[:,1], label=r'Distance Error, $\sigma=0.002$',linestyle='--',linewidth=3, color='red')
line5 = ax2.plot(angle, data5[:,1], label=r'Distance Error, $\sigma=0.005$',linestyle='--',linewidth=3, color='blue')
line6 = ax2.plot(angle, data10[:,1], label=r'Distance Error, $\sigma=0.010$',linestyle='--',linewidth=3, color='green')

ax2.set_ylabel(r'Open3D Estimated Distance Error/mm')
ax2.set_ylim(0,3)
ax2.legend(loc='upper right')

#lns = line1+line2+line3+line4+line5+line6
#labs = [l.get_label() for l in lns]
#ax.legend(lns, labs, loc=0)
plt.tight_layout()
plt.savefig("paper_images/open3d_graph.png", bbox_inches='tight', pad_inches=0, transparent=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(True)

line1 = ax.plot(angle, data2[:,2], label=r'Angle Error, $\sigma=0.002$',linewidth=3, color='red')
line2 = ax.plot(angle, data5[:,2], label=r'Angle Error, $\sigma=0.005$',linewidth=3, color='blue')
line3 = ax.plot(angle, data10[:,2], label=r'Angle Error, $\sigma=0.010$',linewidth=3, color='green')

ax.set_ylabel(r'Ours Estimated Angle Error/deg')
ax.set_xlabel(r'Plane Angle/deg')
ax.set_xlim(60,150)
ax.set_ylim(0,20)
ax.legend(loc='upper left')

ax2 = ax.twinx()

line4 = ax2.plot(angle, data2[:,3], label=r'Distance Error, $\sigma=0.002$',linestyle='--',linewidth=3, color='red')
line5 = ax2.plot(angle, data5[:,3], label=r'Distance Error, $\sigma=0.005$',linestyle='--',linewidth=3, color='blue')
line6 = ax2.plot(angle, data10[:,3], label=r'Distance Error, $\sigma=0.010$',linestyle='--',linewidth=3, color='green')

ax2.set_ylabel(r'Ours Estimated Distance Error/mm')
ax2.set_ylim(0,3)
ax2.legend(loc='upper right')

#lns = line1+line2+line3+line4+line5+line6
#labs = [l.get_label() for l in lns]
#ax.legend(lns, labs, loc=0)
plt.tight_layout()

plt.legend()
plt.savefig("paper_images/ours_graph.png", bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()