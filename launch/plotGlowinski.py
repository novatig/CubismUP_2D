import os
import math
import numpy as np
from pylab import figure, show
import matplotlib.pyplot as plt
import matplotlib

fig = figure(figsize=(16,9))
fig.canvas.set_window_title('Falling Cylinders - Glowinski')

rootDir = '/cluster/scratch_xp/public/cconti/CubismUP/'

data = []
t = []
x = []
y = []
u = []
v = []

p1 = fig.add_subplot(221)
p2 = fig.add_subplot(222)
p3 = fig.add_subplot(223)
p4 = fig.add_subplot(224)

for dirName, subDirList, fileList in os.walk(rootDir):
	if "GlowinskiLee_021215" in dirName:
		for file in fileList:
			if "diagnostics.dat" in file:
				#and "128" in file:
				data.append(np.genfromtxt(fname=dirName+'/'+file))
				idx = len(data)-1
				dataset = data[idx]
				t.append(dataset[:,1])
				x.append(dataset[:,7]*6)
				y.append(dataset[:,8]*6)
				u.append(dataset[:,9])
				v.append(dataset[:,10])
				p1.plot(t[idx],x[idx],label=file)
				p2.plot(t[idx],y[idx],label=file)
				p3.plot(t[idx],u[idx],label=file)
				p4.plot(t[idx],v[idx],label=file)
				p3.legend()


p1.set_xlim(0,4)
p1.set_ylim(0, 2)
p2.set_xlim(0,4)
p2.set_ylim(0, 6)
p3.set_xlim(0,4)
p3.set_ylim(-.01,.01)
p4.set_xlim(0,4)
p4.set_ylim(-.25,0)

#p1.set_title("Namkoong")
p1.set_xlabel('Time')
p1.set_ylabel('position x')

#p2.set_title("Namkoong")
p2.set_xlabel('Time')
p2.set_ylabel('position y')

#p3.set_title("Namkoong")
p3.set_xlabel('Time')
p3.set_ylabel('velocity x')

#p4.set_title("Namkoong")
p4.set_xlabel('Time')
p4.set_ylabel('velocity y')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
fig.savefig("Glowinski",dpi=300,transparent=True)
show()