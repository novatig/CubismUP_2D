import os
import math
import numpy as np
from pylab import figure, show
import matplotlib.pyplot as plt
import matplotlib

fig = figure(figsize=(16,9))
fig.canvas.set_window_title('Falling Cylinders - Variable Density')

rootDir = '/cluster/scratch_xp/public/cconti/CubismUP/'

p1 = fig.add_subplot(221)
p2 = fig.add_subplot(222)
p3 = fig.add_subplot(223)
p4 = fig.add_subplot(224)

for dirName, subDirList, fileList in os.walk(rootDir):
	for file in fileList:
		if "diagnostics.dat" in file and "Falling_VarDensityDisk_0611" in dirName:
			data = []
			data2 = []
			x = []
			u = []
			v = []
			t = []
			dtdt = []
			xcom = []
			ycom = []
			xcentroid = []
			ycentroid = []

			fig = figure(figsize=(16,9))
			fig.canvas.set_window_title('Falling Cylinders - Variable Density')
			p1 = fig.add_subplot(221)
			p2 = fig.add_subplot(222)
			p3 = fig.add_subplot(223)
			p4 = fig.add_subplot(224)

			data.append(np.genfromtxt(fname=dirName+'/'+file))
			idx = len(data)-1
			dataset = data[idx]
			x.append(dataset[:,1])
			u.append(dataset[:,9])
			v.append(dataset[:,10])
			t.append(dataset[:,11])
			dtdt.append(dataset[:,12])
			xcom.append(dataset[:,13])
			ycom.append(dataset[:,14])
			xcentroid.append(dataset[:,15])
			ycentroid.append(dataset[:,16])
			p1.plot(x[idx],t[idx],label=file)
			p2.plot(x[idx],dtdt[idx],label=file)
			p3.plot(x[idx],u[idx],label=file)
			p3.plot(x[idx],v[idx],label=file)
			p3.legend()
			p4.plot(xcom[idx],ycom[idx],label="CoM", color='r')
			p4.plot(xcentroid[idx],ycentroid[idx],label="centroid",color='b')
			#p4.legend()
			
			tend = 20;
			p1.set_xlim(0,tend)
			p1.set_ylim(0,2*math.pi)
			p2.set_xlim(0,tend)
			p2.set_ylim(-2,2)
			p3.set_xlim(0,tend)
			p3.set_ylim(-10,10)
			p4.set_xlim(0,1)
			p4.set_ylim(0,1)

			p1.set_title("Variable Density Rigid Body")
			p1.set_xlabel('Time')
			p1.set_ylabel('theta')
			
			p2.set_title("Variable Density Rigid Body")
			p2.set_xlabel('Time')
			p2.set_ylabel('dthetadt')

			p3.set_title("Variable Density Rigid Body")
			p3.set_xlabel('Time')
			p3.set_ylabel('Velocity')

			p4.set_title("Variable Density Rigid Body")
			p4.set_xlabel('x')
			p4.set_ylabel('y')

			mng = plt.get_current_fig_manager()
			mng.resize(*mng.window.maxsize())
			fig.savefig(file+".png",dpi=300,transparent=True)
#show()