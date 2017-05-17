import os
import math
import numpy as np
from pylab import figure, show
import matplotlib.pyplot as plt
import matplotlib

fig = figure(figsize=(16,9))
fig.canvas.set_window_title('Falling Cylinders - Chen')

rootDir = '/cluster/scratch_xp/public/cconti/CubismUP/'

data = []
t = []
v = []
xref = [0, 4]
u1ref = [-0.00438086064679581, -0.00438086064679581]
u2ref = [-0.00876172129359161, -0.00876172129359161]
u3ref = [-0.0131425819403874, -0.0131425819403874]

p = fig.add_subplot(111)

for dirName, subDirList, fileList in os.walk(rootDir):
	if "Chen_281115" in dirName:
		for file in fileList:
			if "diagnostics.dat" in file:
				#and "128" in file:
				data.append(np.genfromtxt(fname=dirName+'/'+file))
				idx = len(data)-1
				dataset = data[idx]
				t.append(dataset[:,1])
				v.append(dataset[:,10])
				p.plot(t[idx],v[idx],label=file)
				p.legend()


p.plot(xref,u1ref)
p.plot(xref,u2ref)
p.plot(xref,u3ref)

p.set_xlim(0,4)
p.set_ylim(-.02,0)

p.set_xlabel('Time')
p.set_ylabel('velocity y')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
#fig.savefig("Chen",dpi=300,transparent=True)
show()