import os
import numpy as np
from numpy import linalg as la

rootDir = '/cluster/home/infk/cconti/CubismUP_2D/data/'


for dirName, subDirList, fileList in os.walk(rootDir):
	for file in fileList:
		if "ConditionNumber" in file and "Test" in file and "bpd16." in file and "2" in file:
			#print(file)
			fname=dirName+'/'+file
			#matrix = np.loadtxt(fname,delimiter=' ',dtype='float')
			f = open(fname)
			matrix = [row.strip().split(' ') for row in f]
			print(file,la.cond(matrix))
			#print(file)