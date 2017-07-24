import numpy as np
import sys
binary_input = sys.argv[1]
csv_output   = sys.argv[2]

d = np.fromfile(binary_input, dtype=np.float32)
length = d.size/11
print(length)
d = d.reshape([length, 11])
np.savetxt(csv_output, d, fmt='%.6e', delimiter=',', newline='\n')
