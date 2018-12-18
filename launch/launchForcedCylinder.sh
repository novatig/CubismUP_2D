# Reynolds 1000 :
OPTIONS="-bpdx 64 -bpdy 32 -fdump 10 -nu 0.00001 -CFL 0.1 -tend 8 -poissonType fftw "
OBJECTS="disk_radius=0.05_xpos=0.3_bForced=1_bFixed=1_xvel=0.1"

source launchCommon.sh
