halfH=0.000244140625
x_pos=0.150244140625
y_pos=0.250244140625

# Reynolds 1000 :
OPTIONS="-bpdx 32 -bpdy 16 -tdump 0.1 -nu 0.000016 -tend 0"
OBJECTS="halfDisk_radius=0.04_xpos=${x_pos}_ypos=${y_pos}_bForced=1_bFixed=1_xvel=0.2"

source launchCommon.sh
