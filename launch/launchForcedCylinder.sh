# Reynolds 1000 :
NU=${NU:-0.000001644736842}
OPTIONS="-bpdx 64 -bpdy 32 -fdump 0.5 -nu ${NU} -CFL 0.1 -iterativePenalization 1 -tend 50 -poissonType freespace "
OBJECTS="disk_radius=0.0625_xpos=0.3_bForced=1_bFixed=1_xvel=0.125"

source launchCommon.sh
