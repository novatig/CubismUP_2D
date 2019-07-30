OPTIONS="-bpdx 64 -bpdy 32 -tdump 0.01 -nu 0.000004 -tend 20 -CFL 0.1 -DLM 1 -poissonType cosine -iterativePenalization 0"

OBJECTS='stefanfish L=0.1 xpos=0.2 ypos=0.25 bFixed=1 pidpos=1
stefanfish L=0.1 xpos=0.3 ypos=0.20 bFixed=1 pidpos=1
stefanfish L=0.1 xpos=0.3 ypos=0.30 bFixed=1 pidpos=1
stefanfish L=0.1 xpos=0.4 ypos=0.25 bFixed=1 pidpos=1
stefanfish L=0.1 xpos=0.4 ypos=0.35 bFixed=1 pidpos=1
stefanfish L=0.1 xpos=0.4 ypos=0.15 bFixed=1 pidpos=1
stefanfish L=0.1 xpos=0.5 ypos=0.20 bFixed=1 pidpos=1
stefanfish L=0.1 xpos=0.5 ypos=0.30 bFixed=1 pidpos=1
stefanfish L=0.1 xpos=0.6 ypos=0.25 bFixed=1 pidpos=1
'

#OBJECTS='stefanfish L=0.1 xpos=0.2 ypos=0.25 bFixed=1 pid=1
#stefanfish L=0.1 xpos=0.3 ypos=0.20 bFixed=1 pid=1
#stefanfish L=0.1 xpos=0.3 ypos=0.30 bFixed=1 pid=1
#stefanfish L=0.1 xpos=0.4 ypos=0.25 bFixed=1 pid=1
#stefanfish L=0.1 xpos=0.4 ypos=0.35 bFixed=1 pid=1
#stefanfish L=0.1 xpos=0.4 ypos=0.15 bFixed=1 pid=1
#stefanfish L=0.1 xpos=0.5 ypos=0.20 bFixed=1 pid=1
#stefanfish L=0.1 xpos=0.5 ypos=0.30 bFixed=1 pid=1
#stefanfish L=0.1 xpos=0.6 ypos=0.25 bFixed=1 pid=1
#'

source launchCommon.sh
