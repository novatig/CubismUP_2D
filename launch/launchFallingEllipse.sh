# Around Reynolds 1000 :
OPTIONS="-bpdx 16 -bpdy 32 -tdump 1 -nu 0.00004 -tend 8 -poissonType cosine -lambda 1e6"
OBJECTS='ellipse semiAxisX=0.025 semiAxisY=0.125 ypos=0.4 angle=0.0872664626 bFixed=1 rhoS=1.01
'

source launchCommon.sh
