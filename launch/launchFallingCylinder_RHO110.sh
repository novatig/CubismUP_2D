# Around Reynolds 175? :
# Galileo number = sqrt(Delta Rho / Rho_f * g * D) * D / nu = 138
# term vel \approx sqrt(pi * R * (\rho*-1) g )
# Rho_s = 1.01 * Rho_f
NU=${NU:-0.0005548658217}
LAMBDA=${LAMBDA:-3e3}
BPDX=${BPDX:-16}
BPDY=${BPDY:-16}

OPTIONS=" -CFL 0.1 -DLM 0 -bpdx $BPDX -bpdy $BPDY -tdump 0.1 -nu ${NU} -extent 2 -tend 500 -poissonType cosine -iterativePenalization 1 -lambda ${LAMBDA}"
OBJECTS='disk radius=0.1 ypos=0.5 bFixed=1 rhoS=1.10
'

source launchCommon.sh
