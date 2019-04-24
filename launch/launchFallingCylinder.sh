# Around Reynolds 175? :
# Galileo number = sqrt(Delta Rho / Rho_f * g * D) * D / nu = 138
# term vel \approx sqrt(pi * R * (\rho*-1) g )
# Rho_s = 1.01 * Rho_f
VISC=0.0001754639792
OPTIONS=" -CFL 0.02 -DLM 0 -bpdx 16 -bpdy 16 -tdump 0.1 -nu ${VISC} -tend 200 -poissonType freespace -iterativePenalization 1 -lambda 1e6"
OBJECTS='disk radius=0.1 ypos=0.3 bFixed=1 rhoS=1.01
'

source launchCommon.sh
