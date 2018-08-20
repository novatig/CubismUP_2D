# Around Reynolds 1000 :
# Galileo number = sqrt(Delta Rho / Rho_f * g * D) * D / nu = 138
# Rho_s = 1.01 * Rho_f
VISC=0.00007175980525
OPTIONS="-bpdx 32 -bpdy 64 -tdump 0.1 -nu ${VISC} -tend 200"
OBJECTS='disk radius=0.05 ypos=0.25 bFixed=1 rhoS=1.01
'

source launchCommon.sh
