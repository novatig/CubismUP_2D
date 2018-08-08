BASEPATH=/cluster/scratch/novatig/CubismUP_2D
mkdir -p $BASEPATH

CYL='disk radius=0.05 xpos=0.3 bForced=1 bFixed=1 xvel=0.1
'
FSH='stefanfish L=0.2 xpos=0.3 bFixed=1
'

#loop over stefan and cylinder
# resolution starts at 16 by 8 goes to 512 by 256
# either freespace or periodic
# either single or double prec
for BPDY in "256" "128" "64" "32" "16" "8"; do
for BC in "1" "0"; do
for PREC in "double" "single"; do
BPDX=$((2*${BPDY}))

SHROPT="-bFreeSpace ${BC} -bpdx ${BPDX} -bpdy ${BPDY}"
if [ $BC -eq "0" ] ; then
SHRDIR=BPX${BPDX}_${PREC}_PERIODIC
else
SHRDIR=BPX${BPDX}_${PREC}_FRESPACE
fi
make -C ../makefiles clean
make -C ../makefiles nthreads=18 precision=${PREC} -j

CYLOPT1=" -tdump 1 -nu 0.0001   -tend 8"
CYLOPT2=" -tdump 1 -nu 0.00001  -tend 8"
FSHOPT1=" -tdump 1 -nu 0.00008  -tend 8"
FSHOPT2=" -tdump 1 -nu 0.000008 -tend 8"

DIR1=${BASEPATH}/${SHRDIR}_FshRe0500
DIR2=${BASEPATH}/${SHRDIR}_FshRe5000
DIR3=${BASEPATH}/${SHRDIR}_CylRe0100
DIR4=${BASEPATH}/${SHRDIR}_CylRe1000
mkdir -p ${DIR1}
mkdir -p ${DIR2}
mkdir -p ${DIR3}
mkdir -p ${DIR4}
cp ../makefiles/simulation ${DIR1}
cp ../makefiles/simulation ${DIR2}
cp ../makefiles/simulation ${DIR3}
cp ../makefiles/simulation ${DIR4}
export OMP_NUM_THREADS=18;
bsub -R "select[model==XeonGold_6150]" -n 36 'mpirun -cpu-set 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 ./simulation ${OPTIONS} -shapes "${OBJECTS}" & mpirun -cpu-set 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 ./simulation ${OPTIONS} -shapes "${OBJECTS}" & wait'
done
done
done
