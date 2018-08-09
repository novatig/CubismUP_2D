module purge
module load modules new gcc/5.2.0 fftw/3.3.4 fftw_sp/3.3.4 hdf5/1.8.12 mvapich2/2.2

BASEPATH=/cluster/scratch/novatig/CubismUP_2D
mkdir -p $BASEPATH

CYL="disk_radius=0.05_xpos=0.3_bForced=1_bFixed=1_xvel=0.1"
FSH="stefanfish_L=0.2_xpos=0.3_bFixed=1"

#loop over stefan and cylinder. resolution starts at 16 by 8 goes to 512 by 256
# either freespace or periodic,  either single or double prec
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

DIR1=${SHRDIR}_CylRe0100
DIR2=${SHRDIR}_CylRe1000
DIR3=${SHRDIR}_FshRe0500
DIR4=${SHRDIR}_FshRe5000
CYLOPT1=" ${SHROPT} -tdump 1 -nu 0.0001   -tend 8 -file ./${DIR1}"
CYLOPT2=" ${SHROPT} -tdump 1 -nu 0.00001  -tend 8 -file ./${DIR2}"
FSHOPT1=" ${SHROPT} -tdump 1 -nu 0.00008  -tend 8 -file ./${DIR3}"
FSHOPT2=" ${SHROPT} -tdump 1 -nu 0.000008 -tend 8 -file ./${DIR4}"
mkdir -p ${BASEPATH}/${DIR1}
mkdir -p ${BASEPATH}/${DIR2}
mkdir -p ${BASEPATH}/${DIR3}
mkdir -p ${BASEPATH}/${DIR4}
cp ../makefiles/simulation ${BASEPATH}/${DIR1}
cp ../makefiles/simulation ${BASEPATH}/${DIR2}
cp ../makefiles/simulation ${BASEPATH}/${DIR3}
cp ../makefiles/simulation ${BASEPATH}/${DIR4}
cd ${BASEPATH}
export OMP_PLACES=cores
export OMP_PROC_BIND=CLOSE
export OMP_NUM_THREADS=18
unset LSB_AFFINITY_HOSTFILE
bsub -J ${SHRDIR}LoRe -R "select[model==XeonGold_6150]" -n 36 " mpirun -np 1 -bind-to user:0+1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17 ./${DIR1}/simulation ${CYLOPT1} -shapes ${CYL} & mpirun -np 1 -bind-to user:18+19+20+21+22+23+24+25+26+27+28+29+30+31+32+33+34+35 ./${DIR3}/simulation ${FSHOPT1} -shapes ${FSH} & wait "

bsub -J ${SHRDIR}HiRe -R "select[model==XeonGold_6150]" -n 36 " mpirun -np 1 -bind-to user:0+1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17 ./${DIR2}/simulation ${CYLOPT2} -shapes ${CYL} & mpirun -np 1 -bind-to user:18+19+20+21+22+23+24+25+26+27+28+29+30+31+32+33+34+35 ./${DIR4}/simulation ${FSHOPT2} -shapes ${FSH} & wait "
cd -

done
done
done
