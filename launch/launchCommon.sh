HOST=`hostname`
OS_D=`uname`
echo $RUNNAME
if [[ $# -lt 1 && -z "$RUNNAME" ]] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
RUNNAME=$1
fi

unset LSB_AFFINITY_HOSTFILE #euler cluster
export MV2_ENABLE_AFFINITY=0 #MVAPICH

###################################################################################################
if [ ${OS_D} == 'Darwin' ] ; then

BASEPATH="../runs/"
export OMP_NUM_THREADS=`sysctl -n hw.physicalcpu_max`
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log

###################################################################################################
elif [ ${HOST:0:5} == 'daint' ] ; then

BASEPATH="${SCRATCH}/CUP2D/"
export OMP_NUM_THREADS=12
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}

# did we allocate a node?
srun hostname &> /dev/null
if [[ "$?" -gt "0" ]] ; then
source launchSbatch.sh
else
cd ${FOLDERNAME}
srun -n 1 simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log
fi

###################################################################################################
elif [ ${HOST:0:3} == 'eu-' ] ; then

export LD_LIBRARY_PATH=/cluster/home/novatig/hdf5-1.10.1/gcc_6.3.0_openmpi_2.1/lib/:$LD_LIBRARY_PATH
BASEPATH="$SCRATCH/CubismUP_2D"
NCPUSTR=`lscpu | grep "Core"`
#export OMP_NUM_THREADS=${NCPUSTR: -3}
export OMP_NUM_THREADS=36
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}
if [ "${RUNLOCAL}" == "true" ] ; then
mpirun -n 1 ./simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log
else
bsub -n ${OMP_NUM_THREADS} -J ${RUNNAME} -W 24:00 -R "select[model==XeonGold_6150] span[ptile=${OMP_NUM_THREADS}]" mpirun -n 1 ./simulation ${OPTIONS} -shapes "${OBJECTS}"
fi

###################################################################################################
else

BASEPATH="../runs/"
NCPUSTR=`lscpu | grep "Core"`
export OMP_NUM_THREADS=${NCPUSTR: -3}
echo "Setting nThreads to "${OMP_NUM_THREADS}
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

mpirun -n 1 ./simulation ${OPTIONS} -shapes "${OBJECTS}" | tee out.log

fi

