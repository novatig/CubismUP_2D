HOST=`hostname`
OS_D=`uname`
if [ $# -lt 1 ] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
RUNNAME=$1

unset LSB_AFFINITY_HOSTFILE #euler cluster
export MV2_ENABLE_AFFINITY=0 #MVAPICH
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores

if [ ${OS_D} == 'Darwin' ] ;
then

BASEPATH="../runs/"
export OMP_NUM_THREADS=`sysctl -n hw.physicalcpu_max`
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS} -shapes "${OBJECTS}"

elif [ ${HOST:0:5} == 'daint' ] ;
then

BASEPATH="${SCRATCH}/CUP2D/"
export OMP_NUM_THREADS=12
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}

# did we allocate a node?
srun hostname &> /dev/null
if [[ "$?" -gt "0" ]] ;
then
source launchSbatch.sh
else
cd ${FOLDERNAME}
srun -n 1 simulation ${OPTIONS} -shapes "${OBJECTS}"
fi

elif [ ${HOST:0:3} == 'eu-' ] ;
then

BASEPATH="/cluster/scratch/novatig/CubismUP_2D"
NCPUSTR=`lscpu | grep "Core"`
#export OMP_NUM_THREADS=${NCPUSTR: -3}
export OMP_NUM_THREADS=36
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
echo $OMP_NUM_THREADS
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}
./simulation ${OPTIONS} -shapes "${OBJECTS}"

else

BASEPATH="../runs/"
NCPUSTR=`lscpu | grep "Core"`
export OMP_NUM_THREADS=${NCPUSTR: -3}
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS} -shapes "${OBJECTS}"

fi

#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./simulation -tend 10 ${OPTIONS}
