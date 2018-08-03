NTHREADS=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
export OMP_NUM_THREADS=${NTHREADS}

BASEPATH=../runs/
#BASEPATH=/cluster/scratch/novatig/CubismUP_2D
mkdir -p $BASEPATH
FOLDERNAME=${BASEPATH}/$1

OPTIONS="-bpdx 16 -bpdy 4 -tdump 0.1 -nu 0.00008 -tend 10"
OBJECTS='stefanfish L=0.2 xpos=0.25 bFixed=1
stefanfish L=0.2 xpos=0.55 bFixed=1
'

export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/Cellar/vtk/8.1.1/lib/:$DYLD_LIBRARY_PATH

mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS} -shapes "${OBJECTS}"
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./simulation -tend 10 ${OPTIONS}
