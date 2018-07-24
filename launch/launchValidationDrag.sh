#module load gcc

export OMP_NUM_THREADS=4

BASEPATH=../runs
mkdir -p $BASEPATH
FOLDERNAME=${BASEPATH}/$1

OPTIONS="-bpdx 8 -bpdy 16 -tdump 0.1 -nu 0.0001 -tend 10"
OPTIONS+=" -ypos 0.25 -radius 0.05 -bForced 1 -bFixed 1 -yvel 0.1"
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/Cellar/vtk/8.1.1/lib/:$DYLD_LIBRARY_PATH

mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS}
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./simulation -tend 10 ${OPTIONS}
