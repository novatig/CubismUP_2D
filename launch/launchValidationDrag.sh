#module load gcc

export OMP_NUM_THREADS=4

BASEPATH=../runs
mkdir -p $BASEPATH
FOLDERNAME=${BASEPATH}/$1

OPTIONS="-bpdx 16 -bpdy 16 -tdump 0.1 -nu 0.0001 -tend 10"
OPTIONS+=" -shape diskVarDensity -rhoTop 1.5 -rhoBot 0.5 -rhoS 0.5"
OPTIONS+=" -ypos 0.5 -radius 0.1 -angle 0.1"
# -bForced 1 -bFixed 1 -yvel 0.1
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/Cellar/vtk/8.1.1/lib/:$DYLD_LIBRARY_PATH

mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS}
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./simulation -tend 10 ${OPTIONS}
