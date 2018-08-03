#module load gcc

export OMP_NUM_THREADS=4

BASEPATH=../runs
mkdir -p $BASEPATH
FOLDERNAME=${BASEPATH}/$1

OPTIONS="-bpdx 16 -bpdy 8 -tdump 0.1 -nu 0.0001 -tend 10"
#OPTIONS+=" -shape diskVarDensity -rhoTop 1.5 -rhoBot 0.5 -rhoS 0.5"
OPTIONS+=" -shape stefanfish -rhoS 1"
OPTIONS+=" -xpos 0.7 -L 0.25 -angle 0 "
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/Cellar/vtk/8.1.1/lib/:$DYLD_LIBRARY_PATH

mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS}
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./simulation -tend 10 ${OPTIONS}
