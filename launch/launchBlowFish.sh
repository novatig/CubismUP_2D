#module load gcc

export OMP_NUM_THREADS=4

BASEPATH=../runs/
#BASEPATH=/cluster/scratch/eceva/CubismUP_2D
mkdir -p $BASEPATH
FOLDERNAME=${BASEPATH}/$1

OPTIONS="-bpdx 8 -bpdy 8 -tdump 0.05 -shape blowfish -radius 0.15 -nu 0.002 -tend 10 -rhoS 0.5 -angle 0.01"
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/Cellar/vtk/8.1.1/lib/:$DYLD_LIBRARY_PATH

mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation ${OPTIONS}
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./simulation -tend 10 ${OPTIONS}
