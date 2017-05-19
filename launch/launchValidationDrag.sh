#module load gcc

export OMP_NUM_THREADS=48

BASEPATH=/cluster/scratch/eceva/CubismUP_2D
mkdir -p $BASEPATH
FOLDERNAME=${BASEPATH}/$1

OPTIONS="-bpdx 64 -bpdy 128 -tdump 0.05 -CFL 0.1 -radius 0.05 -rhoS 10.00 -lambda 1e5 -nu 0.00002 -ypos 0.85"
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH

mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation -tend 10 ${OPTIONS}
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./simulation -tend 10 ${OPTIONS}


