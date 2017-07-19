#module load gcc

export OMP_NUM_THREADS=4

BASEPATH=../runs/
#BASEPATH=/cluster/scratch/eceva/CubismUP_2D
mkdir -p $BASEPATH
FOLDERNAME=${BASEPATH}/$1

OPTIONS="-bpdx 16 -bpdy 16 -tdump 0.05 -CFL 0.1 -shape blowfish -radius 0.125 -lambda 1e5 -nu 0.002 -ypos 0.5"
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH

mkdir -p ${FOLDERNAME}
cp ../makefiles/simulation ${FOLDERNAME}
cd ${FOLDERNAME}

./simulation -tend 10 ${OPTIONS}
#valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes ./simulation -tend 10 ${OPTIONS}
