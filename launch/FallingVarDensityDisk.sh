cd ../makefiles
make clean
make config=production poisson=split-fftw bc=mixed precision=double rk2=true particles=false dlm=true constnu=true -j
#
cd ../launch

NAME=Falling_VarDensityDisk_0611_falling_1.25_TopHeavy_bpd16_eps2_nu1e-6_constnu
FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/${NAME}
mkdir ${FOLDER}
cp ../makefiles/simulation ${FOLDER}
cp FallingVarDensityDisk.sh ${FOLDER}
cd ${FOLDER}
export OMP_NUM_THREADS=48
#bsub -n 48 -W 10:00 -o ${NAME} -J ${NAME}
./simulation -file ${FOLDER}/${NAME} -CFL 0.01 -dlm 1 -lambda 1e6 -bpdx 16 -bpdy 16 -radius .025 -tend 10. -rhoS 1.3 -rhoS2 1.2 -angle .2 -ypos .9 -xpos .5 -nu 0.000001 -sim falling -shape diskVarDensity -fdump 1000 -eps 2
cd -