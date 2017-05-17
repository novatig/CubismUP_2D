cd ../makefiles
make clean
make config=production poisson=split-fftw bc=mixed precision=double rk2=true particles=false dlm=true -j
#make config=debug poisson=split-fftw bc=mixed precision=double rk2=true particles=false dlm=true -j
cd ../launch

for E in 0 2
do
for C in 0.1 0.01
do
for R in 0.01 0.1 0.9
do
for B in 8 16 32
do
for D in 0.1 0.5 1.0 5.0
do
NAME=Falling_VarDensityDisk_0611_NeutrallyBuoyantTopLight${R}_DLM${D}_bpd${B}_CFL${C}_eps${E}
FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/${NAME}
mkdir ${FOLDER}
cp ../makefiles/simulation ${FOLDER}
cp FallingVarDensityDisk.sh ${FOLDER}
cd ${FOLDER}
R1=$(awk -v r="$R" 'BEGIN{print 1-r}')
R2=$(awk -v r="$R" 'BEGIN{print 1+r}')
export OMP_NUM_THREADS=48
bsub -n 48 -W 10:00 -o ${NAME} -J ${NAME} ./simulation -file ${FOLDER}/${NAME} -CFL ${C} -dlm ${D} -bpdx ${B} -bpdy ${B} -radius .05 -tend 20. -rhoS ${R1} -rhoS2 ${R2} -angle .2 -ypos .5 -xpos .5 -nu 0.001 -sim falling -shape diskVarDensity -tdump .02 -eps ${E}
cd -
done
done
done
done
done


for E in 0 2
do
for C in 0.1 0.01
do
for R in 0.01 0.1 0.9
do
for B in 8 16 32
do
for D in 0.1 0.5 1.0 5.0
do
NAME=Falling_VarDensityDisk_0611_NeutrallyBuoyantTopHeavy${R}_DLM${D}_bpd${B}_CFL${C}_eps${E}
FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/${NAME}
mkdir ${FOLDER}
cp ../makefiles/simulation ${FOLDER}
cp FallingVarDensityDisk.sh ${FOLDER}
cd ${FOLDER}
R1=$(awk -v r="$R" 'BEGIN{print 1+r}')
R2=$(awk -v r="$R" 'BEGIN{print 1-r}')
export OMP_NUM_THREADS=48
bsub -n 48 -W 10:00 -o ${NAME} -J ${NAME} ./simulation -file ${FOLDER}/${NAME} -CFL ${C} -dlm ${D} -bpdx ${B} -bpdy ${B} -radius .05 -tend 20. -rhoS ${R1} -rhoS2 ${R2} -angle .2 -ypos .5 -xpos .5 -nu 0.001 -sim falling -shape diskVarDensity -tdump .02 -eps ${E}
cd -
done
done
done
done
done


for E in 0 2
do
for C in 0.1 0.01
do
for R in 0.01 0.1
do
for B in 8 16 32
do
for D in 0.1 0.5 1.0 5.0
do
NAME=Falling_VarDensityDisk_0611_FallingTopLight${R}_DLM${D}_bpd${B}_CFL${C}_eps${E}
FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/${NAME}
mkdir ${FOLDER}
cp ../makefiles/simulation ${FOLDER}
cp FallingVarDensityDisk.sh ${FOLDER}
cd ${FOLDER}
R1=$(awk -v r="$R" 'BEGIN{print 1.2-r}')
R2=$(awk -v r="$R" 'BEGIN{print 1.2+r}')
export OMP_NUM_THREADS=48
bsub -n 48 -W 10:00 -o ${NAME} -J ${NAME} ./simulation -file ${FOLDER}/${NAME} -CFL ${C} -dlm ${D} -bpdx ${B} -bpdy ${B} -radius .05 -tend 20. -rhoS ${R1} -rhoS2 ${R2} -angle .2 -ypos .8 -xpos .5 -nu 0.01 -sim falling -shape diskVarDensity -tdump .02 -eps ${E}
cd -
done
done
done
done
done


for E in 0 2
do
for C in 0.1 0.01
do
for R in 0.01 0.1
do
for B in 8 16 32
do
for D in 0.1 0.5 1.0 5.0
do
NAME=Falling_VarDensityDisk_0611_FallingTopHeavy${R}_DLM${D}_bpd${B}_CFL${C}_eps${E}
FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/${NAME}
mkdir ${FOLDER}
cp ../makefiles/simulation ${FOLDER}
cp FallingVarDensityDisk.sh ${FOLDER}
cd ${FOLDER}
R1=$(awk -v r="$R" 'BEGIN{print 1.2+r}')
R2=$(awk -v r="$R" 'BEGIN{print 1.2-r}')
export OMP_NUM_THREADS=48
bsub -n 48 -W 10:00 -o ${NAME} -J ${NAME} ./simulation -file ${FOLDER}/${NAME} -CFL ${C} -dlm ${D} -bpdx ${B} -bpdy ${B} -radius .05 -tend 20. -rhoS ${R1} -rhoS2 ${R2} -angle .2 -ypos .8 -xpos .5 -nu 0.01 -sim falling -shape diskVarDensity -tdump .02 -eps ${E}
cd -
done
done
done
done
done