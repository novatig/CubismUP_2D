cd ../makefiles
make clean
make config=production poisson=split-fftw bc=box precision=double rk2=true particles=false dlm=true -j
cd ../launch

FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/Chen_281115
mkdir ${FOLDER}
cp ../makefiles/simulation ${FOLDER}
cp Glowinski.sh ${FOLDER}

for D in 1 10
do
for C in 0.01 0.1
do
for R in 1.05 1.1 1.15
do
for BPD in 4 8 16
#32 64
do
	NAME=Chen_bpdx${BPD}_CFL${C}_DLM${D}_R${R}
	export OMP_NUM_THREADS=48
	bsub -n 48 -W 10:00 -o ${NAME} -J ${NAME} ${FOLDER}/simulation -file ${FOLDER}/${NAME} -CFL ${C} -bpdx ${BPD} -bpdy $[${BPD}*4] -radius 0.03125 -tend 1 -rhoS ${R} -ypos .625 -nu 0.015625 -sim falling -dlm ${D}
done
done
done
done