cd ../makefiles
make clean
make config=production poisson=split-fftw bc=mixed precision=double rk2=true particles=false dlm=true -j
cd ../launch
for C in 0.01 0.1
do
	for BPD in 4 8 16 32
	do
		NAME=Sanders_2411_DLM10_bpd${BPD}_CFL${C}
		FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/${NAME}
		mkdir ${FOLDER}
		cp ../makefiles/simulation ${FOLDER}
        cp FallingRisingSanders.sh ${FOLDER}
        cd ${FOLDER}
		export OMP_NUM_THREADS=48
		bsub -n 48 -W 10:00 -o Falling${NAME} -J Falling${NAME} ./simulation -file ${FOLDER}/Falling${NAME} -CFL ${C} -bpdx ${BPD} -bpdy $[${BPD}*3] -radius .025 -tend 2. -rhoS 1.2 -ypos .67 -nu .01 -sim falling -DLM 10
		bsub -n 48 -W 10:00 -o Rising${NAME} -J Rising${NAME} ./simulation -file ${FOLDER}/Rising${NAME} -CFL ${C} -bpdx ${BPD} -bpdy $[${BPD}*3] -radius .025 -tend 2. -rhoS 0.8 -ypos .33 -nu .01 -sim falling -DLM 10
        cd -
	done
done