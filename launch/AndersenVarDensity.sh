cd ../makefiles
make clean
make config=production poisson=split-fftw bc=mixed precision=single particles=false rk2=true dlm=true -j
cd ../launch

for P in 0.5
do
for CFL in 0.1
#0.05 0.1 0.2
do
for BPD in 32
#32 64 128
do
	BASENAME=Andersen_0811_50FPS_F_P${P}_CFL${CFL}_bpd${BPD}
	FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/${BASENAME}
    mkdir ${FOLDER}
    cp ../makefiles/simulation ${FOLDER}/simulation_T
    cp ../makefiles/simulation ${FOLDER}/simulation_F
    cp Andersen.sh ${FOLDER}
cd ${FOLDER}
export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME}_T_1 -J ${BASENAME}_T_1 ./simulation_T -file ${FOLDER}/${BASENAME}_T_1 -CFL ${CFL} -bpdx ${BPD} -bpdy ${BPD} -tend 20. -rhoS 4.0 -rhoS2 3.9 -ypos .9 -xpos .5 -nu 0.0000190757887517147 -sim falling -shape ellipseVarDensity -semiAxisX 0.025 -semiAxisY 0.003125 -angle 1.78 -tdump 0.02 -dlm 1
export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME}_F_1 -J ${BASENAME}_F_1 ./simulation_F -file ${FOLDER}/${BASENAME}_F_1 -CFL ${CFL} -bpdx ${BPD} -bpdy ${BPD} -tend 20. -rhoS 1.25 -rhoS2 1.2 -ypos .9 -xpos .5 -nu 0.0000190757887517147 -sim falling -shape ellipseVarDensity -semiAxisX 0.025 -semiAxisY 0.003125 -angle 1.78 -tdump 0.02 -dlm 1
export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME}_T_2 -J ${BASENAME}_T_2 ./simulation_T -file ${FOLDER}/${BASENAME}_T_2 -CFL ${CFL} -bpdx ${BPD} -bpdy ${BPD} -tend 20. -rhoS 4.0 -rhoS2 3.9 -ypos .9 -xpos .5 -nu 0.0000190757887517147 -sim falling -shape ellipseVarDensity -semiAxisX 0.003125 -semiAxisY 0.025 -angle 1.78 -tdump 0.02 -dlm 1
export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME}_F_2 -J ${BASENAME}_F_2 ./simulation_F -file ${FOLDER}/${BASENAME}_F_2 -CFL ${CFL} -bpdx ${BPD} -bpdy ${BPD} -tend 20. -rhoS 1.25 -rhoS2 1.2 -ypos .9 -xpos .5 -nu 0.0000190757887517147 -sim falling -shape ellipseVarDensity -semiAxisX 0.003125 -semiAxisY 0.025 -angle 1.78 -tdump 0.02 -dlm 1
export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME}_T_3 -J ${BASENAME}_T_3 ./simulation_T -file ${FOLDER}/${BASENAME}_T_3 -CFL ${CFL} -bpdx ${BPD} -bpdy ${BPD} -tend 20. -rhoS 3.9 -rhoS2 4.0 -ypos .9 -xpos .5 -nu 0.0000190757887517147 -sim falling -shape ellipseVarDensity -semiAxisX 0.025 -semiAxisY 0.003125 -angle 1.78 -tdump 0.02 -dlm 1
export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME}_F_3 -J ${BASENAME}_F_3 ./simulation_F -file ${FOLDER}/${BASENAME}_F_3 -CFL ${CFL} -bpdx ${BPD} -bpdy ${BPD} -tend 20. -rhoS 1.2 -rhoS2 1.25 -ypos .9 -xpos .5 -nu 0.0000190757887517147 -sim falling -shape ellipseVarDensity -semiAxisX 0.025 -semiAxisY 0.003125 -angle 1.78 -tdump 0.02 -dlm 1
export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME}_T_4 -J ${BASENAME}_T_4 ./simulation_T -file ${FOLDER}/${BASENAME}_T_4 -CFL ${CFL} -bpdx ${BPD} -bpdy ${BPD} -tend 20. -rhoS 3.9 -rhoS2 4.0 -ypos .9 -xpos .5 -nu 0.0000190757887517147 -sim falling -shape ellipseVarDensity -semiAxisX 0.003125 -semiAxisY 0.025 -angle 1.78 -tdump 0.02 -dlm 1
export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME}_F_4 -J ${BASENAME}_F_4 ./simulation_F -file ${FOLDER}/${BASENAME}_F_4 -CFL ${CFL} -bpdx ${BPD} -bpdy ${BPD} -tend 20. -rhoS 1.2 -rhoS2 1.25 -ypos .9 -xpos .5 -nu 0.0000190757887517147 -sim falling -shape ellipseVarDensity -semiAxisX 0.003125 -semiAxisY 0.025 -angle 1.78 -tdump 0.02 -dlm 1
    cd -
done
done
done