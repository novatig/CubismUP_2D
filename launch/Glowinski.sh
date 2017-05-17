cd ../makefiles
make clean
make config=production poisson=split-fftw bc=box precision=double rk2=true particles=false dlm=true -j
cd ../launch

FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/GlowinskiLee_031215_box
mkdir ${FOLDER}
cp ../makefiles/simulation ${FOLDER}
cp Glowinski.sh ${FOLDER}

for D in 0.1 1
#10
do
for C in 0.1
#0.01 0.05 0.1
do
for BPD in 16 32
#4 8 16 32 64
do
	NAME=GlowinskiLee_bpdx${BPD}_CFL${C}_DLM${D}_box_
	export OMP_NUM_THREADS=48
#${FOLDER}/simulation -file ${FOLDER}/${NAME}1 -CFL ${C} -bpdx ${BPD} -bpdy $[${BPD}*3] -radius 0.04125 -tend 10. -rhoS 1.25 -ypos .67 -nu 0.001089 -sim falling -DLM 10 -fdump 1 -verbose
#bsub -n 48 -W 10:00 -o ${NAME}1 -J ${NAME}1 ${FOLDER}/simulation -file ${FOLDER}/${NAME}1 -CFL ${C} -bpdx ${BPD} -bpdy $[${BPD}*3] -radius 0.0208333333333333 -tend 5. -rhoS 1.25 -ypos .67 -nu 0.000680413817439772 -sim falling -lambda ${D} -g 9.8
bsub -n 48 -W 10:00 -o ${NAME}1 -J ${NAME}1 ${FOLDER}/simulation -file ${FOLDER}/${NAME}1 -CFL ${C} -bpdx ${BPD} -bpdy $[${BPD}*3] -radius 0.02 -tend 5. -rhoS 1.25 -ypos .67 -nu 0.00064 -sim falling -lambda ${D} -g 9.81
#bsub -n 48 -W 10:00 -o ${NAME}1b -J ${NAME}1b ${FOLDER}/simulation -file ${FOLDER}/${NAME}1b -CFL ${C} -bpdx ${BPD} -bpdy $[${BPD}*3] -radius 0.021 -tend 5. -rhoS 1.25 -ypos .67 -nu 0.0007 -sim falling -dlm ${D} -g 9.8
#bsub -n 48 -W 10:00 -o ${NAME}2 -J ${NAME}2 ${FOLDER}/simulation -file ${FOLDER}/${NAME}2 -CFL ${C} -bpdx ${BPD} -bpdy $[${BPD}*3] -radius 0.02125 -tend 10. -rhoS 1.5 -ypos .67 -nu 0.0001089 -sim falling -DLM 1
done
done
done