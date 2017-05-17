#module load gcc
export OMP_NUM_THREADS=48

BASEPATH=/cluster/scratch_xp/public/cconti/CubismUP/
BASENAME=OscillatingCylinder_0312_HR_freq5_drag_CFL0.1_DLM_separateF

cd ../makefiles/
make clean
make config=production bc=box poisson=split-fftw precision=double particles=false dlm=true -j
cd ../launch/

for B in 8 16
#16 32 64
do
	OPTIONS=" -bpdx ${B} -bpdy ${B} -CFL 0.1 -radius .05 -rhoS 1. -lambda 1e6 -dlm 1"

	NAME=_$[${B}*32]


	FOLDER=${BASEPATH}${BASENAME}${NAME}
	mkdir ${FOLDER}
	cp ../makefiles/simulation ${FOLDER}
	cp oscillatingCylinder.sh ${FOLDER}
	bsub -n 48 -W 10:00 -o ${BASENAME}${NAME} -J ${BASENAME}${NAME} ${FOLDER}/simulation -file ${FOLDER}/${BASENAME}${NAME} -sim oscillating -Re 100 -kc 5 -tend 10  ${OPTIONS}
#${FOLDER}/simulation -file ${FOLDER}/${BASENAME}${NAME} -sim oscillating -Re 100 -kc 5 -tend 10 -fdump 1 ${OPTIONS}
done