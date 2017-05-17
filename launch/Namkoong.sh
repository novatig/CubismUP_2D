cd ../makefiles
make clean
make config=production poisson=split-fftw bc=mixed particles=false rk2=true precision=double dlm=true -j
cd ../launch

#unclear how it should be run: ratio 1.01 or ratio 1.014?
for R in 1.01
#1.014
do
for CFL in 0.1
#0.01
do
for BPD in 16 32 64 128 256
do
	BASENAME=FallingCylinder_Namkoong_2711b_DLM10_RK2_CFL${CFL}_bpdy${BPD}_R${R}
	FOLDER=/cluster/scratch_xp/public/cconti/CubismUP/${BASENAME}
	mkdir ${FOLDER}
	cp ../makefiles/simulation ${FOLDER}
    cp Namkoong.sh ${FOLDER}
    cd ${FOLDER}
	export OMP_NUM_THREADS=48;bsub -n 48 -W 10:00 -o ${BASENAME} -J ${BASENAME} ./simulation -file /cluster/scratch_xp/public/cconti/CubismUP/${BASENAME}/${BASENAME} -CFL ${CFL} -bpdx $[${BPD}/8] -bpdy ${BPD} -tend 25. -rhoS ${R} -ypos .85 -nu 0.0000044442057811955 -sim falling -shape disk -radius 0.0078125 -lambda 1e4 -dlm 10
    cd -
done
done
done