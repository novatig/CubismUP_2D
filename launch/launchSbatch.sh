if [ $# -lt 1 ] ; then
	echo "Usage: ./launch_*.sh RUNNAME"
	exit 1
fi
RUNNAME=$1

cd ${FOLDERNAME}
cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s929
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu

# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
# #SBATCH --constraint=mc
# #SBATCH --mail-user="${MYNAME}@ethz.ch"
# #SBATCH --mail-type=ALL

export OMP_NUM_THREADS=12
export CRAY_CUDA_MPS=1
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores


srun --ntasks 1 --cpus-per-task=12 --threads-per-core=1 --ntasks-per-node=1 ./simulation ${OPTIONS} -shapes "${OBJECTS}"
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
