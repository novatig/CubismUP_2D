MODIF=impl_zero_long
export BPDX=16; export BPDY=16;
./launchFallingCylinder_RHO101.sh fallingCyl_Re200_D01_RHO101_BPD${BPDX}_$MODIF
./launchFallingCylinder_RHO110.sh fallingCyl_Re200_D01_RHO110_BPD${BPDX}_$MODIF
./launchFallingCylinder_RHO200.sh fallingCyl_Re200_D01_RHO200_BPD${BPDX}_$MODIF
export BPDX=32; export BPDY=32;
./launchFallingCylinder_RHO101.sh fallingCyl_Re200_D01_RHO101_BPD${BPDX}_$MODIF
./launchFallingCylinder_RHO110.sh fallingCyl_Re200_D01_RHO110_BPD${BPDX}_$MODIF
./launchFallingCylinder_RHO200.sh fallingCyl_Re200_D01_RHO200_BPD${BPDX}_$MODIF
export BPDX=64; export BPDY=64;
./launchFallingCylinder_RHO101.sh fallingCyl_Re200_D01_RHO101_BPD${BPDX}_$MODIF
./launchFallingCylinder_RHO110.sh fallingCyl_Re200_D01_RHO110_BPD${BPDX}_$MODIF
./launchFallingCylinder_RHO200.sh fallingCyl_Re200_D01_RHO200_BPD${BPDX}_$MODIF
