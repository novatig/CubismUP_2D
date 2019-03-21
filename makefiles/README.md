Command used to compiler HYPRE locally with icc:
```
CXXFLAGS="-fast" CFLAGS="-qopenmp -Ofast -ffnalias -fbuiltin -falias -DNDEBUG" LDFLAGS="-qopenmp" ./configure --prefix=${HOME}/hypre/build_intel/ --enable-fortran=no --with-openmp --with-blas --with-lapack
```
