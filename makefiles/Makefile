ifeq ($(INTEL_LICENSE_FILE),)
compiler ?= gnu
else
compiler ?= intel
endif

NVCC ?= nvcc

hypre ?= true
gpu ?= false
hdf5 ?= true
config ?= production
precision ?= double

ifeq ($(shell uname -s), Darwin)
include make.macos
endif

ifneq "$(findstring panda,$(shell hostname))" ""
include make.falcon
endif
ifneq "$(findstring falcon,$(shell hostname))" ""
include make.falcon
endif

ifeq "$(findstring eu-,$(shell hostname))" "eu-"
include make.euler
endif
ifneq "$(findstring euler,$(shell hostname))" ""
include make.euler
endif

ifneq "$(findstring daint,$(shell hostname))" ""
include make.daint
endif

CXX=$(MPICXX)
LD=$(MPICXX)

ifeq "$(shell $(CXX) --version | grep -ci 'icpc')" "0"
ifeq "$(shell $(CXX) --version | grep -ci 'llvm')" "0"
compiler=gnu
include make.gcc.flags
else
compiler=llvm
include make.llvm.flags
endif
else
compiler=intel
include make.icc.flags
endif

bs ?= 32
nthreads ?= 24
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -DNTHREADS=$(nthreads)
NVFLAGS+= -code=sm_60 -arch=compute_60

BUILDDIR = .
CPPFLAGS += -I$(BUILDDIR)/../Cubism/include/
NVFLAGS += -I$(BUILDDIR)/../Cubism/include/
SRC_DIR = $(BUILDDIR)/../source/
DIRS = $(sort $(dir $(wildcard ../source/*/)))

#OBJECTS = Simulation.o Definitions.o Shape.o ShapeLibrary.o ShapesSimple.o
#advDiff_RK.o
OBJECTS = \
		Simulation.o SimulationData.o Shape.o ShapeLibrary.o ShapesSimple.o \
		PressureSingle.o PressureIterator_unif.o PressureIterator_approx.o \
		HYPREdirichlet.o FFTW_freespace.o PutObjectsOnGrid.o UpdateObjects.o \
		advDiff.o BufferedLogger.o Helpers.o Glider.o Fish.o SmartCylinder.o \
		FishLibrary.o ArgumentParser.o PoissonSolver.o advDiffGravStaggered.o \
		advDiffGrav.o BlowFish.o StefanFish.o CarlingFish.o ComputeForces.o \
		HYPREdirichletVarRho.o PressureVarRho_proper.o UpdateObjectsStaggered.o
#Penalization.o PressureIterator.o  PressureVarRho.o PressureSingleStaggered.o

ifneq ($(FFTW_ROOT),)
	# do not overwrite FFTW_ROOT
else ifneq ($(FFTW3_ROOT_DIR),)
	FFTW_ROOT = $(FFTW3_ROOT_DIR)
else ifneq ($(FFTW_DIR),)
	FFTW_ROOT = $(FFTW_DIR)/..
endif
ifneq ($(FFTW_ROOT),)
	LIBS  += -L$(FFTW_ROOT)/lib
	CPPFLAGS  += -I$(FFTW_ROOT)/include
endif
ifeq "$(precision)" "single"
	CPPFLAGS += -D_FLOAT_PRECISION_
	NVFLAGS += -D_FLOAT_PRECISION_
	LIBS += -lfftw3f -lfftw3f_omp
else
	LIBS += -lfftw3  -lfftw3_omp
endif

ifneq "$(gpu)" "false"
LIBS += -lcufft
OBJECTS += CUDA_all.o
CPPFLAGS += -DCUDAFFT
endif

ifneq "$(hypre)" "false"
	ifneq ($(HYPRE_ROOT),)
		LIBS     += -L$(HYPRE_ROOT)/lib
		CPPFLAGS += -I$(HYPRE_ROOT)/include
	endif
LIBS+= -lHYPRE
CPPFLAGS += -DHYPREFFT
endif
CPP_FILES = $(notdir $(OBJECTS:.o=.cpp))

VPATH := $(DIRS) $(BUILDDIR)/../Cubism/src/
#  vpath %.cpp $(DIRS)
#  vpath %.h   $(DIRS)

ifneq "$(GSL_ROOT_DIR)" ""
	CPPFLAGS += -I$(GSL_ROOT_DIR)/include
	LIBS += -L$(GSL_ROOT_DIR)/lib
endif
LIBS += -lgsl -lgslcblas

ifneq "$(hdf)" "false"
	ifneq ($(HDF5_ROOT),)
		# do not overwrite HDF5_ROOT
  else ifneq ($(HDF5ROOT),)
		HDF5_ROOT = $(HDF5ROOT)
	endif
	ifneq ($(HDF5_ROOT),)
		LIBS     += -L$(HDF5_ROOT)/lib
		CPPFLAGS += -I$(HDF5_ROOT)/include
	endif
	LIBS     += -lhdf5
	CPPFLAGS += -DCUBISM_USE_HDF
endif

ifneq ($(ERROR),)
$(error $(ERROR))
endif

all: simulation
.DEFAULT: all;

leadFollow: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

blowfish: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

smartCyl: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

glider: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

simulation: main.o $(OBJECTS)
	$(LD) main.o $(OBJECTS) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(CPPFLAGS) -c -MD $<

CUDA_all.o: CUDA_all.cu
	nvcc $(NVFLAGS) -c $< -o $@

clean:
	rm -f simulation libsimulation.a
	rm -f *.o *.d
