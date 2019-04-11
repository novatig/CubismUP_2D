ifeq ($(INTEL_LICENSE_FILE),)
compiler ?= gnu
else
compiler ?= intel
endif
ifeq "$(compiler)" "intel"
CXX ?= icpc
include make.icc.flags
else
CXX ?= g++
include make.gcc.flags
endif

NVCC ?= nvcc

hypre ?= true
gpu ?= false
hdf5 ?= true
config ?= production
precision ?= double
smarties ?= false

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
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

bs ?= 32
nthreads ?= 24
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -DNTHREADS=$(nthreads)
NVFLAGS+= -code=sm_60 -arch=compute_60

BUILDDIR = .
CPPFLAGS += -I../source/ -I$(BUILDDIR)/../Cubism/include/
NVFLAGS += -I$(BUILDDIR)/../Cubism/include/
SRC_DIR = $(BUILDDIR)/../source/
DIRS = $(sort $(dir $(wildcard ../source/*/)))

#OBJECTS = Simulation.o Definitions.o Shape.o ShapeLibrary.o ShapesSimple.o
#advDiff_RK.o
OBJECTS = \
		Simulation.o SimulationData.o Shape.o ShapeLibrary.o ShapesSimple.o \
		PressureSingle.o PressureIterator.o presRHS_step1.o PoissonSolver.o \
		HYPREdirichlet.o FFTW_freespace.o PutObjectsOnGrid.o UpdateObjects.o \
		advDiff.o BufferedLogger.o Helpers.o BlowFish.o Fish.o SmartCylinder.o \
		FishLibrary.o StefanFish.o CarlingFish.o PressureVarRho.o Penalization.o \
		advDiffGrav.o Glider.o PressureIterator_approx.o ArgumentParser.o \
		HYPREdirichletVarRho.o PressureVarRho_proper.o PressureIterator_unif.o


ifneq "$(gpu)" "false"
LIBS += -lcufft
OBJECTS += CUDA_all.o
CPPFLAGS += -DCUDAFFT
endif

ifeq "$(precision)" "single"
    CPPFLAGS += -D_FLOAT_PRECISION_
    NVFLAGS += -D_FLOAT_PRECISION_
    LIBS += -lfftw3f -lfftw3f_omp
else
    LIBS += -lfftw3  -lfftw3_omp
endif

ifneq "$(hypre)" "false"
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

ifneq "$(smarties)" "false"
	SMARTIESDIR ?= ../../smarties/source/
	CPPFLAGS += -I$(SMARTIESDIR)/Communicators/ -DSMARTIES_APP
endif

ifneq "$(hdf5)" "false"
LIBS+= -lhdf5
CPPFLAGS+= -DCUBISM_USE_HDF
endif

all: simulation
.DEFAULT: all;

leadFollow: main_RL_2fish.o $(OBJECTS)
	ar rs libsimulation.a $^

blowfish: main_RL_blowfish.o $(OBJECTS)
	ar rs libsimulation.a $^

smartCyl: main_RL_smartCyl.o $(OBJECTS)
	ar rs libsimulation.a $^

glider: main_RL_glider.o $(OBJECTS)
	ar rs libsimulation.a $^

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
