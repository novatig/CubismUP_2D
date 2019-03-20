ifeq ($(INTEL_LICENSE_FILE),)
compiler ?= gnu
else
compiler ?= intel
endif
ifeq "$(compiler)" "intel"
CXX ?= icpc
LD = $(CXX)
include make.icc.flags
else
CXX ?= g++
LD = $(CXX)
include make.gcc.flags
endif

NVCC ?= nvcc

hypre ?= false
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
CPPFLAGS+= -D_BS_=$(bs) -D_ALIGNBYTES_=32 -DNTHREADS=$(nthreads)
NVFLAGS+= -code=sm_60 -arch=compute_60

CPPFLAGS += -I../source/ -I../Cubism/source/
NVFLAGS += -I../Cubism/source/

BUILDDIR = .
SRC_DIR = $(BUILDDIR)/../source/
DIRS = $(sort $(dir $(wildcard ../source/*/)))

CUBISM_DIR = $(BUILDDIR)/../Cubism/source/
#OBJECTS = Simulation.o Definitions.o Shape.o ShapeLibrary.o ShapesSimple.o
OBJECTS = \
		Simulation.o SimulationData.o Shape.o ShapeLibrary.o ShapesSimple.o \
		PressureSingle.o PressureIterator.o presRHS_step1.o PoissonSolver.o \
		HYPREdirichlet.o FFTW_freespace.o PutObjectsOnGrid.o UpdateObjects.o \
		advDiff.o advDiff_RK.o BufferedLogger.o Helpers.o BlowFish.o Fish.o \
		FishLibrary.o StefanFish.o CarlingFish.o PressureVarRho.o \
		Penalization.o advDiffGrav.o SmartCylinder.o Glider.o

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

VPATH := $(DIRS) $(CUBISM_DIR)
#  vpath %.cpp $(DIRS)
#  vpath %.h   $(DIRS)

ifneq "$(smarties)" "false"
SMARTIESDIR ?= ../../smarties/source/
CPPFLAGS += -I$(SMARTIESDIR)/Communicators/ -DSMARTIES_APP
endif

ifneq "$(hdf5)" "false"
LIBS+= -lhdf5
CPPFLAGS+= -D_USE_HDF_
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
	$(CC) $(CPPFLAGS) -c -MD $<

CUDA_all.o: CUDA_all.cu
	nvcc $(NVFLAGS) -c $< -o $@

clean:
	rm -f simulation libsimulation.a
	rm -f *.o *.d
