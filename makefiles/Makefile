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

poisson ?= cpu
config ?= production
precision ?= single
smarties ?= false

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
include make.macos
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

bs ?= 32
nthreads ?= 24
CPPFLAGS+= -D_BS_=$(bs) -D_ALIGNBYTES_=32 -DNTHREADS=$(nthreads)
NVFLAGS+= -code=sm_60 -arch=compute_60

CPPFLAGS += -I../source/ -I../Cubism/source/
NVFLAGS += -I../Cubism/source/

ifeq "$(precision)" "single"
    CPPFLAGS += -D_FLOAT_PRECISION_
    NVFLAGS += -D_FLOAT_PRECISION_
    LIBS += -lfftw3f -lfftw3f_omp
endif
ifeq "$(precision)" "double"
    LIBS += -lfftw3  -lfftw3_omp
endif

BUILDDIR = .
SRC_DIR = $(BUILDDIR)/../source/
CUBISM_DIR = $(BUILDDIR)/../Cubism/source/
OBJECTS = Simulation.o Definitions.o Shape.o ShapeLibrary.o ShapesSimple.o BlowFish.o Fish.o FishLibrary.o StefanFish.o BufferedLogger.o

VPATH := $(SRC_DIR):$(CUBISM_DIR)

ifneq "$(smarties)" "false"
SMARTIESDIR ?= ../../smarties/source/
CPPFLAGS += -I$(SMARTIESDIR)/Communicators/ -DSMARTIES_APP
endif

hdf5 ?= on
ifeq "$(hdf5)" "on"
LIBS+= -lhdf5
CPPFLAGS+= -D_USE_HDF_
endif

ifeq "$(poisson)" "gpu"
LIBS += -lcufft
OBJECTS += PoissonSolverScalarCUDA.o
CPPFLAGS += -DCUDAFFT
endif

all: simulation
.DEFAULT: all;

leadFollow: main_RL_2fish.o $(OBJECTS)
	ar rs libsimulation.a $^

blowfish: main_RL_blowfish.o $(OBJECTS)
	ar rs libsimulation.a $^

simulation: main.o $(OBJECTS)
	$(LD) main.o $(OBJECTS) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(CPPFLAGS) -c -MD $<

PoissonSolverScalarCUDA.o: PoissonSolverScalarCUDA.cu
	nvcc $(NVFLAGS) -c $< -o $@

clean:
	rm -f simulation libsimulation.a
	rm -f *.o *.d
