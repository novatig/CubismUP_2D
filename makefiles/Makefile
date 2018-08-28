CC ?= g++
LD ?= g++
NVCC?=nvcc

poisson ?= cpu
nthreads ?= 24
config ?= production
precision ?= single
bs ?= 32
smarties ?= false

CPPFLAGS= -D_BS_=$(bs)
NVFLAGS= -code=sm_60 -arch=compute_60

ifeq "$(config)" "debug"
NVFLAGS += -g
CPPFLAGS += -g
#CPPFLAGS += -fsanitize=address
#LIBS += -fsanitize=address
else
NVFLAGS += -DNDEBUG -O3 --use_fast_math
CPPFLAGS += -DNDEBUG -O3 -fstrict-aliasing -march=native -mtune=native -ffast-math -falign-functions=32
endif

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

CPPFLAGS += -std=c++11 -fopenmp -Wall -D_ALIGNBYTES_=32
CPPFLAGS += -DNTHREADS=$(nthreads)
CPPFLAGS += -I../source/ -I../Cubism/source/
NVFLAGS += -I../Cubism/source/
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align -Woverloaded-virtual
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow
CPPFLAGS+= -Wwrite-strings -Wno-unused-parameter #-Wno-cpp
CPPFLAGS+= -Wno-float-equal
LIBS += -fopenmp

ifeq "$(precision)" "single"
    CPPFLAGS += -D_FLOAT_PRECISION_
    NVFLAGS += -D_FLOAT_PRECISION_
    LIBS += -lfftw3f -lfftw3f_threads
endif
ifeq "$(precision)" "double"
    LIBS += -lfftw3  -lfftw3_threads
endif

BUILDDIR = .
SRC_DIR = $(BUILDDIR)/../source/
CUBISM_DIR = $(BUILDDIR)/../Cubism/source/
OBJECTS = Simulation.o Definitions.o Shape.o ShapeLibrary.o ShapesSimple.o BlowFish.o Fish.o FishLibrary.o StefanFish.o BufferedLogger.o

VPATH := $(SRC_DIR):$(CUBISM_DIR)

ifneq "$(smarties)" "false"
SMARTIESDIR ?= ../../smarties/source/
CPPFLAGS += -I$(SMARTIESDIR)/Communicators/
OBJECTS += Communicator.o
VPATH := $(VPATH):$(SMARTIESDIR)/Communicators/
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
	$(CC) $(CPPFLAGS) -c $< -o $@

PoissonSolverScalarCUDA.o: PoissonSolverScalarCUDA.cu
	nvcc $(NVFLAGS) -c $< -o $@

clean:
	rm -f simulation libsimulation.a
	rm -f *.o *.d
