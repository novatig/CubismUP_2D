CC ?= g++
LD ?= g++

nthreads ?= 24
config ?= production
#precision ?= double
precision ?= single
multiphase ?= false
bs ?= 32
smarties ?= false

CPPFLAGS+= -D_BS_=$(bs)

ifeq "$(config)" "production"
	CPPFLAGS += -DNDEBUG -O3 -fstrict-aliasing -march=native -mtune=native -ffast-math -falign-functions=32
else
	CPPFLAGS += -g #-fsanitize=address
	#LIBS += -fsanitize=address
endif

ifeq "$(multiphase)" "true"
	CPPFLAGS += -D_MULTIPHASE_
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
include make.macos
endif

ifneq "$(findstring falcon,$(shell hostname))" ""
include make.falcon
endif

ifneq "$(findstring brutus,$(shell hostname))" ""
include make.brutus
endif

ifneq "$(findstring euler,$(shell hostname))" ""
include make.euler
endif

ifneq "$(findstring daint,$(shell hostname))" ""
include make.daint
endif

CPPFLAGS += -std=c++11 -fopenmp -Wall -D_ALIGNBYTES_=32 #-fpermissive
CPPFLAGS += -DNTHREADS=$(nthreads)
CPPFLAGS += -I../source/ -I../Cubism/source/ #-I$(vtk-inc) -I$(fftw-inc)
CPPFLAGS += -D_USE_FPZIP_  -I../tools/fpzip/inc
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align -Woverloaded-virtual
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow
CPPFLAGS+= -Wwrite-strings -Wno-cpp -Wno-unused-parameter
CPPFLAGS+= -Wno-float-equal
LDFLAGS = $(CPPFLAGS)

ifeq "$(precision)" "single"
    CPPFLAGS += -D_FLOAT_PRECISION_
    LIBS += -lfftw3 -lfftw3f -lfftw3f_threads
endif
ifeq "$(precision)" "double"
    LIBS += -lfftw3 -lfftw3_threads
endif

LIBS += -L../tools/fpzip/lib -lfpzip

BUILDDIR = .
SRC_DIR = $(BUILDDIR)/../source/
CUBISM_DIR = $(BUILDDIR)/../Cubism/source/
OBJECTS = Sim_FSI_Gravity.o Definitions.o

ifeq "$(smarties)" "true"
SMARTIESDIR = ../../smarties/source/Communicators/
#CPPFLAGS += -DRL_MPI_CLIENT
CPPFLAGS += -I$(SMARTIESDIR)
OBJECTS += Communicator.o
VPATH := $(SRC_DIR):$(CUBISM_DIR):$(SMARTIESDIR)
all: blowfish
else
VPATH := $(SRC_DIR):$(CUBISM_DIR)
all: simulation
endif

.DEFAULT: all;

blowfish: main_RL_blowfish.o $(OBJECTS)
	$(LD) main_RL_blowfish.o $(OBJECTS) -o $@ $(LIBS) $(LDFLAGS)

simulation: main.o $(OBJECTS)
	$(LD) main.o $(OBJECTS) -o $@ $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@


clean:
	rm -f simulation blowfish
	rm -f *.o *.d
