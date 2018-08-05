CC ?= g++
LD ?= g++

nthreads ?= 24
config ?= production
precision ?= single
cheap ?= false
bs ?= 32
smarties ?= false

CPPFLAGS+= -D_BS_=$(bs)

ifeq "$(config)" "debug"
CPPFLAGS += -g
#CPPFLAGS += -fsanitize=address
#LIBS += -fsanitize=address
else
CPPFLAGS += -DNDEBUG -O3 -fstrict-aliasing -march=native -mtune=native -ffast-math -falign-functions=32
endif

ifeq "$(cheap)" "true"
CPPFLAGS += -DCHEAPFFT
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
include make.macos
endif

ifneq "$(findstring falcon,$(shell hostname))" ""
include make.falcon
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
OBJECTS = Sim_FSI_Gravity.o Definitions.o Shape.o ShapeLibrary.o ShapesSimple.o BlowFish.o Fish.o FishLibrary.o StefanFish.o

VPATH := $(SRC_DIR):$(CUBISM_DIR)

ifneq "$(smarties)" "false"
SMARTIESDIR = ../../smarties/source/Communicators/
CPPFLAGS += -I$(SMARTIESDIR)
OBJECTS += Communicator.o
VPATH := $(VPATH):$(SMARTIESDIR)
endif

LDFLAGS = $(CPPFLAGS)

all: simulation
.DEFAULT: all;

leadFollow: main_RL_2fish.o $(OBJECTS)
	ar rs libsimulation.a $^ #$(LIBS) $(LDFLAGS)
	# TODO: expect user to put SMARTDIR in bashrc
	cp libsimulation.a ../../smarties/makefiles/libsimulation.a
blowfish: main_RL_blowfish.o $(OBJECTS)
	ar rs libsimulation.a $^ #$(LIBS) $(LDFLAGS)
	# TODO: expect user to put SMARTDIR in bashrc
	cp libsimulation.a ../../smarties/makefiles/libsimulation.a

simulation: main.o $(OBJECTS)
	$(LD) main.o $(OBJECTS) -o $@ $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@


clean:
	rm -f simulation libsimulation.a
	rm -f *.o *.d
