CC ?= g++
LD=$(CC)

nthreads ?= 48
config ?= production 
bc ?= periodic
precision ?= double
poisson ?= fftw
multiphase ?= false
bs ?= 32
vertexcentered ?= false
particles ?= false
dlm ?= false

constnu ?= false
densitydiff ?= false
#movingframe ?= true
movingframe ?= false
rk2 ?= false

CPPFLAGS+= -w

ifeq "$(bc)" "periodic"
	CPPFLAGS += -D_PERIODIC_
endif
ifeq "$(bc)" "pipe"
	CPPFLAGS += -D_PIPE_
endif
ifeq "$(bc)" "vortex"
	CPPFLAGS += -D_VORTEX_
endif
ifeq "$(bc)" "mixed"
	CPPFLAGS += -D_MIXED_
endif
ifeq "$(bc)" "openbox"
	CPPFLAGS += -D_OPENBOX_
endif
ifeq "$(bc)" "box"
	CPPFLAGS += -D_BOX_
endif

ifeq "$(precision)" "single"
	CPPFLAGS += -D_SP_COMP_
endif

CPPFLAGS+= -D_BS_=$(bs)

ifeq "$(config)" "production"
	CPPFLAGS += -DNDEBUG -O3
else
	CPPFLAGS += -g
endif

ifeq "$(poisson)" "split-fftw"
	CPPFLAGS += -D_SPLIT_
endif

ifeq "$(multiphase)" "true"
	CPPFLAGS += -D_MULTIPHASE_
endif

ifeq "$(vertexcentered)" "true"
	CPPFLAGS += -D_VERTEXCENTERED_
endif

ifeq "$(particles)" "true"
	CPPFLAGS += -D_PARTICLES_
endif

ifeq "$(dlm)" "true"
	CPPFLAGS += -D_DLM_
endif

ifeq "$(constnu)" "true"
	CPPFLAGS += -D_CONSTNU_
endif

ifeq "$(densitydiff)" "true"
	CPPFLAGS += -D_DENSITYDIFF_
endif

ifeq "$(rk2)" "true"
	CPPFLAGS += -D_RK2_
endif

ifeq "$(movingframe)" "true"
CPPFLAGS += -D_MOVING_FRAME_
endif


ifneq "$(findstring brutus,$(shell hostname))" ""
include make.brutus
endif

ifneq "$(findstring euler,$(shell hostname))" ""
include make.euler
endif

CPPFLAGS += -std=c++11 -fopenmp -Wall #-fpermissive
CPPFLAGS += -DNTHREADS=$(nthreads)
CPPFLAGS += -I../source/ -I../Cubism/ #-I$(vtk-inc) -I$(fftw-inc)
CPPFLAGS += -D_USE_FPZIP_  -I../tools/fpzip/inc

LDFLAGS = $(CPPFLAGS)

LIBS += -lstdc++
#	-lm \
#	-L$(vtk-lib) \
#	-lvtkHybrid \
#	-lvtkVolumeRendering \
#	-lvtkRendering \
#	-lvtkIO \
#	-lvtkGenericFiltering \
#	-lvtkGraphics \
#	-lvtkImaging \
#	-lvtkFiltering \
#	-lvtkCommon \
#	-lvtkftgl \
#	-lvtkfreetype \
#	-lvtkDICOMParser \
#	-lvtkexpat \
#	-lvtktiff \
#	-lvtkpng \
#	-lvtkjpeg \
#	-lvtkzlib \
#	-lvtksys -lz \
#	-L$(fftw-lib)

ifeq "$(precision)" "single"
    LIBS += -lfftw3 -lfftw3f -lfftw3f_threads
endif
ifeq "$(precision)" "double"
    LIBS += -lfftw3 -lfftw3_threads
endif

LIBS += -L../tools/fpzip/lib -lfpzip


BUILDDIR = .
SRC_DIR = $(BUILDDIR)/../source/
CUBISM_DIR = $(BUILDDIR)/../Cubism/
OBJECTS = PoissonSolverScalarFFTW.o Sim_FSI_Gravity.o Profiler.o main.o 

VPATH := $(SRC_DIR):$(CUBISM_DIR)

.DEFAULT: all;

all: simulation #test

simulation: $(OBJECTS)
#	echo Linking
#	echo OBJECTS = $(OBJECTS)
#	echo ************************
	$(LD) $(OBJECTS) -o $@ $(LIBS) $(LDFLAGS)
#	echo ************************

test: $(OBJECTS_TEST)
#	echo Linking
#	echo OBJECTS = $(OBJECTS_TEST)
#	echo ************************
	$(LD) $(OBJECTS_TEST) -o $@ $(LIBS) $(LDFLAGS)
#	echo ************************

%.o: $(SRC_DIR)/%.cpp
#	echo PRODUCING $@
	$(CC) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -f simulation test
	rm -f $(OBJECTS) $(OBJECTS_TEST)
