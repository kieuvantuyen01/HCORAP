# For the record:
# $@ The name of the target file (the one before the colon)
# $< The name of the first (or only) prerequisite file (the first one after the colon)
# $^ The names of all the prerequisite files (space separated)
# $* The stem (the bit which matches the % wildcard in the rule definition.

# GCC environment variables
# http://gcc.gnu.org/onlinedocs/gcc-4.8.1/gcc/Environment-Variables.html#Environment-Variables



#============================================
# INSTRUCTIONS
#============================================
# 											\
GCC minimum version 4.7 required.			\
											\
It requires Yices 2.6 to be installed.		\
If yices is not install at the default		\
system paths, two environment variables have\
to be set to specify the path to Yices' 	\
shared binaries	both to the compiler and the\
system:										\
											\
>	export LIBRARY_PATH="path/to/lib"		\
>	export LD_LIBRARY_PATH="path/to/lib"	\
											\
and one to the headers:						\
											\
>	export CPATH="path/to/include"			\
											\
To compile run: 							\
											\
>	make 									\
											\
It can also be compiled without enabling	\
Yices as a backend solver, in which 		\
case is not necessary to specify the path	\
to the binaries and headers. Run:			\
											\
>	make YICES=0							\
											\
											\
When calling a backend solver as an external\
binary, some auxilliary files will 			\
be created, and might not be deleted if the	\
execution is aborted. 						\
By default the temp. files are created in	\
the directory where this application runs.	\
You can set a permanent path by setting 	\
the following variable:						\
											\
> 	export TMPFILESPATH="path/to/tmp"		\
#============================================



CC := g++
SHELL := /bin/bash
HOST := $(shell hostname)

SRCROOT := src
RELEASE_BUILDROOT := build/release
DEBUG_BUILDROOT := build/debug
RELEASE_BINROOT := bin/release
DEBUG_BINROOT := bin/debug

DEBUG := 0
YICES := 0 
GLUCOSE := 0
MINISAT := 0
CUSTOMYICES := 0
CPOPTIMIZER := 0



DIRECTORIES := 	smtapi/src \
			smtapi/src/util \
			smtapi/src/MDD \
			smtapi/src/encoders \
			smtapi/src/optimizers \
			smtapi/src/controllers \
			smtapi/src/solvers \
		encodings \
			encodings/HCORAP \
		proposed/cpp \
			proposed/cpp/encodings \
		parser \
		controllers

SOURCES := $(addprefix smtapi/src/util/, \
	util.cpp \
	errors.cpp \
	bipgraph.cpp \
	disjointset.cpp \
	predgraph.cpp \
)

SOURCES += $(addprefix smtapi/src/optimizers/, \
	optimizer.cpp \
	singlecheck.cpp \
	uboptimizer.cpp \
	buoptimizer.cpp \
	dicooptimizer.cpp \
	nativeoptimizer.cpp \
)


SOURCES += $(addprefix smtapi/src/MDD/, \
	mdd.cpp \
	mddbuilder.cpp \
	amopbmddbuilder.cpp \
	amopbbddbuilder.cpp \
)

SOURCES += $(addprefix smtapi/src/encoders/, \
	encoder.cpp \
	apiencoder.cpp \
	fileencoder.cpp \
	dimacsfileencoder.cpp \
	smtlib2fileencoder.cpp \
)

ifeq ($(YICES),1)
SOURCES += $(addprefix smtapi/src/encoders/, \
	yices2apiencoder.cpp \
)
endif

ifeq ($(GLUCOSE),1)
SOURCES += $(addprefix smtapi/src/encoders/, \
	glucoseapiencoder.cpp)

DIRECTORIES += $(addprefix smtapi/src/solvers/glucose/, \
	core \
	simp \
	utils \
)

SOURCES += $(addprefix smtapi/src/solvers/glucose/, \
	simp/SimpSolver.cc \
	utils/Options.cc \
	utils/System.cc \
	core/Solver.cc \
)

endif


ifeq ($(MINISAT),1)

SOURCES += $(addprefix smtapi/src/encoders/, \
	minisatapiencoder.cpp \
)

DIRECTORIES += $(addprefix smtapi/src/solvers/minisat/, \
	core \
	simp \
	utils \
)

SOURCES += $(addprefix smtapi/src/solvers/minisat/, \
	simp/SimpSolver.cc \
	utils/Options.cc \
	utils/System.cc \
	core/Solver.cc \
)

endif


SOURCES += $(addprefix smtapi/src/controllers/, \
 	solvingarguments.cpp \
 	basiccontroller.cpp \
 	arguments.cpp \
)

SOURCES += $(addprefix smtapi/src/, \
	smtapi.cpp \
	smtformula.cpp \
	encoding.cpp \
)

SOURCES += $(addprefix encodings/HCORAP/,\
	hcorap.cpp \
	HCORAPEncoding.cpp \
	HCORAPNServicesEncoding.cpp\
)

SOURCES += $(addprefix proposed/cpp/encodings/,\
	CardinalityNetwork.cpp \
	ImpliedConstraints.cpp \
	SymmetryBreaking.cpp \
	HCORAPMultiObjectiveEncoding.cpp\
)


SOURCES += $(addprefix parser/, \
 	parser.cpp \
)


# ----------------------------------------------------
# GCC Compiler flags
# ----------------------------------------------------
CFLAGS := -w -std=c++11 -Wall -Wextra

ifeq ($(DEBUG),1)
CFLAGS+= -g -O0 -fbuiltin -fstack-protector-all
else
CFLAGS+= -O3 
endif

ifeq ($(DEBUG),0)
DEFS+= -DNDEBUG
endif

ifeq ($(CPOPTIMIZER),1)
CFLAGS+= -fPIC -fstrict-aliasing -pedantic -fexceptions -frounding-math -Wno-long-long -m64
DEFS+= -DCPOPTIMIZER -DIL_STD -DILOUSEMT -D_REENTRANT -DILM_REENTRANT
LFLAGS+= -lcp -lcplex -lconcert -lpthread -lm -ldl
endif

ifeq ($(CUSTOMYICES),1)
DEFS+= "-DCUSTOMYICES"
endif

ifeq ($(YICES),1)
DEFS+= "-DUSEYICES"
endif

ifeq ($(GLUCOSE),1)
DEFS+= "-DUSEGLUCOSE"
endif

ifeq ($(MINISAT),1)
DEFS+= "-DUSEMINISAT"
endif

ifneq ($(TMPFILESPATH),"")
DEFS+= "-DTMPFILESPATH=\"$(TMPFILESPATH)\""
endif


ifeq ($(DEBUG),1)
BUILDROOT := $(DEBUG_BUILDROOT)
BINROOT := $(DEBUG_BINROOT)
else
BUILDROOT := $(RELEASE_BUILDROOT)
BINROOT := $(RELEASE_BINROOT)
endif



# -----------------------------------------------------
# Links
# -----------------------------------------------------
LFLAGS += -L./$(BUILDROOT)
ifeq ($(YICES),1)
LFLAGS += -lyices
endif



# -----------------------------------------------------
# Include directories
# -----------------------------------------------------
INCLUDES += -I./$(SRCROOT)
INCLUDES += $(addprefix -I./$(SRCROOT)/,$(DIRECTORIES))





OBJS := $(SOURCES:%.cpp=$(BUILDROOT)/%.o)
OBJS := $(OBJS:%.cc=$(BUILDROOT)/%.o)
SOURCES := $(addprefix src/, $(SOURCES))


.PHONY: all hcorap2sat hcorap_multi

.SECONDARY: $(OBJS)


all: hcorap2sat hcorap_multi

clean:
	@rm -rf build
	@rm -rf bin


hcorap2sat: $(BUILDROOT) $(BINROOT) $(addprefix $(BUILDROOT)/, $(DIRECTORIES)) $(BUILDROOT)/hcorap2sat.o $(BINROOT)/hcorap2sat

hcorap_multi: $(BUILDROOT) $(BINROOT) $(addprefix $(BUILDROOT)/, $(DIRECTORIES)) $(BUILDROOT)/hcorap_multi.o $(BINROOT)/hcorap_multi

# Compile the binary by calling the compiler with cflags, lflags, and any libs (if defined) and the list of objects.
$(BINROOT)/%: $(OBJS) $(BUILDROOT)/%.o
	@printf "Linking $@ ... "
	@$(CC) $(OBJS) $(@:$(BINROOT)/%=$(BUILDROOT)/%.o) $(CFLAGS) $(INCLUDES) $(LFLAGS) -o $@
	@echo "DONE"


# Get a .o from a .cpp by calling compiler with cflags and includes (if defined)
$(BUILDROOT)/%.o: $(SRCROOT)/%.cpp
	@printf "Compiling $< ... "
	@$(CC) $(CFLAGS) $(INCLUDES) $(DEFS) -c $< -o $@
	@echo "DONE"

$(BUILDROOT)/%.o: $(SRCROOT)/%.cc
	@printf "Compiling $< ... "
	@$(CC) $(CFLAGS) $(INCLUDES) $(DEFS) -c $< -o $@
	@echo "DONE"

$(BUILDROOT):
	@mkdir -p $@


$(BINROOT):
	@mkdir -p $@


$(addprefix $(BUILDROOT)/, $(DIRECTORIES)): % :
	@mkdir -p $@

