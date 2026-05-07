# ================================================================
# HOW-MHD Makefile
# MPI + Kokkos/OpenMP hybrid build
# ================================================================

# ------------------------------------------------
# User settings
# ------------------------------------------------
KOKKOS_ROOT ?= $(HOME)/codes/kokkos-install

BIN_DIR   := bin
BUILD_DIR := build

MPI_EXE := $(BIN_DIR)/how_mhd_mpi
SER_EXE := $(BIN_DIR)/how_mhd

# ------------------------------------------------
# Source files
# ------------------------------------------------
SRC = $(wildcard src/*.cpp) \
      $(wildcard src/problems/*.cpp)

OBJ_MPI = $(patsubst %.cpp,$(BUILD_DIR)/mpi/%.o,$(SRC))
OBJ_SER = $(patsubst %.cpp,$(BUILD_DIR)/ser/%.o,$(SRC))

# ------------------------------------------------
# Compilers
# ------------------------------------------------
CXX_MPI ?= mpicxx
CXX_SER ?= g++

# ------------------------------------------------
# Flags
# ------------------------------------------------
CXXFLAGS_COMMON = -O3 -std=c++20 -Wall -Wextra -pedantic
CXXFLAGS_OPENMP = -fopenmp

INCLUDES = -I$(KOKKOS_ROOT)/include \
           -Isrc \
           -Isrc/problems

# Kokkos may install libraries in lib64 or lib.
KOKKOS_LIBDIR ?= $(shell \
	if [ -d "$(KOKKOS_ROOT)/lib64" ]; then echo "$(KOKKOS_ROOT)/lib64"; \
	else echo "$(KOKKOS_ROOT)/lib"; fi)

KOKKOS_LIBS = -L$(KOKKOS_LIBDIR) \
              -lkokkoscore \
              -lkokkoscontainers \
              -ldl \
              -lpthread

# ------------------------------------------------
# Default target
# ------------------------------------------------
.PHONY: all
all: mpi

# ------------------------------------------------
# MPI build
# ------------------------------------------------
.PHONY: mpi
mpi: $(MPI_EXE)

$(MPI_EXE): $(OBJ_MPI)
	@mkdir -p $(BIN_DIR)
	$(CXX_MPI) $(CXXFLAGS_COMMON) $(CXXFLAGS_OPENMP) $^ -o $@ $(KOKKOS_LIBS)

$(BUILD_DIR)/mpi/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX_MPI) $(CXXFLAGS_COMMON) $(CXXFLAGS_OPENMP) -DUSE_MPI $(INCLUDES) -c $< -o $@

# ------------------------------------------------
# Serial / Kokkos-only build
# ------------------------------------------------
.PHONY: serial
serial: $(SER_EXE)

$(SER_EXE): $(OBJ_SER)
	@mkdir -p $(BIN_DIR)
	$(CXX_SER) $(CXXFLAGS_COMMON) $(CXXFLAGS_OPENMP) $^ -o $@ $(KOKKOS_LIBS)

$(BUILD_DIR)/ser/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX_SER) $(CXXFLAGS_COMMON) $(CXXFLAGS_OPENMP) $(INCLUDES) -c $< -o $@

# ------------------------------------------------
# Utility targets
# ------------------------------------------------
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)/how_mhd_mpi $(BIN_DIR)/how_mhd

.PHONY: distclean
distclean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: info
info:
	@echo "KOKKOS_ROOT   = $(KOKKOS_ROOT)"
	@echo "KOKKOS_LIBDIR = $(KOKKOS_LIBDIR)"
	@echo "SRC           = $(SRC)"
	@echo "MPI_EXE       = $(MPI_EXE)"
	@echo "SER_EXE       = $(SER_EXE)"
