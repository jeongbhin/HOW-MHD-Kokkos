CXX = g++
CXXFLAGS = -O3 -std=c++20 -fopenmp
CXXFLAGS += -Isrc

KOKKOS_DIR = $(HOME)/kokkos-install

INCLUDES = -I$(KOKKOS_DIR)/include -Isrc
LIBDIRS  = -L$(KOKKOS_DIR)/lib64
LIBS     = -lkokkoscore -lkokkoscontainers -lkokkossimd -ldl

PROJECT = how-mhd
TARGET  = bin/$(PROJECT)

SRC = $(wildcard src/*.cpp) \
      $(wildcard src/problems/*.cpp)

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(SRC) $(INCLUDES) $(LIBDIRS) $(LIBS) -o $(TARGET)

run: $(TARGET)
	cd bin && ./$(PROJECT)

clean:
	rm -f $(TARGET)
