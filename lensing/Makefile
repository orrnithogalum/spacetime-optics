# lensim -- build with:  make            (MPI + CUDA, H100 / sm_90 by default)
#                        make NOMPI=1    (single GPU, no MPI)
#                        make ARCH=sm_80 (A100) etc.
NVCC    ?= nvcc
MPICXX  ?= mpicxx
CXX     ?= g++
ARCH    ?= sm_90
CUDA_HOME ?= $(abspath $(dir $(shell which $(NVCC)))/..)

BUILD   := build
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter -I$(CUDA_HOME)/include -I../common
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr -I../common \
            -Xptxas -O3 -Xcompiler -O3,-Wall -Wno-deprecated-gpu-targets
LDLIBS   := -L$(CUDA_HOME)/lib64 -lcufft -lcudart -lz -lpthread

ifeq ($(NOMPI),1)
  HOSTCXX := $(CXX)
  CXXFLAGS += -DLENSIM_NO_MPI
else
  HOSTCXX := $(MPICXX)
endif

COMMON := $(BUILD)/config.o $(BUILD)/ephemeris.o $(BUILD)/spectrum.o $(BUILD)/atmosphere_model.o \
          $(BUILD)/image_io.o $(BUILD)/scene_build.o
HEADERS := $(wildcard src/*.hpp src/*.cuh src/*.h ../common/*.hpp ../common/*.h)

all: lensim lensim-prep

lensim: $(BUILD)/main.o $(BUILD)/kernels.o $(COMMON)
	$(HOSTCXX) -o $@ $^ $(LDLIBS)

lensim-prep: $(BUILD)/prep_textures.o $(BUILD)/image_io.o
	$(CXX) -o $@ $^ -lz -lpthread

$(BUILD)/kernels.o: src/kernels.cu $(HEADERS) | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/main.o: src/main.cpp $(HEADERS) | $(BUILD)
	$(HOSTCXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: ../common/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) lensim lensim-prep

.PHONY: all clean
