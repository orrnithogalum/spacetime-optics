# bhrt -- build with:  make            (MPI + CUDA, H100 / sm_90 by default)
#                      make NOMPI=1    (single GPU, no MPI)
NVCC    ?= nvcc
MPICXX  ?= mpicxx
CXX     ?= g++
ARCH    ?= sm_90
CUDA_HOME ?= $(abspath $(dir $(shell which $(NVCC)))/..)

BUILD    := build
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter -I$(CUDA_HOME)/include -I../common
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr -I../common -Xptxas -O3 \
            -Xcompiler -O3,-Wall -Wno-deprecated-gpu-targets
LDLIBS   := -L$(CUDA_HOME)/lib64 -lcudart -lz -lpthread

ifeq ($(NOMPI),1)
  HOSTCXX := $(CXX)
  CXXFLAGS += -DBH_NO_MPI
else
  HOSTCXX := $(MPICXX)
endif

OBJS    := $(BUILD)/main.o $(BUILD)/render.o $(BUILD)/config.o $(BUILD)/image_io.o $(BUILD)/colour.o \
           $(BUILD)/scene_build.o $(BUILD)/shot.o
HEADERS := $(wildcard src/*.hpp src/*.cuh src/*.h ../common/*.hpp ../common/*.h)

all: bhrt bh-prep-sky

bhrt: $(OBJS)
	$(HOSTCXX) -o $@ $^ $(LDLIBS)

bh-prep-sky: $(BUILD)/prep_sky.o $(BUILD)/colour.o
	$(CXX) -o $@ $^ -lz -lpthread

$(BUILD)/render.o: src/render.cu $(HEADERS) | $(BUILD)
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
	rm -rf $(BUILD) bhrt bh-prep-sky

.PHONY: all clean
