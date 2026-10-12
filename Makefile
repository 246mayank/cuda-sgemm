# Build on Colab/Kaggle:  make            (T4, sm_75)
#                         make ARCH=sm_60 (Kaggle P100)
ARCH      ?= sm_75
NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
LDLIBS    := -lcublas

KERNELS := $(wildcard src/kernels/*.cu)
HEADERS := src/common.h src/kernels/kernels.h

all: bench bandwidth reduce

# -Xptxas -v prints registers, spills and shared memory per kernel at compile time; the same
# numbers are also written at run time to results/kernels_*.csv by bench.
bench: src/bench.cu $(KERNELS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Xptxas -v -arch=$(ARCH) -DARCH_STR='"$(ARCH)"' -o $@ src/bench.cu $(KERNELS) $(LDLIBS)

bandwidth: src/bandwidth.cu src/common.h
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DARCH_STR='"$(ARCH)"' -o $@ src/bandwidth.cu

clean:
	rm -f bench bandwidth reduce

.PHONY: all clean

reduce: src/reduce.cu src/common.h
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DARCH_STR='"$(ARCH)"' -o $@ src/reduce.cu
