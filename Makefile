# Build on Colab/Kaggle:  make            (T4, sm_75)
#                         make ARCH=sm_60 (Kaggle P100)
ARCH      ?= sm_75
NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
LDLIBS    := -lcublas

KERNELS := $(wildcard src/kernels/*.cu)
HEADERS := src/common.h src/kernels/kernels.h

bench: src/bench.cu $(KERNELS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DARCH_STR='"$(ARCH)"' -o $@ src/bench.cu $(KERNELS) $(LDLIBS)

clean:
	rm -f bench bandwidth

.PHONY: clean
