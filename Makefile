#NVCC ?= /usr/local/cuda-12.6/bin/nvcc
NVCC ?= ./cuda_12/bin/nvcc

.PHONY: all
all: bin/render.so

.PHONY: clean
clean:
	rm -f bin/render.so

JAX_INCLUDE_DIR=$(shell python -c 'import jax.ffi; print(jax.ffi.include_dir())')

bin/render.so: src/functional_radiance/models/our_model/render.cu
	mkdir -p bin/
	$(NVCC) -O3 --use_fast_math -ftz=true -fmad=true -arch=native -Xcompiler="-O3 -march=native -Wall -shared -fPIC" --std=c++17 -I$(JAX_INCLUDE_DIR) $< -o $@
