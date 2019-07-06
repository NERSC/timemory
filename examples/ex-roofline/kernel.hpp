
#pragma once

#include <cinttypes>
#include <cstdint>
#include <timemory/timemory.hpp>

#ifdef ERT_GPU
extern int gpu_blocks;
extern int gpu_threads;
#endif

#if defined(MACROS)
#    define KERNEL1(a, b, c) ((a) = (b) + (c))
#    define KERNEL2(a, b, c) ((a) = (a) * (b) + (c))
#endif

void
initialize(uint64_t nsize, double* __restrict__ array, double value);

#ifdef ERT_GPU
void
gpuKernel(uint64_t nsize, uint64_t ntrials, double* __restrict__ array,
          int* bytes_per_elem, int* mem_accesses_per_elem);
#else
void
kernel(uint64_t nsize, uint64_t ntrials, double* __restrict__ array, int* bytes_per_elem,
       int* mem_accesses_per_elem);
#endif

int
ert_main(int argc, char** argv);

using namespace tim::component;

using roofline_t  = cpu_roofline<double, PAPI_DP_OPS>;
using auto_roof_t = tim::auto_tuple<roofline_t>;
using comp_roof_t = typename auto_roof_t::component_type;
