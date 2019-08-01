
#pragma once

#include <cinttypes>
#include <cstdint>
#include <timemory/timemory.hpp>

#if defined(MACROS)
#    define KERNEL1(a, b, c) ((a) = (b) + (c))
#    define KERNEL2(a, b, c) ((a) = (a) * (b) + (c))
#endif

void
initialize(uint64_t nsize, double* __restrict__ array, double value);

void
kernel(uint64_t nsize, uint64_t ntrials, double* __restrict__ array,
       int* bytes_per_element, int* memory_accesses_per_element);

int
ert_main(int argc, char** argv);

using namespace tim::component;

using roofline_t  = cpu_roofline<double, PAPI_DP_OPS>;
using auto_roof_t = tim::auto_tuple<roofline_t>;
using comp_roof_t = typename auto_roof_t::component_type;
