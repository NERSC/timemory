# ex-cuda-event

This example executes a set of kernels on CUDA device to measure CUDA events along with other performance metrics such as wall clock time, CPU hardware counters, system clock and so on. The component bundles used for performance measurements are as follows:

```c
using auto_tuple_t = tim::auto_tuple_t<wall_clock, system_clock, cpu_clock, cpu_util,
                                       nvtx_marker, papi_array_t>;
using comp_tuple_t = typename auto_tuple_t::component_type;
using cuda_tuple_t = tim::auto_tuple_t<cuda_event, nvtx_marker>;
using counter_t    = wall_clock;
using ert_data_t   = tim::ert::exec_data<counter_t>;
```

## Build

See [examples](../README.md##Build). Further requires `-DTIMEMORY_USE_CUDA=ON` flag enabled in cmake as well as CUDA installed on the system for this example to build.
