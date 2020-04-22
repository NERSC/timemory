# timemory KokkosP Profiling Tool

[timemory](https://github.com/NERSC/timemory) is a __modular__ performance measurement and analysis library.

Several components are built-in but you can create you own components within this code and provide your
own measurements or analysis.

## Overview of Features

- Several Components are built-in
    - Timing
        - wall-clock, cpu-clock, thread-cpu-clock, cpu-utilization, and more
    - Memory
        - peak resident set size, page resident set size
    - Resource Usage
        - page faults, bytes written, bytes read, context switches, etc.
    - Hardware Counters
        - PAPI for CPUs
        - CUPTI for NVIDIA GPUs
    - [Roofline Performance Model](https://docs.nersc.gov/programming/performance-debugging-tools/roofline/)
        - Requires PAPI for CPUs
        - Requires CUPTI for NVIDIA GPUs
    - TAU instrumentation
    - NVTX instrumentation
    - LIKWID instrumentation
    - VTune instrumentation
    - Caliper instrumentation
    - gperftools instrumentation
- Custom Performance Measurements and Analysis methods can be easily created
    - See [Creating Custom Components](#creating-custom-components)
    - Includes GOTCHA capabilities for non-Kokkos functions


## Relevant Links

- [GitHub](https://github.com/NERSC/timemory)
- [Documentation](https://timemory.readthedocs.io/en/latest/)
- [Doxygen](https://timemory.readthedocs.io/en/latest/doxygen-docs/)

## timemory Install

timemory uses a standard CMake installation system. There are a lot of external (optional) packages so pay
attention to the summary at the end of the CMake configuration if a component in particular is desired.

There are two summaries: a configuration variable summary and an interface library summary. The interface
library summary will provide the easiest method of determining which feature should be added.

### Sample Installation

```console
git clone https://github.com/NERSC/timemory.git timemory
mkdir build-timemory
cd build-timemory
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../timemory
make -j8
make -j8 install
```

### Requirements

- C++ compiler (gcc, clang, Intel)
- CMake v3.11 or higher
  - If you don't have a more recent CMake installation but have `conda`, this will provide a quick installation:
    - `conda create -n cmake -c conda-forge cmake`
    - `source activate cmake`
- [Installation Documentation](https://timemory.readthedocs.io/en/latest/installation/)

## Quick Start for Kokkos

### Building timemory-connector

From the folder of this document, it is recommended to build with CMake. A Makefile is provided but
it will require a lot of configuration. Be sure to use `ccmake` or `cmake-gui` to view all the available
options when configurating the connector. The installation of timemory itself configures several interface
libraries that are not automatically imported into this build -- you may need to toggle some CMake
options in this configuration to actually enable them.

```bash
mkdir build && cd build
cmake -DENABLE_ROOFLINE=ON -Dtimemory_DIR=/opt/timemory ..
make
```

In general, if you do `ENABLE_<FEATURE>` and it cannot be enabled, CMake will fail.

### Using Different Components

Use the command-line tool provided by timemory to find the alias for the tool desired. Use `-d` to get a
description of the tool or `-h` to see all options. Once the desired components have been identified, place
the components in a comma-delimited list in the environment variable `KOKKOS_TIMEMORY_COMPONENTS`, e.g.

```console
export KOKKOS_TIMEMORY_COMPONENTS="wall_clock, peak_rss, cpu_roofline_dp_flops"
```

#### Example

```console
timemory-avail -a
```

| COMPONENT                                  | AVAILABLE | C++ ALIAS / PYTHON ENUMERATION |
| ------------------------------------------ | --------- | ------------------------------ |
| `caliper`                                  | true      | `caliper`                      |
| `cpu_clock`                                | true      | `cpu_clock`                    |
| `cpu_roofline<double>`                     | true      | `cpu_roofline_dp_flops`        |
| `cpu_roofline<float, double>`              | true      | `cpu_roofline_flops`           |
| `cpu_roofline<float>`                      | true      | `cpu_roofline_sp_flops`        |
| `cpu_util`                                 | true      | `cpu_util`                     |
| `cuda_event`                               | false     | `cuda_event`                   |
| `cuda_profiler`                            | false     | `cuda_profiler`                |
| `cupti_activity`                           | false     | `cupti_activity`               |
| `cupti_counters`                           | false     | `cupti_counters`               |
| `data_rss`                                 | true      | `data_rss`                     |
| `gperf_cpu_profiler`                       | false     | `gperf_cpu_profiler`           |
| `gperf_heap_profiler`                      | false     | `gperf_heap_profiler`          |
| `gpu_roofline<double>`                     | false     | `gpu_roofline_dp_flops`        |
| `gpu_roofline<cuda::half2, float, double>` | false     | `gpu_roofline_flops`           |
| `gpu_roofline<cuda::half2>`                | false     | `gpu_roofline_hp_flops`        |
| `gpu_roofline<float>`                      | false     | `gpu_roofline_sp_flops`        |
| `likwid_nvmarker`                          | false     | `likwid_nvmarker`              |
| `likwid_marker`                            | true      | `likwid_marker`                |
| `monotonic_clock`                          | true      | `monotonic_clock`              |
| `monotonic_raw_clock`                      | true      | `monotonic_raw_clock`          |
| `num_io_in`                                | true      | `num_io_in`                    |
| `num_io_out`                               | true      | `num_io_out`                   |
| `num_major_page_faults`                    | true      | `num_major_page_faults`        |
| `num_minor_page_faults`                    | true      | `num_minor_page_faults`        |
| `num_msg_recv`                             | true      | `num_msg_recv`                 |
| `num_msg_sent`                             | true      | `num_msg_sent`                 |
| `num_signals`                              | true      | `num_signals`                  |
| `num_swap`                                 | true      | `num_swap`                     |
| `nvtx_marker`                              | false     | `nvtx_marker`                  |
| `page_rss`                                 | true      | `page_rss`                     |
| `papi_array<8ul>`                          | true      | `papi_array_t`                 |
| `peak_rss`                                 | true      | `peak_rss`                     |
| `priority_context_switch`                  | true      | `priority_context_switch`      |
| `process_cpu_clock`                        | true      | `process_cpu_clock`            |
| `process_cpu_util`                         | true      | `process_cpu_util`             |
| `read_bytes`                               | true      | `read_bytes`                   |
| `stack_rss`                                | true      | `stack_rss`                    |
| `system_clock`                             | true      | `system_clock`                 |
| `tau_marker`                               | true      | `tau_marker`                   |
| `thread_cpu_clock`                         | true      | `thread_cpu_clock`             |
| `thread_cpu_util`                          | true      | `thread_cpu_util`              |
| `trip_count`                               | true      | `trip_count`                   |
| `user_bundle<10101ul, native_tag>`         | true      | `user_tuple_bundle`            |
| `user_bundle<11011ul, native_tag>`         | true      | `user_list_bundle`             |
| `user_clock`                               | true      | `user_clock`                   |
| `virtual_memory`                           | true      | `virtual_memory`               |
| `voluntary_context_switch`                 | true      | `voluntary_context_switch`     |
| `vtune_event`                              | false     | `vtune_event`                  |
| `vtune_frame`                              | false     | `vtune_frame`                  |
| `wall_clock`                               | true      | `wall_clock`                   |
| `written_bytes`                            | true      | `written_bytes`                |

## Run kokkos application with timemory enabled

Before executing the Kokkos application you have to set the environment variable `KOKKOS_PROFILE_LIBRARY` to point to the name of the dynamic library. Also add the library path of PAPI and PAPI connector to `LD_LIBRARY_PATH`.

```console
export KOKKOS_PROFILE_LIBRARY=kp_timemory.so
```

## Run kokkos application with PAPI recording enabled

Internally, timemory uses the `TIMEMORY_PAPI_EVENTS` environment variable for specifying arbitrary events.
However, this library will attempt to read `PAPI_EVENTS` and set `TIMEMORY_PAPI_EVENTS` before the PAPI
component is initialized, if using `PAPI_EVENTS` does not provide the desired events, use `TIMEMORY_PAPI_EVENTS`.

Example enabling (1) total instructions, (2) total cycles, (3) total load/stores

```console
export PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_LST_INS"
export TIMEMORY_PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_LST_INS"
```

## Run kokkos application with Roofline recording enabled

[Roofline Performance Model](https://docs.nersc.gov/programming/performance-debugging-tools/roofline/)

On both the CPU and GPU, calculating the roofline requires two executions of the application.
It is recommended to use the timemory python interface to generate the roofline because
the `timemory.roofline` submodule provides a mode that will handle executing the application
twice and generating the plot. For advanced usage, see the
[timemory Roofline Documentation](https://timemory.readthedocs.io/en/latest/getting_started/roofline/).

```console
export KOKKOS_ROOFLINE=ON
export OMP_NUM_THREADS=4
export KOKKOS_TIMEMORY_COMPONENTS="cpu_roofline_dp_flops"
timemory-roofline -n 4 -t cpu_roofline -- ./sample
```

## Building Sample

```shell
cmake -DBUILD_SAMPLE=ON ..
make -j2
```

## Sample Output

```console
#---------------------------------------------------------------------------#
# KokkosP: TiMemory Connector (sequence is 0, version: 0)
#---------------------------------------------------------------------------#

#--------------------- tim::manager initialized [0][0] ---------------------#

fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073

#---------------------------------------------------------------------------#
KokkosP: Finalization of TiMemory Connector. Complete.
#---------------------------------------------------------------------------#


[peak_rss]|0> Outputting 'docker-desktop_26927/peak_rss.json'...
[peak_rss]|0> Outputting 'docker-desktop_26927/peak_rss.txt'...

>>> sample                              :  214.9 MB peak_rss,  1 laps, depth 0
>>> |_kokkos/dev0/thread_creation       :    0.3 MB peak_rss,  1 laps, depth 1
>>> |_kokkos/dev0/fibonacci             : 2142.2 MB peak_rss, 10 laps, depth 1 (exclusive:  39.8%)
>>>   |_kokkos/dev0/fibonacci_runtime_0 :  209.8 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_1 :  202.7 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_2 :  191.2 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_3 :  175.8 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_4 :  156.3 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_5 :  133.0 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_6 :  105.8 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_7 :   74.7 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_8 :   39.7 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_9 :    0.8 MB peak_rss,  1 laps, depth 2

[wall]|0> Outputting 'docker-desktop_26927/wall.json'...
[wall]|0> Outputting 'docker-desktop_26927/wall.txt'...

>>> sample                              :   21.612 sec wall,  1 laps, depth 0
>>> |_kokkos/dev0/thread_creation       :    0.007 sec wall,  1 laps, depth 1
>>> |_kokkos/dev0/fibonacci             :  200.820 sec wall, 10 laps, depth 1 (exclusive:   2.8%)
>>>   |_kokkos/dev0/fibonacci_runtime_0 :   19.136 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_1 :   19.368 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_2 :   19.497 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_3 :   19.537 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_4 :   19.555 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_5 :   19.621 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_6 :   19.650 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_7 :   19.621 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_8 :   19.621 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_9 :   19.558 sec wall,  1 laps, depth 2

[metadata::manager::finalize]> Outputting 'docker-desktop_26927/metadata.json'...


#---------------------- tim::manager destroyed [0][0] ----------------------#
```

## Creating Custom Components

### Simple Trip Counter

This tool currently uses `component::user_bundle<size_t, T>` within a variadic collection of tools
`component_tuple<T...>`. The `user_bundle` is relatively heavyweight component that is capable
of activating any of the built-in components to timemory. If new analysis is required, define
the component, insert (or replace) the new component inside the template parameters of
`component_tuple`, recompile, and use the tool.

```cpp
using profile_entry_t  = tim::component_tuple<KokkosUserBundle>;
```

with the definition of your component and either add it to the template parameters
of `profile_entry_t` or make it the exclusive template parameter:

```cpp
namespace tim { namespace component {

struct MyTripCount : public base<trip_count, int64_t>
{
    using value_type = int64_t;
    using this_type  = MyTripCount;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "trip_count"; }
    static std::string description() { return "trip counts"; }
    static value_type  record() { return 1; }

    value_type get() const { return accum; }
    value_type get_display() const { return get(); }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        accum += value;
        set_stopped();
    }

}; } }

using profile_entry_t  = tim::component_tuple<MyTripCount>;
```

### Simple GOTCHA replacement for exp

Components can also be used to implement GOTCHAs to either
instrument the original function call or as a whole-sale replacement
for the function call.

```cpp
namespace tim { namespace component {

struct exp_intercept : public base<exp_intercept, void>
{
    // replaces 'double exp(double)' with 'double expf(float)'
    double operator()(double val)
    {
        return expf(static_cast<float>(val));
    }

}; } }

// for component to implement operator() in lieu of wrapper, must define empty component tuple
using exp_dummy_t      = tim::component_tuple<>;
using exp_gotcha_t     = tim::component::gotcha<1, exp_dummy_t, exp_intercept>;

extern "C" void
kokkosp_init_library(...)
{
    // ...
    // add below macro to generate and activate the gotcha
    TIMEMORY_C_GOTCHA(exp_gotcha_t, 0, exp);
}
```

### Simple GOTCHA timer for cos

```cpp
// for component to implement operator() in lieu of wrapper, must define empty component tuple
using tim::component::wall_clock;
using cos_timer_t      = tim::component_tuple<wall_clock>;
using cos_gotcha_t     = tim::component::gotcha<1, cos_timer_t>;

extern "C" void
kokkosp_init_library(...)
{
    // ...
    // add below macro to generate and activate the gotcha
    TIMEMORY_C_GOTCHA(cos_gotcha_t, 0, cos);
}
```
