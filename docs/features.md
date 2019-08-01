# Features

## Cross-Language Support: C, C++, CUDA, and Python

It is very common for Python projects to implement expensive routines in C or C++. Implementing a TiMemory auto-tuple in any combination of these languages will produce one combined report for all the languages (provided each language links to the same library).
However, this is a feature of TiMemory. TiMemory can be used in standalone C, C++, or Python projects.

## Multithreading

The multithreading overhead is essentially zero.
All TiMemory components use static thread-local singletons of call-graps that are automatically created when a
new thread starts recording a component. The state of the singleton on the master thread is bookmarked and when the
thread is destroyed, the thread-local call-graph is merged back into the master call-graph. Only during the
one-time merge into the master call-graph is a synchronization lock (mutex) utilized.

## MPI

If a project uses MPI, TiMemory will combined the reports from all the MPI ranks when a report is requested.

## PAPI

PAPI counters are available as a component in the same way timing and rusage components are available. If TiMemory
is not compiled with PAPI, it is safe to keep their declaration in the code and their output will be suppressed.

There are two components for PAPI counters detailed [here](/components#hardware-counter-components).

Example:

```c++
using papi_tuple_t = papi_tuple<0, PAPI_TOT_CYC, PAPI_TOT_INS>;
using auto_tuple_t = tim::auto_tuple<real_clock, system_clock, cpu_clock, cpu_util,
                                       peak_rss, current_rss, papi_tuple_t>;

void some_function()
{
    TIMEMORY_AUTO_TUPLE(auto_tuple_t, "");
    // ...
}
```

## CUDA

At this stage, TiMemory implements a `cudaEvent_t` that will record the elapsed time between
two points in the stream pipeline execution. The CUDA documentation for this component
(`tim::component::cuda_event`) can be found
[here](https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/)
and should be constructed on a per-stream basis:

```c++
tim::component::cuda_event evt(stream);
saxpy<<<ngrid, block, 0, stream>>>(N, 1.0f, _dx, _dy);
```

Support for CUPTI (CUDA hardware counters) is in development.

## Plot Generation in Python

The results from TiMemory can be serialized to JSON and the JSON output can be used to produce performance plots
via the standalone `timemory-plotter` or `timemory.plotting` Python module. For roofline analysis,
the `timemory.roofline` module can be used.

### Command-line Tools

UNIX systems provide `timem` executable that works like `time`. On all systems, `pytimem` is provided.

```shell
$ ./timem sleep 5
> [sleep] total execution time :
       5.005e+00 sec real
       0.000e+00 sec sys
       0.000e+00 sec cpu
       0.000e+00 % cpu_util
       6.719e-01 MB peak_rss
             310 minor_page_flts
               1 major_page_flts
               2 vol_cxt_swch
               5 prio_cxt_swch
```
