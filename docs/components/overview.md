# Components

## Querying Available Components

Timemory provides an executable `timemory-avail` which provides a lot of relevant information
about what the current build of timemory supports.

```console
$ timemory-avail

|------------------------------------------|---------------|
| COMPONENT                                  | AVAILABLE       |
| ------------------------------------------ | --------------- |
| caliper                                    | true            |
| cpu_clock                                  | true            |
| cpu_roofline<double>                       | true            |
| cpu_roofline<float, double>                | true            |
| cpu_roofline<float>                        | true            |
| cpu_util                                   | true            |
| cuda_event                                 | false           |
| cuda_profiler                              | false           |
| cupti_activity                             | false           |
| cupti_counters                             | false           |
| data_rss                                   | true            |
| gperf_cpu_profiler                         | true            |
| gperf_heap_profiler                        | true            |
| gpu_roofline<double>                       | false           |
| gpu_roofline<cuda::half2, float, double>   | false           |
| gpu_roofline<cuda::half2>                  | false           |
| gpu_roofline<float>                        | false           |
| likwid_nvmon                               | false           |
| likwid_perfmon                             | true            |
| monotonic_clock                            | true            |
| monotonic_raw_clock                        | true            |
| num_io_in                                  | true            |
| num_io_out                                 | true            |
| num_major_page_faults                      | true            |
| num_minor_page_faults                      | true            |
| num_msg_recv                               | true            |
| num_msg_sent                               | true            |
| num_signals                                | true            |
| num_swap                                   | true            |
| nvtx_marker                                | false           |
| page_rss                                   | true            |
| papi_array<8ul>                            | true            |
| peak_rss                                   | true            |
| priority_context_switch                    | true            |
| process_cpu_clock                          | true            |
| process_cpu_util                           | true            |
| read_bytes                                 | true            |
| stack_rss                                  | true            |
| system_clock                               | true            |
| tau_marker                                 | true            |
| thread_cpu_clock                           | true            |
| thread_cpu_util                            | true            |
| trip_count                                 | true            |
| user_bundle<10101ul, native_tag>           | true            |
| user_bundle<11011ul, native_tag>           | true            |
| user_clock                                 | true            |
| virtual_memory                             | true            |
| voluntary_context_switch                   | true            |
| vtune_event                                | false           |
| vtune_frame                                | false           |
| wall_clock                                 | true            |
| written_bytes                              | true            |
| ------------------------------------------ | --------------- |
```

## [Supported Components](supported.md)

- List of all component wrappers
    - Namespace: `tim`
- List of all supported components
    - Namespace: `tim::component`

## [Native Components](native.md)

- Detailed documentation on components available on all operating systems
- All components here are available directly though the timemory headers and require no external packages

## [System-Dependent Components](os-dependent.md)

- Detailed documentation on components available on certain operating systems
- All components here are available directly though the timemory headers and require no external packages

## [GOTCHA](gotcha.md)

- Support for [GOTCHA](gotcha.md) is provided by the `gotcha` component
    - `gotcha` components in timemory can be used to:
        - Instrument dynamically linked external function calls
        - Provide wholesale replacements of the original function call
- The GOTCHA library is included by timemory as git submodule

## [Caliper Annotations](https://github.com/LLNL/Caliper)

- Support for [Caliper](https://github.com/LLNL/Caliper) is provided through the `caliper` component
- [Caliper Toolkit Documentation](https://software.llnl.gov/Caliper/index.html)
- The Caliper library is included by timemory as a submodule

## LIKWID Markers

- Support for [LIKWID](https://github.com/RRZE-HPC/likwid) marker API is provided through the
  `likwid_perfmon` compnent (CPU) and `likwid_nvmon` component (NVIDIA GPU)
- [LIKWID Wiki](https://github.com/rrze-likwid/likwid/wiki)

## Intel VTune and Advisor

- Support for the Intel ittnotify instrumentation API

## [PAPI](papi.md)

- Detailed documentation on PAPI components
- The PAPI components require access to the PAPI headers and linking to the PAPI library

## [CUDA](cuda.md)

- The `cuda_event` component provides coarse-grained wall-clock timing between two points in
  a CUDA stream
- The `cuda_profiler` component provides handles to start/stop the profilers provided by NVIDIA,
  e.g. Nsight-System, Nsight-Compute, NVprof
- The `nvtx_marker` component provides the instrumentation API for marking ranges in the profilers
  provided by NVIDIA
- The `cupti_activity` component provides fine-grained tracing capabilities
- The `cupti_counters` component provides hardware-counter capabilities

## [Roofline](roofline.md)

- Detailed documentation on roofline components for CPU and GPU

## [Bundling Components](bundling.md)

- Detailed documentation on bundling components with variadic wrappers
- Example
