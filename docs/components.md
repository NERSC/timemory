# TiMemory Components

For C++, the following components are in the `tim::component` namespace.

## Native Components

These components are available on all operating systems (Windows, macOS, Linux).

| Component Name | Category | Dependencies | Description                                                                                                  |
| -------------- | -------- | ------------ | ------------------------------------------------------------------------------------------------------------ |
| real_clock     | timing   | Native       | Timer for the system's real time (i.e. wall time) clock                                                      |
| user_clock     | timing   | Native       | records the CPU time spent in user-mode                                                                      |
| system_clock   | timing   | Native       | records only the CPU time spent in kernel-mode                                                               |
| cpu_clock      | timing   | Native       | Timer reporting the number of CPU clock cycles / number of cycles per second                                 |
| cpu_util       | timing   | Native       | Percentage of CPU time vs. wall-clock time                                                                   |
| wall_clock     | timing   | Native       | Alias to `real_clock` for convenience                                                                        |
| virtual_clock  | timing   | Native       | Alias to `real_clock` since time is a construct of our consciousness                                         |
| current_rss    | memory   | Native       | The total size of the pages of memory allocated excluding swap                                               |
| peak_rss       | memory   | Native       | The peak amount of utilized memory (resident-set size) at that point of execution ("high-water" memory mark) |

## POSIX Components

These components are only available on POSIX systems. The components in the "resource usage" category are provided by POSIX `rusage` (`man getrusage`).

| Component Name           | Category       | Dependencies | Description                                                                                                                                              |
| ------------------------ | -------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| process_cpu_clock        | timing         | POSIX        | CPU timer that tracks the amount of CPU (in user- or kernel-mode) used by the calling process (excludes child processes)                                 |
| thread_cpu_clock         | timing         | POSIX        | CPU timer that tracks the amount of CPU (in user- or kernel-mode) used by the calling thread (excludes sibling/child threads)                            |
| process_cpu_util         | timing         | POSIX        | Percentage of process CPU time (`process_cpu_clock`) vs. wall-clock time                                                                                 |
| thread_cpu_util          | timing         | POSIX        | Percentage of thread CPU time (`thread_cpu_clock`) vs. `wall_clock`                                                                                      |
| monotonic_clock          | timing         | POSIX        | Real-clock timer that increments monotonically, unaffected by frequency or time adjustments, that increments while system is asleep                      |
| monotonic_raw_clock      | timing         | POSIX        | Real-clock timer that increments monotonically, unaffected by frequency or time adjustments                                                              |
| data_rss                 | memory         | POSIX        | Unshared memory residing the data segment of a process                                                                                                   |
| stack_rss                | resource usage | POSIX        | Integral value of the amount of unshared memory residing in the stack segment of a process                                                               |
| num_io_in                | resource usage | POSIX        | Number of times the file system had to perform input                                                                                                     |
| num_io_out               | resource usage | POSIX        | Number of times the file system had to perform output                                                                                                    |
| num_major_page_faults    | resource usage | POSIX        | Number of page faults serviced that required I/O activity                                                                                                |
| num_minor_page_faults    | resource usage | POSIX        | Number of page faults serviced without any I/O activity<sup>[[1]](#fn1)</sup>                                                                            |
| num_msg_recv             | resource usage | POSIX        | Number of IPC messages received                                                                                                                          |
| num_msg_sent             | resource usage | POSIX        | Number of IPC messages sent                                                                                                                              |
| num_signals              | resource usage | POSIX        | Number of signals delivered                                                                                                                              |
| num_swap                 | resource_usage | POSIX        | Number of swaps out of main memory                                                                                                                       |
| priority_context_switch  | resource usage | POSIX        | Number of times a context switch resulted due to a higher priority process becoming runnable or bc the current process exceeded its time slice.          |
| voluntary_context_switch | resource usage | POSIX        | Number of times a context switch resulted due to a process voluntarily giving up the processor before its time slice was completed<sup>[[2]](#fn2)</sup> |
| cuda_event               | timing         | CUDA         | Elapsed time between two points in a CUDA stream                                                                                                         |

<a name="fn1">[1]</a>: Here I/O activity is avoided by reclaiming a page frame from the list of pages awaiting reallocation

<a name="fn2">[2]</a>: Usually to await availability of a resource

## CUDA Components

| Component Name | Category | Dependencies | Description                                      |
| -------------- | -------- | ------------ | ------------------------------------------------ |
| cuda_event     | timing   | CUDA runtime | Elapsed time between two points in a CUDA stream |

## Hardware Counter Components

| Component Name | Category | Template Specification      | Dependencies | Description                                                                             |
| -------------- | -------- | --------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| papi_tuple     | CPU      | `papi_tuple<EventTypes...>` | PAPI         | Variadic list of compile-time specified list of PAPI preset types (e.g. `PAPI_TOT_CYC`) |
| papi_array     | CPU      | `papi_array<N>`             | PAPI         | Variable set of PAPI counters up to size _N_. Supports native hardware counter types    |
| cupti_event    | GPU      |                             | CUDA, CUPTI  | Provides NVIDIA GPU hardware counters for events and metrics                            |

## Roofline Components

Roofline is a visually intuitive performance model used to bound the performance of various numerical methods and operations running on multicore, manycore, or accelerator processor architectures.
Rather than simply using percent-of-peak estimates, the model can be used to assess the quality of attained performance by combining locality, bandwidth, and different parallelization paradigms
into a single performance figure.
One can examine the resultant Roofline figure in order to determine both the implementation and inherent performance limitations.

More documentation can be found [here](https://docs.nersc.gov/programming/performance-debugging-tools/roofline/).

| Component Name | Category | Template Specification              | Dependencies | Description                                                     |
| -------------- | -------- | ----------------------------------- | ------------ | --------------------------------------------------------------- |
| cpu_roofline   | CPU      | `cpu_roofline<Type, EventTypes...>` | PAPI         | Records the rate at which the hardware counters are accumulated |

The roofline components provided by TiMemory execute a workflow during application termination that calculates the theoretical peak for the roofline.
A pre-defined set of algorithms for the theoretical peak are provided but these can be customized assigning a custom function pointer to
`tim::component::cpu_roofline<Type, EventTypes...>::get_finalize_function()`. An example can be found in `timemory/examples/ex-roofline/test_cpu_roofline.cpp`
file of the source code within the `customize_roofline` function.

### Pre-defined Types

| Component Name        | Underlying Template Specification   | Description                     |
| --------------------- | ----------------------------------- | ------------------------------- |
| cpu_roofline_dp_flops | `cpu_roofline<double, PAPI_DP_OPS>` | Rate of double-precision FLOP/s |
| cpu_roofline_sp_flops | `cpu_roofline<float, PAPI_SP_OPS>`  | Rate of single-precision FLOP/s |

### Generating Roofline Plot

Currently, some hardware counters cannot be accumulated in a single-pass and as a result, the application must be executed twice to generate a roofline plot:

```bash
export TIMEMORY_JSON_OUTPUT=ON
TIMEMORY_ROOFLINE_MODE=op ./test_cxx_roofline
TIMEMORY_ROOFLINE_MODE=ai ./test_cxx_roofline
python -m timemory.roofline \
    -ai timemory-test-cxx-roofline-output/cpu_roofline_ai.json \
    -op timemory-test-cxx-roofline-output/cpu_roofline_op.json \
    -d
```

- `-ai ...`
    - JSON output containing the hardware counters for arithmetic intensity
- `-op ...`
    - JSON output containing the hardware counters for operations-per-second
- `-d` (optional)
    - interactive display

