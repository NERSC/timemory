# TiMemory Components

## Native Components

> Namespace: `tim::component`

These components are available on all operating systems (Windows, macOS, Linux).

| Component Name      | Category | Dependencies | Description                                                                                                  |
| ------------------- | -------- | ------------ | ------------------------------------------------------------------------------------------------------------ |
| **`real_clock`**    | timing   | Native       | Timer for the system's real time (i.e. wall time) clock                                                      |
| **`user_clock`**    | timing   | Native       | records the CPU time spent in user-mode                                                                      |
| **`system_clock`**  | timing   | Native       | records only the CPU time spent in kernel-mode                                                               |
| **`cpu_clock`**     | timing   | Native       | Timer reporting the number of CPU clock cycles / number of cycles per second                                 |
| **`cpu_util`**      | timing   | Native       | Percentage of CPU time vs. wall-clock time                                                                   |
| **`wall_clock`**    | timing   | Native       | Alias to `real_clock` for convenience                                                                        |
| **`virtual_clock`** | timing   | Native       | Alias to `real_clock` since time is a construct of our consciousness                                         |
| **`current_rss`**   | memory   | Native       | The total size of the pages of memory allocated excluding swap                                               |
| **`peak_rss`**      | memory   | Native       | The peak amount of utilized memory (resident-set size) at that point of execution ("high-water" memory mark) |

## POSIX Components

> Namespace: `tim::component`

These components are only available on POSIX systems. The components in the "resource usage" category are provided by POSIX `rusage` (`man getrusage`).

| Component Name                 | Category       | Dependencies | Description                                                                                                                                              |
| ------------------------------ | -------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`process_cpu_clock`**        | timing         | POSIX        | CPU timer that tracks the amount of CPU (in user- or kernel-mode) used by the calling process (excludes child processes)                                 |
| **`thread_cpu_clock`**         | timing         | POSIX        | CPU timer that tracks the amount of CPU (in user- or kernel-mode) used by the calling thread (excludes sibling/child threads)                            |
| **`process_cpu_util`**         | timing         | POSIX        | Percentage of process CPU time (`process_cpu_clock`) vs. wall-clock time                                                                                 |
| **`thread_cpu_util`**          | timing         | POSIX        | Percentage of thread CPU time (`thread_cpu_clock`) vs. `wall_clock`                                                                                      |
| **`monotonic_clock`**          | timing         | POSIX        | Real-clock timer that increments monotonically, unaffected by frequency or time adjustments, that increments while system is asleep                      |
| **`monotonic_raw_clock`**      | timing         | POSIX        | Real-clock timer that increments monotonically, unaffected by frequency or time adjustments                                                              |
| **`data_rss`**                 | memory         | POSIX        | Unshared memory residing the data segment of a process                                                                                                   |
| **`stack_rss`**                | resource usage | POSIX        | Integral value of the amount of unshared memory residing in the stack segment of a process                                                               |
| **`num_io_in`**                | resource usage | POSIX        | Number of times the file system had to perform input                                                                                                     |
| **`num_io_out`**               | resource usage | POSIX        | Number of times the file system had to perform output                                                                                                    |
| **`num_major_page_faults`**    | resource usage | POSIX        | Number of page faults serviced that required I/O activity                                                                                                |
| **`num_minor_page_faults`**    | resource usage | POSIX        | Number of page faults serviced without any I/O activity<sup>[[1]](#fn1)</sup>                                                                            |
| **`num_msg_recv`**             | resource usage | POSIX        | Number of IPC messages received                                                                                                                          |
| **`num_msg_sent`**             | resource usage | POSIX        | Number of IPC messages sent                                                                                                                              |
| **`num_signals`**              | resource usage | POSIX        | Number of signals delivered                                                                                                                              |
| **`num_swap`**                 | resource_usage | POSIX        | Number of swaps out of main memory                                                                                                                       |
| **`priority_context_switch`**  | resource usage | POSIX        | Number of times a context switch resulted due to a higher priority process becoming runnable or bc the current process exceeded its time slice.          |
| **`voluntary_context_switch`** | resource usage | POSIX        | Number of times a context switch resulted due to a process voluntarily giving up the processor before its time slice was completed<sup>[[2]](#fn2)</sup> |

<a name="fn1">[1]</a>: Here I/O activity is avoided by reclaiming a page frame from the list of pages awaiting reallocation

<a name="fn2">[2]</a>: Usually to await availability of a resource

## CUDA Components

> Namespace: `tim::component`

| Component Name | Category | Dependencies | Description                                      |
| -------------- | -------- | ------------ | ------------------------------------------------ |
| **`cuda_event`** | timing   | CUDA runtime | Elapsed time between two points in a CUDA stream |

## Hardware Counter Components

> Namespace: `tim::component`

| Component Name    | Category | Template Specification      | Dependencies | Description                                                                             |
| ----------------- | -------- | --------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| **`papi_tuple`**  | CPU      | `papi_tuple<EventTypes...>` | PAPI         | Variadic list of compile-time specified list of PAPI preset types (e.g. `PAPI_TOT_CYC`) |
| **`papi_array`**  | CPU      | `papi_array<N>`             | PAPI         | Variable set of PAPI counters up to size _N_. Supports native hardware counter types    |
| **`cupti_event`** | GPU      |                             | CUDA, CUPTI  | Provides NVIDIA GPU hardware counters for events and metrics                            |

## Miscellaneous Components

> Namespace: `tim::component`

These components provide tools similar to TiMemory but are commonly used to enable their service/features through the TiMemory interface which enables codes
to provide __*optional*__ support for these tools (via template filtering and/or CMake COMPONENTS).
In the future, there is planned support for features from [LIKWID](https://github.com/RRZE-HPC/likwid).

| Component Name | Category                                                                   | Dependences                                        | Description                                            |
| -------------- | -------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| **`caliper`**  | timing, memory, annotation, MPI, CUDA, PAPI call-stack unwinding, sampling | [Caliper Toolkit](https://github.com/LLNL/Caliper) | Caliper is a flexible application introspection system |

## Roofline Components

> Namespace: `tim::component`

Roofline is a visually intuitive performance model used to bound the performance of various numerical methods and operations running on multicore, manycore, or accelerator processor architectures.
Rather than simply using percent-of-peak estimates, the model can be used to assess the quality of attained performance by combining locality, bandwidth, and different parallelization paradigms
into a single performance figure.
One can examine the resultant Roofline figure in order to determine both the implementation and inherent performance limitations.

More documentation can be found [here](https://docs.nersc.gov/programming/performance-debugging-tools/roofline/).

| Component Name     | Category | Template Specification              | Dependencies | Description                                                     |
| ------------------ | -------- | ----------------------------------- | ------------ | --------------------------------------------------------------- |
| **`cpu_roofline`** | CPU      | `cpu_roofline<Type, EventTypes...>` | PAPI         | Records the rate at which the hardware counters are accumulated |

The roofline components provided by TiMemory execute a workflow during application termination that calculates the theoretical peak for the roofline.
A pre-defined set of algorithms for the theoretical peak are provided but these can be customized assigning a custom function pointer to
`tim::component::cpu_roofline<Type, EventTypes...>::get_finalize_function()`. An example can be found in `timemory/examples/ex-roofline/test_cpu_roofline.cpp`
file of the source code within the `customize_roofline` function.

### Pre-defined Types

> Namespace: `tim::component`

| Component Name              | Underlying Template Specification   | Description                     |
| --------------------------- | ----------------------------------- | ------------------------------- |
| **`cpu_roofline_dp_flops`** | `cpu_roofline<double, PAPI_DP_OPS>` | Rate of double-precision FLOP/s |
| **`cpu_roofline_sp_flops`** | `cpu_roofline<float, PAPI_SP_OPS>`  | Rate of single-precision FLOP/s |

## Variadic Component Wrappers

> Namespace: `tim`

All the components listed above can be used directly but it is recommended to use the variadic wrapper types.
The variadic wrapper types provide bulk operations on all the specified types, e.g. `start()` member function
that calls `start()` on all the specified types.

| Type                       | Description                                                                                        |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| **`component_tuple<...>`** |                                                                                                    |
| **`component_list<...>`**  | Specified types are wrapped into pointers and initially null. Operations applied to non-null types |
| **`auto_tuple<...>`**      | `component_tuple<...>` + auto start/stop via scope                                                 |
| **`auto_list<...>`**       | `component_list<...>` + auto start/stop via scope                                                  |

### Example

```cpp
#include <timemory/timemory.hpp>
#include <thread>
#include <chrono>

using namespace tim::component;
using auto_tuple_t = tim::auto_tuple<real_clock, cpu_clock, cpu_util, peak_rss>;
using auto_list_t  = tim::auto_list<real_clock, cpu_clock, cpu_util, peak_rss>;

void some_func()
{
    auto_tuple_t at("some_func");
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

void another_func()
{
    auto_list_t al("another_func");
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main()
{
    // runtime customization of auto_list_t
    auto_list_t::get_initializer() = [](auto_list_t& al)
    { al.initialize<real_clock, cpu_clock, cpu_util, peak_rss>(); };

    some_func();
    another_func();
}
```
