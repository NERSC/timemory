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
| **`trip_count`**    | counting | Native       | Recording the number of trips through a section of code                                                      |

## System-Dependent Components

> Namespace: `tim::component`

These components are only available on certain operating systems. In general, all of these components are available for POSIX systems.
The components in the "resource usage" category are provided by POSIX `rusage` (`man getrusage`).

| Component Name                 | Category       | Dependencies | Description                                                                                                                                                                                    |
| ------------------------------ | -------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`read_bytes`**               | I/O            | Linux/macOS  | Attempt to count the number of bytes which this process really did cause to be fetched from the storage layer. Done at the submit_bio() level, so it is accurate for block-backed filesystems. |
| **`written_bytes`**            | I/O            | Linux/macOS  | Attempt to count the number of bytes which this process caused to be sent to the storage layer. This is done at page-dirtying time.                                                            |
| **`process_cpu_clock`**        | timing         | POSIX        | CPU timer that tracks the amount of CPU (in user- or kernel-mode) used by the calling process (excludes child processes)                                                                       |
| **`thread_cpu_clock`**         | timing         | POSIX        | CPU timer that tracks the amount of CPU (in user- or kernel-mode) used by the calling thread (excludes sibling/child threads)                                                                  |
| **`process_cpu_util`**         | timing         | POSIX        | Percentage of process CPU time (`process_cpu_clock`) vs. wall-clock time                                                                                                                       |
| **`thread_cpu_util`**          | timing         | POSIX        | Percentage of thread CPU time (`thread_cpu_clock`) vs. `wall_clock`                                                                                                                            |
| **`monotonic_clock`**          | timing         | POSIX        | Real-clock timer that increments monotonically, unaffected by frequency or time adjustments, that increments while system is asleep                                                            |
| **`monotonic_raw_clock`**      | timing         | POSIX        | Real-clock timer that increments monotonically, unaffected by frequency or time adjustments                                                                                                    |
| **`data_rss`**                 | resource usage | POSIX        | Unshared memory residing the data segment of a process                                                                                                                                         |
| **`stack_rss`**                | resource usage | POSIX        | Integral value of the amount of unshared memory residing in the stack segment of a process                                                                                                     |
| **`num_io_in`**                | resource usage | POSIX        | Number of times the file system had to perform input                                                                                                                                           |
| **`num_io_out`**               | resource usage | POSIX        | Number of times the file system had to perform output                                                                                                                                          |
| **`num_major_page_faults`**    | resource usage | POSIX        | Number of page faults serviced that required I/O activity                                                                                                                                      |
| **`num_minor_page_faults`**    | resource usage | POSIX        | Number of page faults serviced without any I/O activity<sup>[[1]](#fn1)</sup>                                                                                                                  |
| **`num_msg_recv`**             | resource usage | POSIX        | Number of IPC messages received                                                                                                                                                                |
| **`num_msg_sent`**             | resource usage | POSIX        | Number of IPC messages sent                                                                                                                                                                    |
| **`num_signals`**              | resource usage | POSIX        | Number of signals delivered                                                                                                                                                                    |
| **`num_swap`**                 | resource_usage | POSIX        | Number of swaps out of main memory                                                                                                                                                             |
| **`priority_context_switch`**  | resource usage | POSIX        | Number of times a context switch resulted due to a higher priority process becoming runnable or bc the current process exceeded its time slice.                                                |
| **`voluntary_context_switch`** | resource usage | POSIX        | Number of times a context switch resulted due to a process voluntarily giving up the processor before its time slice was completed<sup>[[2]](#fn2)</sup>                                       |

<a name="fn1">[1]</a>: Here I/O activity is avoided by reclaiming a page frame from the list of pages awaiting reallocation

<a name="fn2">[2]</a>: Usually to await availability of a resource

## CUDA Components

> Namespace: `tim::component`

| Component Name       | Category      | Dependencies | Description                                                                 |
| -------------------- | ------------- | ------------ | --------------------------------------------------------------------------- |
| **`cuda_event`**     | timing        | CUDA runtime | Elapsed time between two points in a CUDA stream                            |
| **`nvtx_marker`**    | external tool | CUDA runtime | Inserts CUDA NVTX markers into the code for `nvprof` and/or `NsightSystems` |
| **`cupti_activity`** | GPU           | CUDA, CUPTI  | Provides high-precision runtime activity tracing                            |

## Hardware Counter Components

> Namespace: `tim::component`

| Component Name                         | Category | Template Specification      | Dependencies | Description                                                                             |
| -------------------------------------- | -------- | --------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| **`papi_tuple`**                       | CPU      | `papi_tuple<EventTypes...>` | PAPI         | Variadic list of compile-time specified list of PAPI preset types (e.g. `PAPI_TOT_CYC`) |
| **`papi_array`**<sup>[[3]](#fn3)</sup> | CPU      | `papi_array<N>`             | PAPI         | Variable set of PAPI counters up to size _N_. Supports native hardware counter types    |
| **`cupti_counters`**                   | GPU      |                             | CUDA, CUPTI  | Provides NVIDIA GPU hardware counters for events and metrics                            |

<a name="fn3">[3]</a>: `tim::component::papi_array_t` is pre-defined as `tim::component::papi_array<32>`

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

| Type                               | Description                                                                                                                                                                                                                |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`component_tuple<...>`**         |                                                                                                                                                                                                                            |
| **`component_list<...>`**          | Specified types are wrapped into pointers and initially null. Operations applied to non-null types.                                                                                                                        |
| **`component_hybrid<Tuple,List>`** | Provides static (compile-time enabled) reporting for components specified in `Tuple` (`tim::component_tuple<...>`) and dynamic (runtime-enabled) reporting for components specified in `List` (`tim::component_list<...>`) |
| **`auto_tuple<...>`**              | `component_tuple<...>` + auto start/stop via scope                                                                                                                                                                         |
| **`auto_list<...>`**               | `component_list<...>` + auto start/stop via scope                                                                                                                                                                          |
| **`auto_hybrid<Tuple,List>`**      | `component_hybrid<Tuple, List>` + auto start/stop via scope                                                                                                                                                                |

### Example

```cpp

#include <timemory/timemory.hpp>

#include <cstdint>
#include <cstdio>

//--------------------------------------------------------------------------------------//
//
//      TiMemory specifications
//
//--------------------------------------------------------------------------------------//

using namespace tim::component;

// specify a component_tuple and it's auto type
using tuple_t      = tim::component_tuple<real_clock, cpu_clock>;
using auto_tuple_t = typename tuple_t::auto_type;

using list_t      = tim::component_list<cpu_util, peak_rss>;
using auto_list_t = typename list_t::auto_type;

// specify hybrid of: a tuple (always-on) and a list (optional) and the auto type
using hybrid_t      = tim::component_hybrid<tuple_t, list_t>;
using auto_hybrid_t = typename hybrid_t::auto_type;

//--------------------------------------------------------------------------------------//
//
//      Pre-declarations of implementation details
//
//--------------------------------------------------------------------------------------//

void do_alloc(uint64_t);
void do_sleep(uint64_t);
void do_work(uint64_t);

//--------------------------------------------------------------------------------------//
//
//      Measurement functions
//
//--------------------------------------------------------------------------------------//

void
tuple_func()
{
    auto_tuple_t at("tuple_func");
    do_alloc(100 * tim::units::get_page_size());
    do_sleep(750);
    do_work(250);
}

void
list_func()
{
    auto_list_t al("list_func");
    do_alloc(250 * tim::units::get_page_size());
    do_sleep(250);
    do_work(750);
}

void
hybrid_func()
{
    auto_hybrid_t ah("hybrid_func");
    do_alloc(500 * tim::units::get_page_size());
    do_sleep(500);
    do_work(500);
}

//--------------------------------------------------------------------------------------//
//
//      Main
//
//--------------------------------------------------------------------------------------//

int
main()
{
    auto_hybrid_t ah(__FUNCTION__);
    tuple_func();
    list_func();
    hybrid_func();
}

//--------------------------------------------------------------------------------------//
//
//      Implementation details
//
//--------------------------------------------------------------------------------------//

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

template <typename _Tp>
size_t
random_entry(const std::vector<_Tp>& v)
{
    // this function is provided to make sure memory allocation is not optimized away
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

uint64_t
fibonacci(uint64_t n)
{
    // this function is provided to make sure memory allocation is not optimized away
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

void
do_alloc(uint64_t nsize)
{
    // this function allocates approximately nsize bytes of memory
    std::vector<uint64_t> v(nsize, 15);
    uint64_t              nfib = random_entry(v);
    auto                  ret  = fibonacci(nfib);
    printf("fibonacci(%li) = %li\n", (uint64_t) nfib, ret);
}

void
do_sleep(uint64_t n)
{
    // this function does approximately "n" milliseconds of real time
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

void
do_work(uint64_t n)
{
    // this function does approximately "n" milliseconds of cpu time
    using mutex_t   = std::mutex;
    using lock_t    = std::unique_lock<mutex_t>;
    using condvar_t = std::condition_variable;
    mutex_t mutex;
    lock_t  hold_lk(mutex);
    lock_t  try_lk(mutex, std::defer_lock);
    auto    now   = std::chrono::system_clock::now();
    auto    until = now + std::chrono::milliseconds(n);
    while(std::chrono::system_clock::now() < until)
        try_lk.try_lock();
}
```

#### Compile

```console
g++ -O3 -std=c++11 -I/opt/timemory/include test_example.cpp -o test_example
```

#### Standard Execution (no `cpu_util` or `peak_rss`)

```console
./test_example
```

#### Standard Output (no `cpu_util` or `peak_rss`)

```console
fibonacci(15) = 610
fibonacci(15) = 610
fibonacci(15) = 610

[real]> Outputting 'timemory_output/real.txt'... Done

> [cxx] main          :    3.021 sec real, 1 laps, depth 0 (exclusive:  33.4%)
> [cxx] |_tuple_func  :    1.003 sec real, 1 laps, depth 1
> [cxx] |_hybrid_func :    1.010 sec real, 1 laps, depth 1

[cpu]> Outputting 'timemory_output/cpu.txt'... Done

> [cxx] main          :    1.500 sec cpu, 1 laps, depth 0 (exclusive:  50.0%)
> [cxx] |_tuple_func  :    0.250 sec cpu, 1 laps, depth 1
> [cxx] |_hybrid_func :    0.500 sec cpu, 1 laps, depth 1
```

#### Enabling `cpu_util` and `peak_rss` at Runtime

```console
TIMEMORY_COMPONENT_LIST_INIT="cpu_util,peak_rss" ./test_example
```

#### Enabling `cpu_util` and `peak_rss` at Runtime Output

```console
fibonacci(15) = 610
fibonacci(15) = 610
fibonacci(15) = 610

[peak_rss]> Outputting 'timemory_output/peak_rss.txt'... Done

> [cxx] main          :  27.9 MB peak_rss, 1 laps, depth 0 (exclusive:  11.9%)
> [cxx] |_list_func   :   8.2 MB peak_rss, 1 laps, depth 1
> [cxx] |_hybrid_func :  16.4 MB peak_rss, 1 laps, depth 1

[cpu_util]> Outputting 'timemory_output/cpu_util.txt'... Done

> [cxx] main          :   49.7 % cpu_util, 1 laps, depth 0
> [cxx] |_list_func   :   74.6 % cpu_util, 1 laps, depth 1
> [cxx] |_hybrid_func :   49.6 % cpu_util, 1 laps, depth 1

[real]> Outputting 'timemory_output/real.txt'... Done

> [cxx] main          :    3.016 sec real, 1 laps, depth 0 (exclusive:  33.3%)
> [cxx] |_tuple_func  :    1.002 sec real, 1 laps, depth 1
> [cxx] |_hybrid_func :    1.009 sec real, 1 laps, depth 1

[cpu]> Outputting 'timemory_output/cpu.txt'... Done

> [cxx] main          :    1.500 sec cpu, 1 laps, depth 0 (exclusive:  50.0%)
> [cxx] |_tuple_func  :    0.250 sec cpu, 1 laps, depth 1
> [cxx] |_hybrid_func :    0.500 sec cpu, 1 laps, depth 1
```
