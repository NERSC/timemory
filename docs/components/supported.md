# Supported Components

## Component Wrappers

Bundling components is provided through the variadic component wrappers. It is recommended to use these wrappers in all instances.

| C++ (object)                       | C (enum) | Python (enum) |
| ---------------------------------- | -------- | ------------- |
| **`component_tuple<...>`**         |          |               |
| **`component_list<...>`**          |          |               |
| **`component_hybrid<Tuple,List>`** |          |               |
| **`auto_tuple<...>`**              |          |               |
| **`auto_list<...>`**               |          |               |
| **`auto_hybrid<Tuple,List>`**      |          |               |

[Detailed documentation](bundling.md)

## Native Components

These components are available on all operating systems (Windows, macOS, Linux).

| C++ (object)        | C (enum)          | Python (enum)                       |
| ------------------- | ----------------- | ----------------------------------- |
| **`wall_clock`**    | **`WALL_CLOCK`**  | **`timemory.component.wall_clock`** |
| **`user_clock`**    | **`USER_CLOCK`**  | **`timemory.component.user_clock`** |
| **`system_clock`**  | **`SYS_CLOCK`**   | **`timemory.component.sys_clock`**  |
| **`cpu_clock`**     | **`CPU_CLOCK`**   | **`timemory.component.cpu_clock`**  |
| **`cpu_util`**      | **`CPU_UTIL`**    | **`timemory.component.cpu_util`**   |
| **`page_rss`**      | **`CURRENT_RSS`** | **`timemory.component.page_rss`**   |
| **`peak_rss`**      | **`PEAK_RSS`**    | **`timemory.component.peak_rss`**   |
| **`trip_count`**    | **`TRIP_COUNT`**  | **`timemory.component.trip_count`** |
| **`real_clock`**    | **`WALL_CLOCK`**  | **`timemory.component.wall_clock`** |
| **`virtual_clock`** | **`WALL_CLOCK`**  | **`timemory.component.wall_clock`** |

[Detailed documentation](native.md)

## System-Dependent Components

These components are only available on certain operating systems. In general, all of these components are available for POSIX systems.
The components in the "resource usage" category are provided by POSIX `rusage` (`man getrusage`).

| C++ (object)                   | C (enum)                       | Python (enum)                                      |
| ------------------------------ | ------------------------------ | -------------------------------------------------- |
| **`read_bytes`**               | **`READ_BYTES`**               | **`timemory.components.read_bytes`**               |
| **`written_bytes`**            | **`WRITTEN_BYTES`**            | **`timemory.components.written_bytes`**            |
| **`process_cpu_clock`**        | **`PROCESS_CPU_CLOCK`**        | **`timemory.components.process_cpu_clock`**        |
| **`thread_cpu_clock`**         | **`THREAD_CPU_CLOCK`**         | **`timemory.components.thread_cpu_clock`**         |
| **`process_cpu_util`**         | **`PROCESS_CPU_UTIL`**         | **`timemory.components.process_cpu_util`**         |
| **`thread_cpu_util`**          | **`THREAD_CPU_UTIL`**          | **`timemory.components.thread_cpu_util`**          |
| **`monotonic_clock`**          | **`MONOTONIC_CLOCK`**          | **`timemory.components.monotonic_clock`**          |
| **`monotonic_raw_clock`**      | **`MONOTONIC_RAW_CLOCK`**      | **`timemory.components.monotonic_raw_clock`**      |
| **`data_rss`**                 | **`DATA_RSS`**                 | **`timemory.components.data_rss`**                 |
| **`stack_rss`**                | **`STACK_RSS`**                | **`timemory.components.stack_rss`**                |
| **`num_io_in`**                | **`NUM_IO_IN`**                | **`timemory.components.num_io_in`**                |
| **`num_io_out`**               | **`NUM_IO_OUT`**               | **`timemory.components.num_io_out`**               |
| **`num_major_page_faults`**    | **`NUM_MAJOR_PAGE_FAULTS`**    | **`timemory.components.num_major_page_faults`**    |
| **`num_minor_page_faults`**    | **`NUM_MINOR_PAGE_FAULTS`**    | **`timemory.components.num_minor_page_faults`**    |
| **`num_msg_recv`**             | **`NUM_MSG_RECV`**             | **`timemory.components.num_msg_recv`**             |
| **`num_msg_sent`**             | **`NUM_MSG_SENT`**             | **`timemory.components.num_msg_sent`**             |
| **`num_signals`**              | **`NUM_SIGNALS`**              | **`timemory.components.num_signals`**              |
| **`num_swap`**                 | **`NUM_SWAP`**                 | **`timemory.components.num_swap`**                 |
| **`priority_context_switch`**  | **`PRIORITY_CONTEXT_SWITCH`**  | **`timemory.components.priority_context_switch`**  |
| **`voluntary_context_switch`** | **`VOLUNTARY_CONTEXT_SWITCH`** | **`timemory.components.voluntary_context_switch`** |

[Detailed documentation](os-dependent.md)

## CUDA Components

| C++ (object)         | C (enum)             | Python (enum)                            |
| -------------------- | -------------------- | ---------------------------------------- |
| **`cuda_event`**     | **`CUDA_EVENT`**     | **`timemory.components.cuda_event`**     |
| **`nvtx_marker`**    | **`NVTX_MARKER`**    | **`timemory.components.nvtx_marker`**    |
| **`cupti_counters`** | **`CUPTI_COUNTERS`** | **`timemory.components.cupti_counters`** |
| **`cupti_activity`** | **`CUPTI_ACTIVITY`** | **`timemory.components.cupti_activity`** |

[Detailed documentation](cuda.md)

## gperftools Components

The gperftools components provide the ability to start and stop the gperftools CPU profiler and heap profiler. These components depend on the
available of the gperftools "profiler" and "tcmalloc" library and headers, respectively.

| C++ (object)              | C (enum)                  | Python (enum)                                 |
| ------------------------- | ------------------------- | --------------------------------------------- |
| **`gperf_cpu_profiler`**  | **`GPERF_CPU_PROFILER`**  | **`timemory.components.gperf_cpu_profiler`**  |
| **`gperf_heap_profiler`** | **`GPERF_HEAP_PROFILER`** | **`timemory.components.gperf_heap_profiler`** |

## PAPI Components

| C++ (object)               | C (enum)         | Python (enum)                        |
| -------------------------- | ---------------- | ------------------------------------ |
| **`papi_tuple<Types...>`** | N/A              | N/A                                  |
| **`papi_array<Size>`**     | **`PAPI_ARRAY`** | **`timemory.components.papi_array`** |

[Detailed documentation](papi.md)

## External Instrumentation Components

These components provide tools similar to timemory but are commonly used to enable their service/features through the timemory interface.
The primary benefit to using these instrumentations through timemory is the __*optional*__ support for these instrumentation toolkits
via template filtering and/or CMake COMPONENTS.
There is planned support for features from [LIKWID](https://github.com/RRZE-HPC/likwid).

The `caliper` component provides integration and access to the [Caliper Toolkit](https://github.com/LLNL/Caliper)
([documentation](https://software.llnl.gov/Caliper/index.html)). Caliper is a flexible application introspection system that provide numerous similar tools
to timemory and many others including, but not limited to timing, memory, annotation, MPI, CUDA, PAPI call-stack unwinding, sampling.

| C++ (object)  | C (enum)      | Python (enum)                     |
| ------------- | ------------- | --------------------------------- |
| **`caliper`** | **`CALIPER`** | **`timemory.components.caliper`** |

## GOTCHA Components

[GOTCHA](https://github.com/LLNL/GOTCHA) is a library that wraps functions and is used to place hook into external libraries.
It is similar to LD_PRELOAD, but operates via a programmable API.
This enables easy methods of accomplishing tasks like code instrumentation or wholesale replacement of mechanisms in programs without disrupting their source code.

The `gotcha` component in timemory supports implicit extraction of the wrapped function return type and arguments and
significantly reduces the complexity of a traditional GOTCHA specification.
Additionally, limited support for C++ function mangling required to intercept C++ function calls.

| C++ (object) | C (enum) | Python (enum) |
| ------------ | -------- | ------------- |
| **`gotcha`** | N/A      | N/A           |

[Detailed GOTCHA documentation](gotcha.md)

## Roofline Components

| C++ (object)                | C (enum)                    | Python (enum)                                   |
| --------------------------- | --------------------------- | ----------------------------------------------- |
| **`cpu_roofline`**          | **`CPU_ROOFLINE`**          |                                                 |
| **`gpu_roofline`**          | **`GPU_ROOFLINE`**          |                                                 |
| **`cpu_roofline_flops`**    | **`CPU_ROOFLINE_FLOPS`**    | **`timemory.components.cpu_roofline_flops`**    |
| **`cpu_roofline_dp_flops`** | **`CPU_ROOFLINE_DP_FLOPS`** | **`timemory.components.cpu_roofline_dp_flops`** |
| **`cpu_roofline_sp_flops`** | **`CPU_ROOFLINE_SP_FLOPS`** | **`timemory.components.cpu_roofline_sp_flops`** |
| **`gpu_roofline_flops`**    | **`GPU_ROOFLINE_FLOPS`**    | **`timemory.components.gpu_roofline_flops`**    |
| **`gpu_roofline_dp_flops`** | **`GPU_ROOFLINE_DP_FLOPS`** | **`timemory.components.gpu_roofline_dp_flops`** |
| **`gpu_roofline_sp_flops`** | **`GPU_ROOFLINE_SP_FLOPS`** | **`timemory.components.gpu_roofline_sp_flops`** |
| **`gpu_roofline_hp_flops`** | **`GPU_ROOFLINE_HP_FLOPS`** | **`timemory.components.gpu_roofline_hp_flops`** |

[Detailed documentation](roofline.md)
