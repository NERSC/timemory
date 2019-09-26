# CUDA and CUPTI Components

| Component Name       | Category   | Dependencies | Description                                                                 |
| -------------------- | ---------- | ------------ | --------------------------------------------------------------------------- |
| **`cuda_event`**     | timing     | CUDA runtime | Elapsed time between two points in a CUDA stream                            |
| **`nvtx_marker`**    | annotation | CUDA runtime | Inserts CUDA NVTX markers into the code for `nvprof` and/or `NsightSystems` |
| **`cupti_counters`** | GPU        | CUDA, CUPTI  | Provides NVIDIA GPU hardware counters for events and metrics                |
| **`cupti_activity`** | GPU        | CUDA, CUPTI  | Provides high-precision runtime activity tracing                            |


## `cuda_event`

> Dependencies: CUDA

Support for recording the elapsed time between two points in a CUDA stream is provided via the `cuda_event` component.

For asynchronous timing with streams, use `TIMEMORY_*CALIPER(id, ...)` macros and `TIMEMORY_CALIPER_APPLY(id, mark_begin, stream)`
and `TIMEMORY_CALIPER_APPLY(id, mark_end, stream)` where `stream` is `0` for the implicit stream or a variable name aliasing an explicit handle to a `cudaStream_t`.

## `nvtx_marker`

> Dependencies: CUDA, NVTX

Inserts CUDA NVTX markers into the code for `nvprof` and/or `NsightSystems`.

For asynchronous markers with streams, use `TIMEMORY_*CALIPER(id, ...)` macros and `TIMEMORY_CALIPER_APPLY(id, mark_begin, stream)`
and `TIMEMORY_CALIPER_APPLY(id, mark_end, stream)` where `stream` is `0` for the implicit stream or a variable name aliasing an explicit handle to a `cudaStream_t`.

## `cupti_counters`

> Dependencies: CUDA, CUPTI

Provides NVIDIA GPU hardware counters for events and metrics.

## `cupti_activity`

> Dependencies: CUDA, CUPTI

Provides high-precision runtime activity tracing.
