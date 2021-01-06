# timemory-mallocp library

Produces a `libtimemory-mallocp.so` that uses GOTCHA wrappers around common memory allocation functions:

- `malloc`
- `calloc`
- `free`
- `cudaMalloc` (requires `TIMEMORY_USE_CUDA`)
- `cudaMallocHost` (requires `TIMEMORY_USE_CUDA`)
- `cudaMallocManaged` (requires `TIMEMORY_USE_CUDA`)
- `cudaHostAlloc` (requires `TIMEMORY_USE_CUDA`)
- `cudaFree` (requires `TIMEMORY_USE_CUDA`)
- `cudaFreeHost` (requires `TIMEMORY_USE_CUDA`)

Four functions are provided for C, C++, and Fortran:

- `uint64_t timemory_start_mallocp()`
  - Returns the number of initializations
- `uint64_t timemory_stop_mallocp(uint64_t idx)`
  - Removes the initialization request at `idx`
  - Returns the number of remaining initializations
- `void timemory_register_mallocp()`
  - Ensures a global initialization exists until it deregistration
- `void timemory_deregister_mallocp()`
  - Deactivates the global initialization

## Usage

The environement variable `ENABLE_TIMEMORY_MALLOCP` (default: `"ON"`) controls configuration of the instrumentation.

```cpp

#include <timemory/tools/timemory-mallocp.h>

void foo()
{
    // ... something using memory allocation functions ...
}

void bar(int val)
{
    uint64_t mallocp_idx = UINT64_MAX;
    if(val > 100)
        mallocp_idx = timemory_start_mallocp();

    // ... something using memory allocation functions ...

    if(val > 100)
        timemory_stop_mallocp(mallocp_idx);
}

int main()
{
    timemory_register_mallocp();
    // mallocp will be active for all the following
    foo();
    bar(10);
    bar(1000);
    foo();
    timemory_deregister_mallocp();

    foo();     // mallocp will NOT be activate
    bar(10);   // mallocp will NOT be active
    bar(1000); // mallocp will be activated and deactivated
    foo();     // mallocp will NOT be activate
}
```
