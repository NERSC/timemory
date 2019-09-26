# Bundling Components

## Variadic Component Wrappers

> Namespace: `tim`

All the [supported components](supported.md) can be used directly but it is recommended to use these variadic wrapper types.
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

## Example

```cpp

#include <timemory/timemory.hpp>

#include <cstdint>
#include <cstdio>

//--------------------------------------------------------------------------------------//
//
//      timemory specifications
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

### Compile

```console
g++ -O3 -std=c++11 -I/opt/timemory/include test_example.cpp -o test_example
```

### Standard Execution (no `cpu_util` or `peak_rss`)

```console
./test_example
```

### Standard Output (no `cpu_util` or `peak_rss`)

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

### Enabling `cpu_util` and `peak_rss` at Runtime

```console
TIMEMORY_COMPONENT_LIST_INIT="cpu_util,peak_rss" ./test_example
```

### Enabling `cpu_util` and `peak_rss` at Runtime Output

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
