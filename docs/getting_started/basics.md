# Getting Started

For C++ projects, basic functionality simply requires including the header path,
e.g. `-I/usr/local/include` if `timemory.hpp` is in `/usr/local/include/timemory/timemory.hpp`.
However, this will not enable additional capabilities such as PAPI, CUPTI, CUDA kernel timing,
extern templates, etc.

In C++ and Python, TiMemory can be added in one line of code (once the type is declared):

## Supported Languages

### C++

```cpp
using auto_tuple_t = tim::auto_tuple<real_clock, cpu_clock, peak_rss>;
void some_function()
{
    TIMEMORY_OBJECT(auto_tuple_t, "");
    // ...
}
```

### CUDA

```cpp
using auto_tuple_t = tim::auto_tuple<real_clock, cpu_clock, peak_rss, cuda_event>;
void some_function()
{
    TIMEMORY_OBJECT(auto_tuple_t, "");
    // ...
}
```

### Python

```python
@timemory.util.auto_timer()
def some_function():
    # ...
```

### C

In C, TiMemory requires only two lines of code

```c
void* timer = TIMEMORY_OBJECT("", WALL_CLOCK, SYS_CLOCK, USER_CLOCK, PEAK_RSS, CUDA_EVENT);
// ...
FREE_TIMEMORY_OBJECT(timer);
```

When the application terminates, output to text and JSON is automated or controlled via
environment variables.

## Environment Controls

| Environment Variable          | Value Type                                | Description                                                              | Default            |
| ----------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ | ------------------ |
| TIMEMORY_ENABLE               | boolean                                   | Enable TiMemory                                                          | ON                 |
| TIMEMORY_MAX_DEPTH            | integral                                  | Max depth for function call stack to record                              | UINT16_MAX         |
| TIMEMORY_AUTO_OUTPUT          | boolean                                   | Automatic output at the end of application                               | ON                 |
| TIMEMORY_COUT_OUTPUT          | boolean                                   | Enable output to stdout                                                  | ON                 |
| TIMEMORY_FILE_OUTPUT          | boolean                                   | Enable output to file (text and/or JSON)                                 | ON                 |
| TIMEMORY_JSON_OUTPUT          | boolean                                   | Enable JSON output                                                       | OFF                |
| TIMEMORY_TEXT_OUTPUT          | boolean                                   | Enable/disable text output                                               | ON                 |
| TIMEMORY_OUTPUT_PATH          | string                                    | Output folder                                                            | "timemory-output"  |
| TIMEMORY_OUTPUT_PREFIX        | string                                    | Filename prefix for component outputs                                    | ""                 |
| TIMEMORY_WIDTH                | integral                                  | Output width for all component values                                    | component-specific |
| TIMEMORY_TIMING_WIDTH         | integral                                  | Output width of timing component values                                  | component-specific |
| TIMEMORY_MEMORY_WIDTH         | integral                                  | Output width of memory component values                                  | component-specific |
| TIMEMORY_PRECISION            | integral                                  | Precision for all output values                                          | component-specific |
| TIMEMORY_TIMING_PRECISION     | integral                                  | Precision for timing component values                                    | component-specific |
| TIMEMORY_MEMORY_PRECISION     | integral                                  | Precision for memory component values                                    | component-specific |
| TIMEMORY_SCIENTIFIC           | boolean                                   | Output all component values in scientific notation                       | component-specific |
| TIMEMORY_TIMING_SCIENTIFIC    | boolean                                   | Output timing component values in scientific notation                    | component-specific |
| TIMEMORY_MEMORY_SCIENTIFIC    | boolean                                   | Output memory component values in scientific notation                    | component-specific |
| TIMEMORY_TIMING_UNITS         | sec, dsec, csec, msec, usec, nsec, psec   | Units for timing component values                                        | sec (seconds)      |
| TIMEMORY_MEMORY_UNITS         | B, KB, MB, TB, PB, KiB, MiB, TiB, PiB     | Units of memory component values                                         | MB                 |
| TIMEMORY_ROOFLINE_MODE        | ai, op                                    | Mode for roofline calculation if can't be completed in one process       |
| TIMEMORY_ROOFLINE_NUM_THREADS | integral                                  | Number of threads to execute when calculating the "roof" of the roofline |
| TIMEMORY_COMPONENTS           | See [component types](docs:components.md) | Controls the types enabled/disabled by `libtimemory-preload`             |
| TIMEMORY_PAPI_MULTIPLEXING    | boolean                                   | Enable/disable multiplexing                                              |
| TIMEMORY_VERBOSE              | integral                                  | Enable/disable extra messages during execution                           |
| TIMEMORY_PAPI_EVENTS          | PAPI preset and/or native HW counters     | Enables these counters in a `papi_array`                                 |
| TIMEM_USE_SHELL               | boolean                                   | Execute via the user's shell when commands are wrapped by `timem`        |

## Example

```c++
#include <timemory/timemory.hpp>

using real_tuple_t = tim::auto_tuple<tim::component::real_clock>;
using auto_tuple_t =
    tim::auto_tuple<tim::component::real_clock, tim::component::cpu_clock,
                    tim::component::cpu_util, tim::component::peak_rss>;
using comp_tuple_t = typename auto_tuple_t::component_type;

intmax_t
fibonacci(intmax_t n)
{
    TIMEMORY_BASIC_OBJECT(real_tuple_t, "");
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

int
main(int argc, char** argv)
{
    tim::settings::timing_units()      = "sec";
    tim::settings::timing_width()      = 12;
    tim::settings::timing_precision()  = 6;
    tim::settings::timing_scientific() = false;
    tim::settings::memory_units()      = "KB";
    tim::settings::memory_width()      = 12;
    tim::settings::memory_precision()  = 3;
    tim::settings::memory_scientific() = false;
    tim::timemory_init(argc, argv);

    // create a component tuple (does not auto-start)
    comp_tuple_t main("overall timer", true);
    main.start();
    for(auto n : { 10, 11, 12 })
    {
        // create a caliper handle to an auto_tuple_t and have it report when destroyed
        TIMEMORY_BLANK_CALIPER(fib, auto_tuple_t, "fibonacci(", n, ")");
        TIMEMORY_CALIPER_APPLY(fib, report_at_exit, true);
        // run calculation
        auto ret = fibonacci(n);
        // manually stop the auto_tuple_t
        TIMEMORY_CALIPER_APPLY(fib, stop);
        printf("\nfibonacci(%i) = %li\n", n, ret);
    }
    // stop and print
    main.stop();
    std::cout << "\n" << main << std::endl;
}
```

Compile:

```shell
g++ -O3 -I/usr/local example.cc -o example
```

Output:

```shell
fibonacci(10) = 55
> [cxx] fibonacci(10) :     0.001084 sec real,     0.000000 sec cpu,     0.000000 % cpu_util,        0.000 KB peak_rss [laps: 1]

fibonacci(11) = 89
> [cxx] fibonacci(11) :     0.001769 sec real,     0.000000 sec cpu,     0.000000 % cpu_util,        0.000 KB peak_rss [laps: 1]

fibonacci(12) = 144
> [cxx] fibonacci(12) :     0.002878 sec real,     0.000000 sec cpu,     0.000000 % cpu_util,        4.096 KB peak_rss [laps: 1]

> [cxx] overall timer :     0.006072 sec real,     0.000000 sec cpu,     0.000000 % cpu_util,       24.576 KB peak_rss [laps: 1]

[real]> Outputting 'timemory-test-cxx-basic-output/real.txt'... Done

> [cxx] overall timer                       :     0.006072 sec real, 1 laps, depth 0 (exclusive:   5.6%)
> [cxx] |_fibonacci(10)                     :     0.001084 sec real, 1 laps, depth 1 (exclusive:   1.8%)
> [cxx]   |_fibonacci                       :     0.001065 sec real, 1 laps, depth 2 (exclusive:   1.1%)
> [cxx]     |_fibonacci                     :     0.001053 sec real, 2 laps, depth 3 (exclusive:   2.3%)
> [cxx]       |_fibonacci                   :     0.001029 sec real, 4 laps, depth 4 (exclusive:   4.4%)
> [cxx]         |_fibonacci                 :     0.000984 sec real, 8 laps, depth 5 (exclusive:  10.2%)
> [cxx]           |_fibonacci               :     0.000884 sec real, 16 laps, depth 6 (exclusive:  21.2%)
> [cxx]             |_fibonacci             :     0.000697 sec real, 32 laps, depth 7 (exclusive:  44.8%)
> [cxx]               |_fibonacci           :     0.000385 sec real, 52 laps, depth 8 (exclusive:  69.6%)
> [cxx]                 |_fibonacci         :     0.000117 sec real, 44 laps, depth 9 (exclusive:  86.3%)
> [cxx]                   |_fibonacci       :     0.000016 sec real, 16 laps, depth 10 (exclusive: 100.0%)
> [cxx]                     |_fibonacci     :     0.000000 sec real, 2 laps, depth 11
> [cxx] |_fibonacci(11)                     :     0.001769 sec real, 1 laps, depth 1 (exclusive:   0.8%)
> [cxx]   |_fibonacci                       :     0.001755 sec real, 1 laps, depth 2 (exclusive:   0.7%)
> [cxx]     |_fibonacci                     :     0.001742 sec real, 2 laps, depth 3 (exclusive:   1.3%)
> [cxx]       |_fibonacci                   :     0.001719 sec real, 4 laps, depth 4 (exclusive:   2.8%)
> [cxx]         |_fibonacci                 :     0.001671 sec real, 8 laps, depth 5 (exclusive:   5.7%)
> [cxx]           |_fibonacci               :     0.001576 sec real, 16 laps, depth 6 (exclusive:  11.8%)
> [cxx]             |_fibonacci             :     0.001390 sec real, 32 laps, depth 7 (exclusive:  27.2%)
> [cxx]               |_fibonacci           :     0.001012 sec real, 62 laps, depth 8 (exclusive:  50.3%)
> [cxx]                 |_fibonacci         :     0.000503 sec real, 84 laps, depth 9 (exclusive:  73.0%)
> [cxx]                   |_fibonacci       :     0.000136 sec real, 58 laps, depth 10 (exclusive:  86.8%)
> [cxx]                     |_fibonacci     :     0.000018 sec real, 18 laps, depth 11 (exclusive: 100.0%)
> [cxx]                       |_fibonacci   :     0.000000 sec real, 2 laps, depth 12
> [cxx] |_fibonacci(12)                     :     0.002878 sec real, 1 laps, depth 1 (exclusive:   0.5%)
> [cxx]   |_fibonacci                       :     0.002864 sec real, 1 laps, depth 2 (exclusive:   0.4%)
> [cxx]     |_fibonacci                     :     0.002852 sec real, 2 laps, depth 3 (exclusive:   0.9%)
> [cxx]       |_fibonacci                   :     0.002827 sec real, 4 laps, depth 4 (exclusive:   1.7%)
> [cxx]         |_fibonacci                 :     0.002779 sec real, 8 laps, depth 5 (exclusive:   3.3%)
> [cxx]           |_fibonacci               :     0.002688 sec real, 16 laps, depth 6 (exclusive:   7.3%)
> [cxx]             |_fibonacci             :     0.002493 sec real, 32 laps, depth 7 (exclusive:  15.3%)
> [cxx]               |_fibonacci           :     0.002111 sec real, 64 laps, depth 8 (exclusive:  32.4%)
> [cxx]                 |_fibonacci         :     0.001426 sec real, 114 laps, depth 9 (exclusive:  56.5%)
> [cxx]                   |_fibonacci       :     0.000621 sec real, 128 laps, depth 10 (exclusive:  75.4%)
> [cxx]                     |_fibonacci     :     0.000153 sec real, 74 laps, depth 11 (exclusive:  90.2%)
> [cxx]                       |_fibonacci   :     0.000015 sec real, 20 laps, depth 12 (exclusive: 100.0%)
> [cxx]                         |_fibonacci :     0.000000 sec real, 2 laps, depth 13

[cpu]> Outputting 'timemory-test-cxx-basic-output/cpu.txt'... Done

> [cxx] overall timer   :     0.000000 sec cpu, 1 laps, depth 0
> [cxx] |_fibonacci(10) :     0.000000 sec cpu, 1 laps, depth 1
> [cxx] |_fibonacci(11) :     0.000000 sec cpu, 1 laps, depth 1
> [cxx] |_fibonacci(12) :     0.000000 sec cpu, 1 laps, depth 1

[cpu_util]> Outputting 'timemory-test-cxx-basic-output/cpu_util.txt'... Done

> [cxx] overall timer   :     0.000000 % cpu_util, 1 laps, depth 0
> [cxx] |_fibonacci(10) :     0.000000 % cpu_util, 1 laps, depth 1
> [cxx] |_fibonacci(11) :     0.000000 % cpu_util, 1 laps, depth 1
> [cxx] |_fibonacci(12) :     0.000000 % cpu_util, 1 laps, depth 1

[peak_rss]> Outputting 'timemory-test-cxx-basic-output/peak_rss.txt'... Done

> [cxx] overall timer   :       24.576 KB peak_rss, 1 laps, depth 0 (exclusive:  83.3%)
> [cxx] |_fibonacci(10) :        0.000 KB peak_rss, 1 laps, depth 1
> [cxx] |_fibonacci(11) :        0.000 KB peak_rss, 1 laps, depth 1
> [cxx] |_fibonacci(12) :        4.096 KB peak_rss, 1 laps, depth 1
```
