# Features

## Cross-Language Support

It is very common for Python projects to implement expensive routines in C or C++. Implementing a timemory auto-tuple in any combination
of these languages will produce one combined report for all the languages (provided each language links to the same library).
However, this is a feature of timemory. Timemory can be used in standalone C, C++, or Python projects.

## Multithreading

The multithreading overhead is essentially zero.
All timemory components use static thread-local singletons of call-graps that are automatically created when a
new thread starts recording a component. The state of the singleton on the master thread is bookmarked and when the
thread is destroyed, the thread-local call-graph is merged back into the master call-graph. Only during the
one-time merge into the master call-graph is a synchronization lock (mutex) utilized.

## Distributed Memory Parallelism

If a project uses MPI or UPC++, timemory components will automatically provide support for combining this data
into a single output, support per-rank output, or.

## Hardware Counters

CPU hardware counters are available via PAPI components.
GPU hardware counters are available via the `cupti_counters` component.

## Plot Generation in Python

The results from timemory can be serialized to JSON and the JSON output can be used to produce performance plots
via the standalone `timemory-plotter` or `timemory.plotting` Python module. For roofline analysis,
the `timemory.roofline` module can be used.

## Command-line Tools

The command-line tools `timem` and `pytimem` work like the `time` executable but with more information.
UNIX systems provide `timem` executable that works like `time`. On all systems, `pytimem` is provided.

```shell
$ ./timem sleep 2

> [sleep] total execution time :
        2.119142 sec real
        0.000000 sec user
        0.000000 sec sys
        0.000000 sec cpu
        0.000000 % cpu_util
        0.745472 MB peak_rss
               0 io_in
               0 io_out
             389 minor_page_flts
               0 major_page_flts
               0 num_signals
              46 vol_cxt_swch
              10 prio_cxt_swch
```

## Signal Detection

Timemory provides a facility for catching signals and printing out a backtrace when the signals are raised:

```cpp
tim::enable_signal_detection({ SIGHUP, SIGINT, SIGQUIT, SIGABRT });
// ...
tim::disable_signal_detection();
```

## Cache Information

Timemory provides method on Linux, Windows, and macOS to query the size of L1, L2, and L3 cache.
A `get_max()` function is provided for convenience as some systems (e.g. KNL) do not have an L3 cache.

> Namespace: `tim::ert::cache_size`

| Cache level | Function(s)                                                      |
| ----------- | ---------------------------------------------------------------- |
| L1          | `tim::ert::cache_size::get(1)`, `tim::ert::cache_size::get<1>()` |
| L2          | `tim::ert::cache_size::get(2)`, `tim::ert::cache_size::get<2>()` |
| L3          | `tim::ert::cache_size::get(3)`, `tim::ert::cache_size::get<3>()` |
| max         | `tim::ert::cache_size::get_max()`                                |

