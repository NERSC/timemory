# KokkosP Profiling Tools

## Quick Start for Kokkos

### Build

```bash
mkdir build && cd build
cmake -DTIMEMORY_BUILD_KOKKOS_TOOLS=ON -DTIMEMORY_KOKKOS_BUILD_CONFIG=ON ..
make
```

### Using Different Components

Use the command-line tool provided by timemory to find the alias for the tool desired. Use `-d` to get a
description of the tool or `-h` to see all options. Once the desired components have been identified, place
the components in a comma-delimited list in the environment variable `KOKKOS_TIMEMORY_COMPONENTS`, e.g.

```console
export KOKKOS_TIMEMORY_COMPONENTS="wall_clock, peak_rss, cpu_roofline_dp_flops"
```

## Run kokkos application with timemory enabled

Before executing the Kokkos application you have to set the environment variable `KOKKOS_PROFILE_LIBRARY` to point to the name of the dynamic library. Also add the library path of PAPI and PAPI connector to `LD_LIBRARY_PATH`.
The kokkos profiling libraries will be installed to `${CMAKE_INSTALL_LIBDIR}/timemory/kokkos`.

```console
export KOKKOS_PROFILE_LIBRARY=kp_timemory.so
```

## Run kokkos application with PAPI recording enabled

Internally, timemory uses the `TIMEMORY_PAPI_EVENTS` environment variable for specifying arbitrary events.
However, this library will attempt to read `PAPI_EVENTS` and set `TIMEMORY_PAPI_EVENTS` before the PAPI
component is initialized, if using `PAPI_EVENTS` does not provide the desired events, use `TIMEMORY_PAPI_EVENTS`.

Example enabling (1) total instructions, (2) total cycles, (3) total load/stores

```console
export PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_LST_INS"
export TIMEMORY_PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_LST_INS"
```

## Run kokkos application with Roofline recording enabled

[Roofline Performance Model](https://docs.nersc.gov/programming/performance-debugging-tools/roofline/)

On both the CPU and GPU, calculating the roofline requires two executions of the application.
It is recommended to use the timemory python interface to generate the roofline because
the `timemory.roofline` submodule provides a mode that will handle executing the application
twice and generating the plot. For advanced usage, see the
[timemory Roofline Documentation](https://timemory.readthedocs.io/en/latest/getting_started/roofline/).

```console
export KOKKOS_ROOFLINE=ON
export OMP_NUM_THREADS=4
export KOKKOS_TIMEMORY_COMPONENTS="cpu_roofline_dp_flops"
timemory-roofline -n 4 -t cpu_roofline -- ./sample
```

## Building Sample

```shell
cmake -DBUILD_SAMPLE=ON ..
make -j2
```

## Sample Output

```console
#---------------------------------------------------------------------------#
# KokkosP: TiMemory Connector (sequence is 0, version: 0)
#---------------------------------------------------------------------------#

#--------------------- tim::manager initialized [0][0] ---------------------#

fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073
fibonacci(47) = 2971215073

#---------------------------------------------------------------------------#
KokkosP: Finalization of TiMemory Connector. Complete.
#---------------------------------------------------------------------------#


[peak_rss]|0> Outputting 'docker-desktop_26927/peak_rss.json'...
[peak_rss]|0> Outputting 'docker-desktop_26927/peak_rss.txt'...

>>> sample                              :  214.9 MB peak_rss,  1 laps, depth 0
>>> |_kokkos/dev0/thread_creation       :    0.3 MB peak_rss,  1 laps, depth 1
>>> |_kokkos/dev0/fibonacci             : 2142.2 MB peak_rss, 10 laps, depth 1 (exclusive:  39.8%)
>>>   |_kokkos/dev0/fibonacci_runtime_0 :  209.8 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_1 :  202.7 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_2 :  191.2 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_3 :  175.8 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_4 :  156.3 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_5 :  133.0 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_6 :  105.8 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_7 :   74.7 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_8 :   39.7 MB peak_rss,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_9 :    0.8 MB peak_rss,  1 laps, depth 2

[wall]|0> Outputting 'docker-desktop_26927/wall.json'...
[wall]|0> Outputting 'docker-desktop_26927/wall.txt'...

>>> sample                              :   21.612 sec wall,  1 laps, depth 0
>>> |_kokkos/dev0/thread_creation       :    0.007 sec wall,  1 laps, depth 1
>>> |_kokkos/dev0/fibonacci             :  200.820 sec wall, 10 laps, depth 1 (exclusive:   2.8%)
>>>   |_kokkos/dev0/fibonacci_runtime_0 :   19.136 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_1 :   19.368 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_2 :   19.497 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_3 :   19.537 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_4 :   19.555 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_5 :   19.621 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_6 :   19.650 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_7 :   19.621 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_8 :   19.621 sec wall,  1 laps, depth 2
>>>   |_kokkos/dev0/fibonacci_runtime_9 :   19.558 sec wall,  1 laps, depth 2

[metadata::manager::finalize]> Outputting 'docker-desktop_26927/metadata.json'...


#---------------------- tim::manager destroyed [0][0] ----------------------#
```

