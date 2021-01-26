# KokkosP Profiling Tools

Kokkos tools support is built into the main library (`libtimemory.so` or `libtimemory.dylib`) and can be
enabled in any Kokkos application by settings the environment variable `KOKKOS_PROFILE_LIBRARY` to the path
to this library or via the command-line `--kokkos-tools-library=/path/to/libtimemory.<ext>` if the application
passes the arguments from `main(...)` to Kokkos during initialization, i.e. `Kokkos::Initialize(argc, argv);`.
Selecting which components are collected can be done via the kokkos-tools specific environment variable
`TIMEMORY_KOKKOS_COMPONENTS` or the generic environment variable `TIMEMORY_GLOBAL_COMPONENTS`. If timemory
is not directly used anywhere in your application, it does not matter which environment variable is used.
If timemory is used elsewhere and you would like the kokkos-tools component selection to not be applied
globally, use the former.

If you would prefer to avoid setting several environment variables, it maybe useful to create a timemory
config file in `~/.timemory.cfg` or `~/.config/timemory.cfg` or specify the location of the config
file via the environment variable `TIMEMORY_CONFIG=/path/to/file`. The format is simple: `setting = value [values...]`.
Additionally, timemory provides several zero-config libraries when cmake is configured with `TIMEMORY_BUILD_KOKKOS_TOOLS=ON`
and `TIMEMORY_BUILD_KOKKOS_CONFIG=ON`. The exact set of libraries built are subject to which components are
available and will be installed in `${CMAKE_PREFIX_PATH}/${CMAKE_INSTALL_LIBDIR}/timemory/kokkos-tools/`.
These libraries have a default output path to `timemory-@LIB_NAME@-output/@TIMESTAMP@`, where `@LIB_NAME@` is the
basename of the library and `@TIMESTAMP@` is the time-stamp from when timemory was initialized.
Use `timemory-avail -S` to see how to disable time-stamping and customize other settings.

Configuring cmake with `TIMEMORY_BUILD_KOKKOS_TOOLS=ON` and `TIMEMORY_BUILD_KOKKOS_CONFIG=OFF` will install just
two libraries in `${CMAKE_PREFIX_PATH}/${CMAKE_INSTALL_LIBDIR}/timemory/kokkos-tools/`: `kp_timemory.<ext>`
and `kp_timemory_filter.<ext>`. In most cases, using `kp_timemory.<ext>` is not necessary, it is essentially
the same as using `libtimemory.<ext>` except it has hidden symbols so that separate data can be collected
when timemory is used in the main application. `kp_timemory_filter.<ext>` ensures that all external
profiler controller components (currently: vtune_profiler, cuda_profiler, craypat_record, and allinea_map)
are disabled outside of the Kokkos regions matching the regex values of the `KOKKOS_PROFILE_REGEX` environment
variable. The default value of `KOKKOS_PROFILE_REGEX` is `"^[A-Za-z]"` (i.e. any Kokkos label starting with a letter).

## Build

```bash
git clone https://github.com/NERSC/timemory.git timemory
cmake -B build-timemory -DTIMEMORY_BUILD_KOKKOS_TOOLS=ON -DTIMEMORY_BUILD_KOKKOS_CONFIG=ON timemory
cmake --build build-timemory --target install --parallel 8
```

> Specify `-DCMAKE_INSTALL_PREFIX=/path/to/install/dir` and add any additional options, e.g. `-DTIMEMORY_USE_GOTCHA=ON`

## Running Kokkos application with timemory enabled

Before executing the Kokkos application you have to set the environment variable `KOKKOS_PROFILE_LIBRARY`
to point to the path to the installed `libtimemory.<ext>` or pre-built component configuration library.
Here are several examples:

```console
export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/libtimemory.so
export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/timemory/kokkos-tools/kp_timemory.so
export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/timemory/kokkos-tools/kp_timemory_context_switch.so
./myexe --kokkos-tools-library=/opt/timemory/lib/libtimemory.so
```

## Selecting Different Components

Use the `timemory-avail` command-line tool provided by timemory to find the alias for the tool desired. Use `-d` to get a
description of the tool or `-h` to see all options. Once the desired components have been identified, place
the components in a comma-delimited list in the environment variable `TIMEMORY_KOKKOS_COMPONENTS`, e.g.

```console
export TIMEMORY_KOKKOS_COMPONENTS="wall_clock, peak_rss, cpu_roofline_dp_flops"
```

## Filtering Kokkos Labels

The following will filter out any Kokkos profiling label which was not specifically given a name
(and thus was labeled with the demangled function name):

```console
export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/timemory/kokkos-tools/kp_timemory_filter.so
export KOKKOS_PROFILE_REGEX="(?!Kokkos::)"
```

## Sample Output

```console
$ export TIMEMORY_KOKKOS_COMPONENTS="wall_clock, peak_rss"
$ export TIMEMORY_TIMING_PRECISION=6
$ export TIMEMORY_MEMORY_UNITS=kb
$ export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/timemory/kokkos-tools/kp_timemory.so
$ ./sample

libgomp: Affinity not supported on this configuration
  Total size S = 4194304 N = 4096 M = 1024
#---------------------------------------------------------------------------#
# KokkosP: timemory Connector (sequence is 0, version: 20200625)
#---------------------------------------------------------------------------#

  Computed result for 4096 x 1024 is 4194304.000000
  N( 4096 ) M( 1024 ) nrepeat ( 100 ) problem( 33.5954 MB ) time( 0.155611 s ) bandwidth( 21.5893 GB/s )

#---------------------------------------------------------------------------#
KokkosP: Finalization of timemory Connector. Complete.
#---------------------------------------------------------------------------#

[kokkos_memory]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/kokkos_memory.json'...
[kokkos_memory]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/kokkos_memory.tree.json'...
[kokkos_memory]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/kokkos_memory.txt'...

[wall]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/wall.json'...
[wall]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/wall.tree.json'...
[wall]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/wall.txt'...

[peak_rss]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/peak_rss.tree.json'...
[peak_rss]|0> Outputting 'timemory-kokkosp-output/2021-01-21_03.10_PM/peak_rss.txt'...

$ cat timemory-kokkosp-output/2021-01-21_03.10_PM/peak_rss.txt
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                              MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                               LABEL                                 | COUNT  | DEPTH  | METRIC   | UNITS  |   SUM     |   MEAN    |   MIN     |   MAX     | STDDEV | % SELF |
|---------------------------------------------------------------------|--------|--------|----------|--------|-----------|-----------|-----------|-----------|--------|--------|
| >>> kokkos/allocate/Host/y                                          |      1 |      0 | peak_rss | KB     | 34013.184 | 34013.184 | 34013.184 | 34013.184 |  0.000 |    0.7 |
| >>> |_kokkos/dev0/Kokkos::View::initialization [unsigned long long] |      1 |      1 | peak_rss | KB     |   143.360 |   143.360 |   143.360 |   143.360 |  0.000 |  100.0 |
| >>> |_kokkos/allocate/Host/x                                        |      1 |      1 | peak_rss | KB     | 33619.968 | 33619.968 | 33619.968 | 33619.968 |  0.000 |    0.0 |
| >>>   |_kokkos/dev0/Kokkos::View::initialization [long long]        |      1 |      2 | peak_rss | KB     |     8.192 |     8.192 |     8.192 |     8.192 |  0.000 |  100.0 |
| >>>   |_kokkos/allocate/Host/A                                      |      1 |      2 | peak_rss | KB     | 33603.584 | 33603.584 | 33603.584 | 33603.584 |  0.000 |    0.1 |
| >>>     |_kokkos/dev0/Kokkos::View::initialization [A]              |      1 |      3 | peak_rss | KB     | 33554.432 | 33554.432 | 33554.432 | 33554.432 |  0.000 |  100.0 |
| >>>     |_kokkos/deep_copy/Host=y/Host=y                            |      1 |      3 | peak_rss | KB     |     4.096 |     4.096 |     4.096 |     4.096 |  0.000 |  100.0 |
| >>>     |_kokkos/deep_copy/Host=x/Host=x                            |      1 |      3 | peak_rss | KB     |     0.000 |     0.000 |     0.000 |     0.000 |  0.000 |    0.0 |
| >>>     |_kokkos/deep_copy/Host=A/Host=A                            |      1 |      3 | peak_rss | KB     |     0.000 |     0.000 |     0.000 |     0.000 |  0.000 |    0.0 |
| >>>     |_kokkos/dev0/yAx                                           |    100 |      3 | peak_rss | KB     |    12.288 |     0.123 |    12.288 |    12.288 |  1.229 |  100.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

$ cat timemory-kokkosp-output/2021-01-21_03.10_PM/wall.txt
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                 REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                               LABEL                                 | COUNT  | DEPTH  | METRIC | UNITS  |   SUM    |   MEAN   |   MIN    |   MAX    | STDDEV   | % SELF |
|---------------------------------------------------------------------|--------|--------|--------|--------|----------|----------|----------|----------|----------|--------|
| >>> kokkos/allocate/Host/y                                          |      1 |      0 | wall   | sec    | 0.173479 | 0.173479 | 0.173479 | 0.173479 | 0.000000 |    0.5 |
| >>> |_kokkos/dev0/Kokkos::View::initialization [unsigned long long] |      1 |      1 | wall   | sec    | 0.000148 | 0.000148 | 0.000148 | 0.000148 | 0.000000 |  100.0 |
| >>> |_kokkos/allocate/Host/x                                        |      1 |      1 | wall   | sec    | 0.172399 | 0.172399 | 0.172399 | 0.172399 | 0.000000 |    1.7 |
| >>>   |_kokkos/dev0/Kokkos::View::initialization [long long]        |      1 |      2 | wall   | sec    | 0.000009 | 0.000009 | 0.000009 | 0.000009 | 0.000000 |  100.0 |
| >>>   |_kokkos/allocate/Host/A                                      |      1 |      2 | wall   | sec    | 0.169482 | 0.169482 | 0.169482 | 0.169482 | 0.000000 |    2.5 |
| >>>     |_kokkos/dev0/Kokkos::View::initialization [A]              |      1 |      3 | wall   | sec    | 0.011172 | 0.011172 | 0.011172 | 0.011172 | 0.000000 |  100.0 |
| >>>     |_kokkos/deep_copy/Host=y/Host=y                            |      1 |      3 | wall   | sec    | 0.000006 | 0.000006 | 0.000006 | 0.000006 | 0.000000 |  100.0 |
| >>>     |_kokkos/deep_copy/Host=x/Host=x                            |      1 |      3 | wall   | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 |  100.0 |
| >>>     |_kokkos/deep_copy/Host=A/Host=A                            |      1 |      3 | wall   | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 |  100.0 |
| >>>     |_kokkos/dev0/yAx                                           |    100 |      3 | wall   | sec    | 0.154131 | 0.001541 | 0.001471 | 0.001471 | 0.000113 |  100.0 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
```
