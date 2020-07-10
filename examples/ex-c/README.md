# ex-c

This example demonstrates the use of timemory timing components (timers), resident set size (rss) and context switches for instrumentation of C code. The instrumentation marker containing the following components is used for instrumentation.

```c
TIMEMORY_MARKER(func, WALL_CLOCK, SYS_CLOCK, USER_CLOCK, CPU_CLOCK,
                      CPU_UTIL, PAGE_RSS, PEAK_RSS, PRIORITY_CONTEXT_SWITCH,
                      VOLUNTARY_CONTEXT_SWITCH, CALIPER);
```

## Build

See [examples](../README.md##Build). This examples requires an additional `-DTIMEMORY_BUILD_C=ON` flag to be built.

## Expected Output

```bash
$ ./ex_c_timing
#------------------------- tim::manager initialized [id=0][pid=29906] -------------------------#

'../examples/ex-c/ex_c_timing.c' : main @ 105. Running fibonacci(43, 28)...
Initializing caliper...
main (untimed): fibonacci(43, 43) = 433494437
main (timed): fibonacci(43, 28) = 433494437
# laps = 9576

'../examples/ex-c/ex_c_timing.c' : main @ 134 --> n = 433494437, 433494437, 433494437
[vol_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/vol_cxt_swch.json'...
[vol_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/vol_cxt_swch.txt'...
Opening 'timemory-ex-c-timing-output/vol_cxt_swch.jpeg' for output...
Closed 'timemory-ex-c-timing-output/vol_cxt_swch.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------------------------|
| NUMBER OF CONTEXT SWITCHES DUE TO A PROCESS VOLUNTARILY GIVING UP THE PROCESSOR BEFORE ITS TIME SLICE WAS COMPLETED                       |
| ----------------------------------------------------------------------------------------------------------------------------------------- |
| LABEL                                                                                                                                     | COUNT   | DEPTH   | METRIC         | UNITS   | SUM   | MEAN   | MIN   | MAX   | STDDEV   | % SELF   |
| -------------------------------------------------------                                                                                   | ------- | ------- | -------------- | ------- | ----- | ------ | ----- | ----- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                           | 1       | 0       | vol_cxt_swch   |         | 0     | 0      | 0     | 0     | 0        | 0        |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                     | 1       | 0       | vol_cxt_swch   |         | 0     | 0      | 0     | 0     | 0        | 0        |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                     | 1       | 0       | vol_cxt_swch   |         | 0     | 0      | 0     | 0     | 0        | 0        |
| ----------------------------------------------------------------------------------------------------------------------------------------- |

[prio_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/prio_cxt_swch.json'...
[prio_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/prio_cxt_swch.txt'...
Opening 'timemory-ex-c-timing-output/prio_cxt_swch.jpeg' for output...
Closed 'timemory-ex-c-timing-output/prio_cxt_swch.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------|
| NUMBER OF CONTEXT SWITCH DUE TO HIGHER PRIORITY PROCESS BECOMING RUNNABLE OR BECAUSE THE CURRENT PROCESS EXCEEDED ITS TIME SLICE)          |
| ------------------------------------------------------------------------------------------------------------------------------------------ |
| LABEL                                                                                                                                      | COUNT   | DEPTH   | METRIC          | UNITS   | SUM   | MEAN   | MIN   | MAX   | STDDEV   | % SELF   |
| -------------------------------------------------------                                                                                    | ------- | ------- | --------------- | ------- | ----- | ------ | ----- | ----- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                            | 1       | 0       | prio_cxt_swch   |         | 2     | 2      | 2     | 2     | 0        | 100      |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                      | 1       | 0       | prio_cxt_swch   |         | 2     | 2      | 2     | 2     | 0        | 100      |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                      | 1       | 0       | prio_cxt_swch   |         | 1     | 1      | 1     | 1     | 0        | 100      |
| ------------------------------------------------------------------------------------------------------------------------------------------ |

[peak_rss]|0> Outputting 'timemory-ex-c-timing-output/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-ex-c-timing-output/peak_rss.txt'...
Opening 'timemory-ex-c-timing-output/peak_rss.jpeg' for output...
Closed 'timemory-ex-c-timing-output/peak_rss.jpeg'...

|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LABEL                                                                                                                                                    | COUNT                     | DEPTH    | METRIC     | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| ------------------------------------------------------------                                                                                             | --------                  | -------- | ---------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                                          | 1                         | 0        | peak_rss   | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0      |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                                    | 1                         | 0        | peak_rss   | MB       | 0.628    | 0.628    | 0.628    | 0.628    | 0.000    | 100.0    |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                                    | 1                         | 0        | peak_rss   | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0      |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 1        | 1          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 2        | 2          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 4        | 3          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 8        | 4          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 16       | 5          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 32       | 6          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 64       | 7          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| ------------------------------------------------------------                                                                                             | --------                  | -------- | ---------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 128      | 8          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 247      | 9          | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 382      | 10         | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 386      | 11         | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 232      | 12         | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 79       | 13         | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 14       | 14         | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                      | _fibonacci[using_tuple=0] | 1        | 15         | peak_rss | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- |

[page_rss]|0> Outputting 'timemory-ex-c-timing-output/page_rss.json'...
[page_rss]|0> Outputting 'timemory-ex-c-timing-output/page_rss.txt'...
Opening 'timemory-ex-c-timing-output/page_rss.jpeg' for output...
Closed 'timemory-ex-c-timing-output/page_rss.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------------------------|
| AMOUNT OF MEMORY ALLOCATED IN PAGES OF MEMORY. UNLIKE PEAK_RSS, VALUE WILL FLUCTUATE AS MEMORY IS FREED/ALLOCATED                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------- |
| LABEL                                                                                                                                               | COUNT    | DEPTH    | METRIC     | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| -------------------------------------------------------                                                                                             | -------- | -------- | ---------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                                     | 1        | 0        | page_rss   | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0      |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                               | 1        | 0        | page_rss   | MB       | 0.643    | 0.643    | 0.643    | 0.643    | 0.000    | 100.0    |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                               | 1        | 0        | page_rss   | MB       | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0      |
| --------------------------------------------------------------------------------------------------------------------------------------------------- |

[cpu_util]|0> Outputting 'timemory-ex-c-timing-output/cpu_util.json'...
[cpu_util]|0> Outputting 'timemory-ex-c-timing-output/cpu_util.txt'...
Opening 'timemory-ex-c-timing-output/cpu_util.jpeg' for output...
Closed 'timemory-ex-c-timing-output/cpu_util.jpeg'...

|-------------------------------------------------------------------------------------------------------------------------------------------------|
| PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------- |
| LABEL                                                                                                                                             | COUNT                     | DEPTH   | METRIC     | UNITS    | SUM     | MEAN    | MIN     | MAX     | STDDEV   | % SELF   |
| ------------------------------------------------------------                                                                                      | -------                   | ------- | ---------- | -------  | ------- | ------- | ------- | ------- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                                   | 1                         | 0       | cpu_util   | %        | 99.7    | 99.7    | 99.7    | 99.7    | 0.0      | 100.0    |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                             | 1                         | 0       | cpu_util   | %        | 99.6    | 99.6    | 99.6    | 99.6    | 0.0      | 100.0    |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                             | 1                         | 0       | cpu_util   | %        | 100.4   | 100.4   | 100.4   | 100.4   | 0.0      | 0.0      |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 1       | 1          | cpu_util | %       | 100.4   | 100.4   | 100.4   | 100.4    | 0.0      | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 2       | 2          | cpu_util | %       | 100.4   | 50.2    | 100.4   | 100.4    | 0.0      | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 4       | 3          | cpu_util | %       | 100.4   | 25.1    | 100.2   | 100.6    | 0.2      | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 8       | 4          | cpu_util | %       | 100.4   | 12.6    | 98.3    | 101.3    | 1.0      | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 16      | 5          | cpu_util | %       | 100.4   | 6.3     | 85.7    | 106.1    | 5.0      | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 32      | 6          | cpu_util | %       | 100.5   | 3.1     | 69.2    | 138.9    | 17.6     | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 64      | 7          | cpu_util | %       | 100.5   | 1.6     | 0.0     | 182.4    | 41.8     | 0.0   |
| ------------------------------------------------------------                                                                                      | -------                   | ------- | ---------- | -------  | ------- | ------- | ------- | ------- | -------- | -------- |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 128     | 8          | cpu_util | %       | 100.6   | 0.8     | 0.0     | 476.8    | 91.0     | 1.1   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 247     | 9          | cpu_util | %       | 99.5    | 0.4     | 0.0     | 778.8    | 142.7    | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 382     | 10         | cpu_util | %       | 103.7   | 0.3     | 0.0     | 779.1    | 202.1    | 2.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 386     | 11         | cpu_util | %       | 101.6   | 0.3     | 0.0     | 779.0    | 219.4    | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 232     | 12         | cpu_util | %       | 113.4   | 0.5     | 0.0     | 779.1    | 259.3    | 6.8   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 79      | 13         | cpu_util | %       | 105.7   | 1.3     | 0.0     | 778.9    | 255.4    | 0.0   |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 14      | 14         | cpu_util | %       | 105.8   | 7.6     | 0.0     | 770.2    | 233.0    | 100.0 |
| >>>                                                                                                                                               | _fibonacci[using_tuple=0] | 1       | 15         | cpu_util | %       | 0.0     | 0.0     | 0.0     | 0.0      | 0.0      | 0.0   |
| ------------------------------------------------------------------------------------------------------------------------------------------------- |

[cpu]|0> Outputting 'timemory-ex-c-timing-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-c-timing-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-c-timing-output/cpu.txt'...
Opening 'timemory-ex-c-timing-output/cpu.jpeg' for output...
Closed 'timemory-ex-c-timing-output/cpu.jpeg'...

|-------------------------------------------------------------------------------------------------------------------------------------------------|
| TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------- |
| LABEL                                                                                                                                             | COUNT    | DEPTH    | METRIC   | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| -------------------------------------------------------                                                                                           | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                                   | 1        | 0        | cpu      | sec      | 1.120    | 1.120    | 1.120    | 1.120    | 0.000    | 100.0    |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                             | 1        | 0        | cpu      | sec      | 1.090    | 1.090    | 1.090    | 1.090    | 0.000    | 100.0    |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                             | 1        | 0        | cpu      | sec      | 1.100    | 1.100    | 1.100    | 1.100    | 0.000    | 100.0    |
| ------------------------------------------------------------------------------------------------------------------------------------------------- |

[user]|0> Outputting 'timemory-ex-c-timing-output/user.flamegraph.json'...
[user]|0> Outputting 'timemory-ex-c-timing-output/user.json'...
[user]|0> Outputting 'timemory-ex-c-timing-output/user.txt'...
Opening 'timemory-ex-c-timing-output/user.jpeg' for output...
Closed 'timemory-ex-c-timing-output/user.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------------|
| CPU TIME SPENT IN USER-MODE                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LABEL                                                                                                                                                  | COUNT                     | DEPTH    | METRIC   | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                                        | 1                         | 0        | user     | sec      | 1.120    | 1.120    | 1.120    | 1.120    | 0.000    | 100.0    |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                                  | 1                         | 0        | user     | sec      | 1.090    | 1.090    | 1.090    | 1.090    | 0.000    | 0.0      |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 1        | 1        | user     | sec      | 1.090    | 1.090    | 1.090    | 1.090    | 0.000    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 2        | 2        | user     | sec      | 1.090    | 0.545    | 0.410    | 0.680    | 0.191    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 4        | 3        | user     | sec      | 1.090    | 0.273    | 0.160    | 0.420    | 0.108    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 8        | 4        | user     | sec      | 1.090    | 0.136    | 0.060    | 0.260    | 0.063    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 16       | 5        | user     | sec      | 1.090    | 0.068    | 0.020    | 0.160    | 0.035    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 32       | 6        | user     | sec      | 1.090    | 0.034    | 0.010    | 0.100    | 0.019    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 64       | 7        | user     | sec      | 1.090    | 0.017    | 0.000    | 0.060    | 0.011    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 128      | 8        | user     | sec      | 1.090    | 0.009    | 0.000    | 0.040    | 0.007    | 0.0   |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 247      | 9        | user     | sec      | 1.090    | 0.004    | 0.000    | 0.020    | 0.005    | 8.3   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 382      | 10       | user     | sec      | 1.000    | 0.003    | 0.000    | 0.010    | 0.004    | 31.0  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 386      | 11       | user     | sec      | 0.690    | 0.002    | 0.000    | 0.010    | 0.004    | 43.5  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 232      | 12       | user     | sec      | 0.390    | 0.002    | 0.000    | 0.010    | 0.004    | 76.9  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 79       | 13       | user     | sec      | 0.090    | 0.001    | 0.000    | 0.010    | 0.003    | 88.9  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 14       | 14       | user     | sec      | 0.010    | 0.001    | 0.000    | 0.010    | 0.003    | 100.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 1        | 15       | user     | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0   |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                                  | 1                         | 0        | user     | sec      | 1.100    | 1.100    | 1.100    | 1.100    | 0.000    | 0.0      |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 1        | 1        | user     | sec      | 1.100    | 1.100    | 1.100    | 1.100    | 0.000    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 2        | 2        | user     | sec      | 1.100    | 0.550    | 0.420    | 0.680    | 0.184    | 0.0   |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 4        | 3        | user     | sec      | 1.100    | 0.275    | 0.160    | 0.420    | 0.108    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 8        | 4        | user     | sec      | 1.100    | 0.138    | 0.060    | 0.260    | 0.062    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 16       | 5        | user     | sec      | 1.100    | 0.069    | 0.020    | 0.160    | 0.035    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 32       | 6        | user     | sec      | 1.100    | 0.034    | 0.010    | 0.100    | 0.020    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 64       | 7        | user     | sec      | 1.100    | 0.017    | 0.000    | 0.060    | 0.011    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 128      | 8        | user     | sec      | 1.100    | 0.009    | 0.000    | 0.040    | 0.008    | 1.8   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 247      | 9        | user     | sec      | 1.080    | 0.004    | 0.000    | 0.030    | 0.006    | 3.7   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 382      | 10       | user     | sec      | 1.040    | 0.003    | 0.000    | 0.020    | 0.005    | 27.9  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 386      | 11       | user     | sec      | 0.750    | 0.002    | 0.000    | 0.010    | 0.004    | 44.0  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 232      | 12       | user     | sec      | 0.420    | 0.002    | 0.000    | 0.010    | 0.004    | 71.4  |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 79       | 13       | user     | sec      | 0.120    | 0.002    | 0.000    | 0.010    | 0.004    | 83.3  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 14       | 14       | user     | sec      | 0.020    | 0.001    | 0.000    | 0.010    | 0.004    | 100.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 1        | 15       | user     | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ |

[sys]|0> Outputting 'timemory-ex-c-timing-output/sys.flamegraph.json'...
[sys]|0> Outputting 'timemory-ex-c-timing-output/sys.json'...
[sys]|0> Outputting 'timemory-ex-c-timing-output/sys.txt'...
Opening 'timemory-ex-c-timing-output/sys.jpeg' for output...
Closed 'timemory-ex-c-timing-output/sys.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------------|
| CPU TIME SPENT IN KERNEL-MODE                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LABEL                                                                                                                                                  | COUNT                     | DEPTH    | METRIC   | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                                        | 1                         | 0        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0      |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                                  | 1                         | 0        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0      |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 1        | 1        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 2        | 2        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 4        | 3        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 8        | 4        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 16       | 5        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 32       | 6        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 64       | 7        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 128      | 8        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 247      | 9        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 382      | 10       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 386      | 11       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 232      | 12       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 79       | 13       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 14       | 14       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 1        | 15       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                                  | 1                         | 0        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0      |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 1        | 1        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 2        | 2        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 4        | 3        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 8        | 4        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 16       | 5        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 32       | 6        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 64       | 7        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 128      | 8        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 247      | 9        | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 382      | 10       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 386      | 11       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 232      | 12       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 79       | 13       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 14       | 14       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 1        | 15       | sys      | sec      | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.0 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ |

[wall]|0> Outputting 'timemory-ex-c-timing-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-c-timing-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-c-timing-output/wall.txt'...
Opening 'timemory-ex-c-timing-output/wall.jpeg' for output...
Closed 'timemory-ex-c-timing-output/wall.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------------|
| REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LABEL                                                                                                                                                  | COUNT                     | DEPTH    | METRIC   | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]                                                                                                        | 1                         | 0        | wall     | sec      | 1.124    | 1.124    | 1.124    | 1.124    | 0.000    | 100.0    |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]                                                                                                  | 1                         | 0        | wall     | sec      | 1.094    | 1.094    | 1.094    | 1.094    | 0.000    | 0.0      |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 1        | 1        | wall     | sec      | 1.094    | 1.094    | 1.094    | 1.094    | 0.000    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 2        | 2        | wall     | sec      | 1.094    | 0.547    | 0.417    | 0.677    | 0.184    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 4        | 3        | wall     | sec      | 1.094    | 0.274    | 0.159    | 0.418    | 0.107    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 8        | 4        | wall     | sec      | 1.094    | 0.137    | 0.061    | 0.258    | 0.061    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 16       | 5        | wall     | sec      | 1.094    | 0.068    | 0.023    | 0.159    | 0.035    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 32       | 6        | wall     | sec      | 1.094    | 0.034    | 0.009    | 0.098    | 0.019    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 64       | 7        | wall     | sec      | 1.094    | 0.017    | 0.003    | 0.061    | 0.011    | 0.1   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 128      | 8        | wall     | sec      | 1.093    | 0.009    | 0.001    | 0.038    | 0.006    | 0.7   |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 247      | 9        | wall     | sec      | 1.085    | 0.004    | 0.001    | 0.023    | 0.003    | 7.6   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 382      | 10       | wall     | sec      | 1.003    | 0.003    | 0.001    | 0.014    | 0.002    | 26.4  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 386      | 11       | wall     | sec      | 0.738    | 0.002    | 0.001    | 0.009    | 0.001    | 49.8  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 232      | 12       | wall     | sec      | 0.370    | 0.002    | 0.001    | 0.006    | 0.001    | 69.3  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 79       | 13       | wall     | sec      | 0.114    | 0.001    | 0.001    | 0.003    | 0.000    | 83.4  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 14       | 14       | wall     | sec      | 0.019    | 0.001    | 0.001    | 0.002    | 0.000    | 93.1  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=1] | 1        | 15       | wall     | sec      | 0.001    | 0.001    | 0.001    | 0.001    | 0.000    | 100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]                                                                                                  | 1                         | 0        | wall     | sec      | 1.096    | 1.096    | 1.096    | 1.096    | 0.000    | 0.0      |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 1        | 1        | wall     | sec      | 1.095    | 1.095    | 1.095    | 1.095    | 0.000    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 2        | 2        | wall     | sec      | 1.095    | 0.548    | 0.418    | 0.677    | 0.183    | 0.0   |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 4        | 3        | wall     | sec      | 1.095    | 0.274    | 0.160    | 0.418    | 0.107    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 8        | 4        | wall     | sec      | 1.095    | 0.137    | 0.061    | 0.259    | 0.062    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 16       | 5        | wall     | sec      | 1.095    | 0.068    | 0.023    | 0.160    | 0.035    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 32       | 6        | wall     | sec      | 1.095    | 0.034    | 0.009    | 0.099    | 0.019    | 0.0   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 64       | 7        | wall     | sec      | 1.095    | 0.017    | 0.003    | 0.061    | 0.011    | 0.1   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 128      | 8        | wall     | sec      | 1.094    | 0.009    | 0.001    | 0.038    | 0.006    | 0.8   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 247      | 9        | wall     | sec      | 1.086    | 0.004    | 0.001    | 0.023    | 0.003    | 7.6   |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 382      | 10       | wall     | sec      | 1.003    | 0.003    | 0.001    | 0.014    | 0.002    | 26.4  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 386      | 11       | wall     | sec      | 0.738    | 0.002    | 0.001    | 0.009    | 0.001    | 49.8  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 232      | 12       | wall     | sec      | 0.370    | 0.002    | 0.001    | 0.006    | 0.001    | 69.3  |
| ------------------------------------------------------------                                                                                           | --------                  | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 79       | 13       | wall     | sec      | 0.114    | 0.001    | 0.001    | 0.003    | 0.000    | 83.4  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 14       | 14       | wall     | sec      | 0.019    | 0.001    | 0.001    | 0.002    | 0.000    | 93.2  |
| >>>                                                                                                                                                    | _fibonacci[using_tuple=0] | 1        | 15       | wall     | sec      | 0.001    | 0.001    | 0.001    | 0.001    | 0.000    | 100.0 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ |


[metadata::manager::finalize]> Outputting 'timemory-ex-c-timing-output/metadata.json'...


#---------------------- tim::manager destroyed [rank=0][id=0][pid=29906] ----------------------#
```