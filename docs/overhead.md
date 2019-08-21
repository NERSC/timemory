# Overhead

Analysis on a fibonacci calculation determined that one TiMemory "component" adds an average overhead of 1 microsecond (`0.000001 s`) when the component
is being inserted into call-graph for the first time.
Once a component exists in the call-graph, the overhead is approximately 0.85 microseconds (`0.00000085 s`).
However, this is for a **_VERY_** large number of measurements, when the number of measurements are kept within a reasonable range (approximately <= 10,000 - 15,000)
and number of unique measurements is kept to a minimum, depending on cache re-use, timemory can have **_ZERO_** overhead.

| Unique Measurements | Total Measurements | No Measurements (sec) | Using Measurements (sec) | Difference (sec) | Avg. Overhead (sec) |
| :-----------------: | :----------------: | :-------------------: | :----------------------: | :--------------: | :-----------------: |
|       16,720        |       16,720       |       1.352e+00       |        1.350e+00         |    -2.421e-03    |     -1.440e-07      |
|         16          |       27,056       |       1.365e+00       |        1.353e+00         |    -1.199e-02    |     -4.430e-07      |
|      2,056,912      |     2,056,912      |       1.367e+00       |        3.425e+00         |    2.059e+00     |      1.000e-06      |
|         27          |     2,056,912      |       1.365e+00       |        3.103e+00         |    1.738e+00     |      8.450e-07      |

> The exact performance is specific to the machine and the overhead for a particular machine can be calculated by running/modifying the `test_cxx_overhead` example
> in `examples/ex-cxx-overhead`.

## Test Problem

The following pair of fibonacci functions provide an almost direct measurment of the overhead of timemory.
Aside from the creation, starting, stopping, storing, and accumulation of the timemory components, the fibonacci calculation is a simple
but highly scalable calculation that just adds a large number of integers together.

```cpp
int64_t fibonacci(int64_t n) { return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2)); }

int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    using auto_tuple_t = tim::auto_tuple<real_clock, system_clock, user_clock, trip_count>;
    if(n > cutoff)
    {
        nlaps += auto_tuple_t::size();
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[", n, "]");
        return (n < 2) ? n : (fibonacci(n - 1, cutoff) + fibonacci(n - 2, cutoff));
    }
    return fibonacci(n);
}
```

In order to re-use measurements (reduce unique measurements), we make the following modification:

```cpp
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    using auto_tuple_t = tim::auto_tuple<real_clock, system_clock, user_clock, trip_count>;
    if(n > cutoff)
    {
        nlaps += auto_tuple_t::size();
        // TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[", n, "]");
        TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_t, __FUNCTION__);
        return (n < 2) ? n : (fibonacci(n - 1, cutoff) + fibonacci(n - 2, cutoff));
    }
    return fibonacci(n);
}
```

## Measuring Overhead of 16,720 unique measurements

Every single instance is unique and `fibonacci(43, 26)` produces:

The difference between the two measurements (i.e. negative overhead per measurement) is due to minute differences in the cache and CPU frequency.

```console
Report from 16720 total measurements:
	> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':86 : 1.352e+00 sec real [laps: 1]
	> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':86 : 1.350e+00 sec real [laps: 1]
	> [cxx] timing difference                                    : -2.421e-03 sec real
	> [cxx] average overhead per timer                           : -1.440e-07 sec real

> [cxx] main[./test_cxx_overhead]@'test_cxx_overhead.cpp':132 : 2.690e+00 sec user, 1 laps, depth  0 (exclusive:  50.2%)
> [cxx] |_fibonacci[43]                                       : 1.340e+00 sec user, 1 laps, depth  1 (exclusive:   0.0%)
> [cxx]   |_fibonacci[42]                                     : 8.300e-01 sec user, 1 laps, depth  2 (exclusive:   0.0%)
> [cxx]     |_fibonacci[41]                                   : 5.100e-01 sec user, 1 laps, depth  3 (exclusive:   0.0%)
> [cxx]       |_fibonacci[40]                                 : 3.200e-01 sec user, 1 laps, depth  4 (exclusive:   0.0%)
> [cxx]         |_fibonacci[39]                               : 2.000e-01 sec user, 1 laps, depth  5 (exclusive:   0.0%)
> [cxx]           |_fibonacci[38]                             : 1.200e-01 sec user, 1 laps, depth  6 (exclusive:   0.0%)
> [cxx]             |_fibonacci[37]                           : 8.000e-02 sec user, 1 laps, depth  7 (exclusive:   0.0%)
> [cxx]               |_fibonacci[36]                         : 5.000e-02 sec user, 1 laps, depth  8 (exclusive:   0.0%)
> [cxx]                 |_fibonacci[35]                       : 3.000e-02 sec user, 1 laps, depth  9 (exclusive:   0.0%)
> [cxx]                   |_fibonacci[34]                     : 2.000e-02 sec user, 1 laps, depth 10 (exclusive:   0.0%)
> [cxx]                     |_fibonacci[33]                   : 1.000e-02 sec user, 1 laps, depth 11 (exclusive:   0.0%)
> [cxx]                       |_fibonacci[32]                 : 1.000e-02 sec user, 1 laps, depth 12 (exclusive:   0.0%)
> [cxx]                         |_fibonacci[31]               : 0.000e+00 sec user, 1 laps, depth 13 (exclusive:   0.0%)
> [cxx]                           |_fibonacci[30]             : 0.000e+00 sec user, 1 laps, depth 14 (exclusive:   0.0%)
> [cxx]                             |_fibonacci[29]           : 0.000e+00 sec user, 1 laps, depth 15 (exclusive:   0.0%)
> [cxx]                               |_fibonacci[28]         : 0.000e+00 sec user, 1 laps, depth 16 (exclusive:   0.0%)
> [cxx]                                 |_fibonacci[27]       : 0.000e+00 sec user, 1 laps, depth 17
> [cxx]                               |_fibonacci[27]         : 0.000e+00 sec user, 1 laps, depth 16
> [cxx]                             |_fibonacci[28]           : 0.000e+00 sec user, 1 laps, depth 15 (exclusive:   0.0%)
> [cxx]                               |_fibonacci[27]         : 0.000e+00 sec user, 1 laps, depth 16
> [cxx]                           |_fibonacci[29]             : 0.000e+00 sec user, 1 laps, depth 14 (exclusive:   0.0%)
> [cxx]                             |_fibonacci[28]           : 0.000e+00 sec user, 1 laps, depth 15 (exclusive:   0.0%)
> [cxx]                               |_fibonacci[27]         : 0.000e+00 sec user, 1 laps, depth 16
> [cxx]                             |_fibonacci[27]           : 0.000e+00 sec user, 1 laps, depth 15
> [cxx]                         |_fibonacci[30]               : 1.000e-02 sec user, 1 laps, depth 13 (exclusive:   0.0%)
> [cxx]                           |_fibonacci[29]             : 1.000e-02 sec user, 1 laps, depth 14 (exclusive:   0.0%)
...
```

> Output is truncated and/or not shown for all components (`real_clock`, `system_clock`, `user_clock`, and `trip_count`)

## Measuring Overhead of 16 unique measurements and 27,056 total measurements

By reducing the number of unique measurements, we can increase the total number of measurements and produce the same effect as above (approx. **_zero overhead_**).
`fibonacci(43, 25)` increases the total number of measurements from 16,720 to 27,056 but reduces the number of unique measurments to 16.

```console
Report from 27056 total measurements:
	> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':86 : 1.365e+00 sec real [laps: 1]
	> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':86 : 1.353e+00 sec real [laps: 1]
	> [cxx] timing difference                                    : -1.199e-02 sec real
	> [cxx] average overhead per timer                           : -4.430e-07 sec real

> [cxx] main[./test_cxx_overhead]@'test_cxx_overhead.cpp':132  : 2.718e+00 sec real,    1 laps, depth  0 (exclusive:   0.0%)
> [cxx] |_run [with timing = false]@'test_cxx_overhead.cpp':86 : 1.365e+00 sec real,    1 laps, depth  1
> [cxx] |_run [with timing =  true]@'test_cxx_overhead.cpp':86 : 1.353e+00 sec real,    1 laps, depth  1 (exclusive:   0.0%)
> [cxx]   |_fibonacci                                          : 1.353e+00 sec real,    1 laps, depth  2 (exclusive:   0.0%)
> [cxx]     |_fibonacci                                        : 1.353e+00 sec real,    2 laps, depth  3 (exclusive:   0.0%)
> [cxx]       |_fibonacci                                      : 1.353e+00 sec real,    4 laps, depth  4 (exclusive:   0.0%)
> [cxx]         |_fibonacci                                    : 1.353e+00 sec real,    8 laps, depth  5 (exclusive:   0.0%)
> [cxx]           |_fibonacci                                  : 1.353e+00 sec real,   16 laps, depth  6 (exclusive:   0.0%)
> [cxx]             |_fibonacci                                : 1.353e+00 sec real,   32 laps, depth  7 (exclusive:   0.0%)
> [cxx]               |_fibonacci                              : 1.353e+00 sec real,   64 laps, depth  8 (exclusive:   0.0%)
> [cxx]                 |_fibonacci                            : 1.352e+00 sec real,  128 laps, depth  9 (exclusive:   0.1%)
> [cxx]                   |_fibonacci                          : 1.352e+00 sec real,  256 laps, depth 10 (exclusive:   0.1%)
> [cxx]                     |_fibonacci                        : 1.350e+00 sec real,  511 laps, depth 11 (exclusive:   1.1%)
> [cxx]                       |_fibonacci                      : 1.336e+00 sec real,  968 laps, depth 12 (exclusive:   7.3%)
> [cxx]                         |_fibonacci                    : 1.238e+00 sec real, 1486 laps, depth 13 (exclusive:  22.9%)
> [cxx]                           |_fibonacci                  : 9.547e-01 sec real, 1586 laps, depth 14 (exclusive:  43.3%)
> [cxx]                             |_fibonacci                : 5.414e-01 sec real, 1093 laps, depth 15 (exclusive:  61.8%)
> [cxx]                               |_fibonacci              : 2.068e-01 sec real,  470 laps, depth 16 (exclusive:  76.1%)
> [cxx]                                 |_fibonacci            : 4.935e-02 sec real,  121 laps, depth 17 (exclusive:  86.7%)
> [cxx]                                   |_fibonacci          : 6.587e-03 sec real,   17 laps, depth 18 (exclusive:  94.3%)
> [cxx]                                     |_fibonacci        : 3.742e-04 sec real,    1 laps, depth 19
```

> Output is truncated and/or not shown for all components (`real_clock`, `system_clock`, `user_clock`, and `trip_count`)

The difference between the two measurements (i.e. negative overhead per measurement) is due to minute differences in the cache and CPU frequency.

## Measuring Overhead of 2,056,912 unique measurements

Every single instance is unique and `fibonacci(43, 16)` produces:

```console
Report from 2056912 total measurements:
	> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':86 : 1.367e+00 sec real [laps: 1]
	> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':86 : 3.425e+00 sec real [laps: 1]
	> [cxx] timing difference                                    : 2.059e+00 sec real
	> [cxx] average overhead per timer                           : 1.000e-06 sec real

> [cxx] main[./test_cxx_overhead]@'test_cxx_overhead.cpp':132                 : 5.388e+00 sec real, 1 laps, depth  0 (exclusive:   0.0%)
> [cxx] |_run [with timing = false]@'test_cxx_overhead.cpp':86                : 1.370e+00 sec real, 1 laps, depth  1
> [cxx] |_run [with timing =  true]@'test_cxx_overhead.cpp':86                : 4.018e+00 sec real, 1 laps, depth  1 (exclusive:   0.0%)
> [cxx]   |_fibonacci[43]                                                     : 4.018e+00 sec real, 1 laps, depth  2 (exclusive:   0.0%)
> [cxx]     |_fibonacci[42]                                                   : 2.486e+00 sec real, 1 laps, depth  3 (exclusive:   0.0%)
> [cxx]       |_fibonacci[41]                                                 : 1.537e+00 sec real, 1 laps, depth  4 (exclusive:   0.0%)
> [cxx]         |_fibonacci[40]                                               : 9.509e-01 sec real, 1 laps, depth  5 (exclusive:   0.0%)
> [cxx]           |_fibonacci[39]                                             : 5.880e-01 sec real, 1 laps, depth  6 (exclusive:   0.0%)
> [cxx]             |_fibonacci[38]                                           : 3.638e-01 sec real, 1 laps, depth  7 (exclusive:   0.0%)
> [cxx]               |_fibonacci[37]                                         : 2.250e-01 sec real, 1 laps, depth  8 (exclusive:   0.0%)
> [cxx]                 |_fibonacci[36]                                       : 1.392e-01 sec real, 1 laps, depth  9 (exclusive:   0.0%)
> [cxx]                   |_fibonacci[35]                                     : 8.607e-02 sec real, 1 laps, depth 10 (exclusive:   0.0%)
> [cxx]                     |_fibonacci[34]                                   : 5.325e-02 sec real, 1 laps, depth 11 (exclusive:   0.0%)
> [cxx]                       |_fibonacci[33]                                 : 3.293e-02 sec real, 1 laps, depth 12 (exclusive:   0.0%)
> [cxx]                         |_fibonacci[32]                               : 2.037e-02 sec real, 1 laps, depth 13 (exclusive:   0.0%)
> [cxx]                           |_fibonacci[31]                             : 1.260e-02 sec real, 1 laps, depth 14 (exclusive:   0.1%)
> [cxx]                             |_fibonacci[30]                           : 7.802e-03 sec real, 1 laps, depth 15 (exclusive:   0.1%)
> [cxx]                               |_fibonacci[29]                         : 4.827e-03 sec real, 1 laps, depth 16 (exclusive:   0.2%)
> [cxx]                                 |_fibonacci[28]                       : 2.992e-03 sec real, 1 laps, depth 17 (exclusive:   0.3%)
> [cxx]                                   |_fibonacci[27]                     : 1.857e-03 sec real, 1 laps, depth 18 (exclusive:   0.5%)
> [cxx]                                     |_fibonacci[26]                   : 1.152e-03 sec real, 1 laps, depth 19 (exclusive:   0.8%)
> [cxx]                                       |_fibonacci[25]                 : 7.149e-04 sec real, 1 laps, depth 20 (exclusive:   1.3%)
> [cxx]                                         |_fibonacci[24]               : 4.468e-04 sec real, 1 laps, depth 21 (exclusive:   1.9%)
> [cxx]                                           |_fibonacci[23]             : 2.829e-04 sec real, 1 laps, depth 22 (exclusive:   5.1%)
> [cxx]                                             |_fibonacci[22]           : 1.706e-04 sec real, 1 laps, depth 23 (exclusive:   6.8%)
> [cxx]                                               |_fibonacci[21]         : 1.041e-04 sec real, 1 laps, depth 24 (exclusive:   8.3%)
> [cxx]                                                 |_fibonacci[20]       : 6.541e-05 sec real, 1 laps, depth 25 (exclusive:  14.9%)
> [cxx]                                                   |_fibonacci[19]     : 4.105e-05 sec real, 1 laps, depth 26 (exclusive:  28.4%)
> [cxx]                                                     |_fibonacci[18]   : 2.300e-05 sec real, 1 laps, depth 27 (exclusive:  61.1%)
> [cxx]                                                       |_fibonacci[17] : 8.943e-06 sec real, 1 laps, depth 28
> [cxx]                                                     |_fibonacci[17]   : 6.383e-06 sec real, 1 laps, depth 27
> [cxx]                                                   |_fibonacci[18]     : 1.458e-05 sec real, 1 laps, depth 26 (exclusive:  57.3%)
> [cxx]                                                     |_fibonacci[17]   : 6.233e-06 sec real, 1 laps, depth 27
> [cxx]                                                 |_fibonacci[19]       : 3.009e-05 sec real, 1 laps, depth 25 (exclusive:  28.1%)
> [cxx]                                                   |_fibonacci[18]     : 1.544e-05 sec real, 1 laps, depth 26 (exclusive:  59.6%)
> [cxx]                                                     |_fibonacci[17]   : 6.231e-06 sec real, 1 laps, depth 27
> [cxx]                                                   |_fibonacci[17]     : 6.182e-06 sec real, 1 laps, depth 26
> [cxx]                                               |_fibonacci[20]         : 5.493e-05 sec real, 1 laps, depth 24 (exclusive:  17.7%)
> [cxx]                                                 |_fibonacci[19]       : 3.117e-05 sec real, 1 laps, depth 25 (exclusive:  35.3%)
> [cxx]                                                   |_fibonacci[18]     : 1.394e-05 sec real, 1 laps, depth 26 (exclusive:  55.9%)
> [cxx]                                                     |_fibonacci[17]   : 6.140e-06 sec real, 1 laps, depth 27
> [cxx]                                                   |_fibonacci[17]     : 6.227e-06 sec real, 1 laps, depth 26
> [cxx]                                                 |_fibonacci[18]       : 1.406e-05 sec real, 1 laps, depth 25 (exclusive:  56.4%)
...

> [cxx] main[./test_cxx_overhead]@'test_cxx_overhead.cpp':132               : 4.170e+00 sec user, 1 laps, depth  0 (exclusive:  31.9%)
> [cxx] |_fibonacci[43]                                                     : 2.840e+00 sec user, 1 laps, depth  1
> [cxx]   |_fibonacci[42]                                                   : 1.760e+00 sec user, 1 laps, depth  2 (exclusive:   0.0%)
> [cxx]     |_fibonacci[41]                                                 : 1.110e+00 sec user, 1 laps, depth  3 (exclusive:   0.0%)
> [cxx]       |_fibonacci[40]                                               : 7.000e-01 sec user, 1 laps, depth  4
> [cxx]         |_fibonacci[39]                                             : 4.300e-01 sec user, 1 laps, depth  5 (exclusive:   0.0%)
> [cxx]           |_fibonacci[38]                                           : 2.700e-01 sec user, 1 laps, depth  6 (exclusive:   0.0%)
> [cxx]             |_fibonacci[37]                                         : 1.700e-01 sec user, 1 laps, depth  7 (exclusive:   0.0%)
> [cxx]               |_fibonacci[36]                                       : 1.100e-01 sec user, 1 laps, depth  8 (exclusive:   0.0%)
> [cxx]                 |_fibonacci[35]                                     : 7.000e-02 sec user, 1 laps, depth  9 (exclusive:   0.0%)
> [cxx]                   |_fibonacci[34]                                   : 4.000e-02 sec user, 1 laps, depth 10 (exclusive:   0.0%)
> [cxx]                     |_fibonacci[33]                                 : 3.000e-02 sec user, 1 laps, depth 11 (exclusive:   0.0%)
> [cxx]                       |_fibonacci[32]                               : 2.000e-02 sec user, 1 laps, depth 12 (exclusive:   0.0%)
> [cxx]                         |_fibonacci[31]                             : 1.000e-02 sec user, 1 laps, depth 13 (exclusive:   0.0%)
> [cxx]                           |_fibonacci[30]                           : 1.000e-02 sec user, 1 laps, depth 14 (exclusive:   0.0%)
> [cxx]                             |_fibonacci[29]                         : 1.000e-02 sec user, 1 laps, depth 15 (exclusive:   0.0%)
> [cxx]                               |_fibonacci[28]                       : 1.000e-02 sec user, 1 laps, depth 16 (exclusive:   0.0%)
> [cxx]                                 |_fibonacci[27]                     : 1.000e-02 sec user, 1 laps, depth 17 (exclusive:   0.0%)
> [cxx]                                   |_fibonacci[26]                   : 1.000e-02 sec user, 1 laps, depth 18 (exclusive:   0.0%)
> [cxx]                                     |_fibonacci[25]                 : 1.000e-02 sec user, 1 laps, depth 19 (exclusive:   0.0%)
> [cxx]                                       |_fibonacci[24]               : 0.000e+00 sec user, 1 laps, depth 20 (exclusive:   0.0%)
> [cxx]                                         |_fibonacci[23]             : 0.000e+00 sec user, 1 laps, depth 21 (exclusive:   0.0%)
> [cxx]                                           |_fibonacci[22]           : 0.000e+00 sec user, 1 laps, depth 22 (exclusive:   0.0%)
> [cxx]                                             |_fibonacci[21]         : 0.000e+00 sec user, 1 laps, depth 23 (exclusive:   0.0%)
> [cxx]                                               |_fibonacci[20]       : 0.000e+00 sec user, 1 laps, depth 24 (exclusive:   0.0%)
> [cxx]                                                 |_fibonacci[19]     : 0.000e+00 sec user, 1 laps, depth 25 (exclusive:   0.0%)
> [cxx]                                                   |_fibonacci[18]   : 0.000e+00 sec user, 1 laps, depth 26 (exclusive:   0.0%)
> [cxx]                                                     |_fibonacci[17] : 0.000e+00 sec user, 1 laps, depth 27
> [cxx]                                                   |_fibonacci[17]   : 0.000e+00 sec user, 1 laps, depth 26
> [cxx]                                                 |_fibonacci[18]     : 0.000e+00 sec user, 1 laps, depth 25 (exclusive:   0.0%)
> [cxx]                                                   |_fibonacci[17]   : 0.000e+00 sec user, 1 laps, depth 26
> [cxx]                                               |_fibonacci[19]       : 0.000e+00 sec user, 1 laps, depth 24 (exclusive:   0.0%)
> [cxx]                                                 |_fibonacci[18]     : 0.000e+00 sec user, 1 laps, depth 25 (exclusive:   0.0%)
> [cxx]                                                   |_fibonacci[17]   : 0.000e+00 sec user, 1 laps, depth 26
> [cxx]                                                 |_fibonacci[17]     : 0.000e+00 sec user, 1 laps, depth 25
> [cxx]                                             |_fibonacci[20]         : 0.000e+00 sec user, 1 laps, depth 23 (exclusive:   0.0%)
> [cxx]                                               |_fibonacci[19]       : 0.000e+00 sec user, 1 laps, depth 24 (exclusive:   0.0%)
> [cxx]                                                 |_fibonacci[18]     : 0.000e+00 sec user, 1 laps, depth 25 (exclusive:   0.0%)
> [cxx]                                                   |_fibonacci[17]   : 0.000e+00 sec user, 1 laps, depth 26
> [cxx]                                                 |_fibonacci[17]     : 0.000e+00 sec user, 1 laps, depth 25
> [cxx]                                               |_fibonacci[18]       : 0.000e+00 sec user, 1 laps, depth 24 (exclusive:   0.0%)
> [cxx]                                                 |_fibonacci[17]     : 0.000e+00 sec user, 1 laps, depth 25
> [cxx]                                           |_fibonacci[21]           : 0.000e+00 sec user, 1 laps, depth 22 (exclusive:   0.0%)
> [cxx]                                             |_fibonacci[20]         : 0.000e+00 sec user, 1 laps, depth 23 (exclusive:   0.0%)
> [cxx]                                               |_fibonacci[19]       : 0.000e+00 sec user, 1 laps, depth 24 (exclusive:   0.0%)
> [cxx]                                                 |_fibonacci[18]     : 0.000e+00 sec user, 1 laps, depth 25 (exclusive:   0.0%)
...
```

> Output is truncated and/or not shown for all components (`real_clock`, `system_clock`, `user_clock`, and `trip_count`)

## Measuring Overhead of 27 unique measurements with 2,056,912 measurements

After making the modifications to re-use measurements, `fibonacci(43, 16)` produces only 27 unique measurements and the overhead is reduced by ~15%:

```shell
Report from 2056912 total measurements:
	> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':86 : 1.365e+00 sec real [laps: 1]
	> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':86 : 3.103e+00 sec real [laps: 1]
	> [cxx] timing difference                                    : 1.738e+00 sec real
	> [cxx] average overhead per timer                           : 8.450e-07 sec real

> [cxx] main[./test_cxx_overhead]@'test_cxx_overhead.cpp':132             : 4.841e+00 sec real,      1 laps, depth  0 (exclusive:   0.0%)
> [cxx] |_run [with timing = false]@'test_cxx_overhead.cpp':86            : 1.338e+00 sec real,      1 laps, depth  1
> [cxx] |_run [with timing =  true]@'test_cxx_overhead.cpp':86            : 3.503e+00 sec real,      1 laps, depth  1 (exclusive:   0.0%)
> [cxx]   |_fibonacci                                                     : 3.502e+00 sec real,      1 laps, depth  2 (exclusive:   0.0%)
> [cxx]     |_fibonacci                                                   : 3.502e+00 sec real,      2 laps, depth  3 (exclusive:   0.0%)
> [cxx]       |_fibonacci                                                 : 3.502e+00 sec real,      4 laps, depth  4 (exclusive:   0.0%)
> [cxx]         |_fibonacci                                               : 3.502e+00 sec real,      8 laps, depth  5 (exclusive:   0.0%)
> [cxx]           |_fibonacci                                             : 3.502e+00 sec real,     16 laps, depth  6 (exclusive:   0.0%)
> [cxx]             |_fibonacci                                           : 3.502e+00 sec real,     32 laps, depth  7 (exclusive:   0.0%)
> [cxx]               |_fibonacci                                         : 3.502e+00 sec real,     64 laps, depth  8 (exclusive:   0.0%)
> [cxx]                 |_fibonacci                                       : 3.501e+00 sec real,    128 laps, depth  9 (exclusive:   0.0%)
> [cxx]                   |_fibonacci                                     : 3.500e+00 sec real,    256 laps, depth 10 (exclusive:   0.1%)
> [cxx]                     |_fibonacci                                   : 3.498e+00 sec real,    512 laps, depth 11 (exclusive:   0.1%)
> [cxx]                       |_fibonacci                                 : 3.495e+00 sec real,   1024 laps, depth 12 (exclusive:   0.2%)
> [cxx]                         |_fibonacci                               : 3.487e+00 sec real,   2048 laps, depth 13 (exclusive:   0.4%)
> [cxx]                           |_fibonacci                             : 3.472e+00 sec real,   4096 laps, depth 14 (exclusive:   0.9%)
> [cxx]                             |_fibonacci                           : 3.442e+00 sec real,   8192 laps, depth 15 (exclusive:   1.7%)
> [cxx]                               |_fibonacci                         : 3.384e+00 sec real,  16369 laps, depth 16 (exclusive:   3.5%)
> [cxx]                                 |_fibonacci                       : 3.266e+00 sec real,  32192 laps, depth 17 (exclusive:   7.0%)
> [cxx]                                   |_fibonacci                     : 3.036e+00 sec real,  58651 laps, depth 18 (exclusive:  13.7%)
> [cxx]                                     |_fibonacci                   : 2.622e+00 sec real,  89846 laps, depth 19 (exclusive:  23.8%)
> [cxx]                                       |_fibonacci                 : 1.999e+00 sec real, 106762 laps, depth 20 (exclusive:  36.3%)
> [cxx]                                         |_fibonacci               : 1.273e+00 sec real,  94184 laps, depth 21 (exclusive:  49.2%)
> [cxx]                                           |_fibonacci             : 6.464e-01 sec real,  60460 laps, depth 22 (exclusive:  61.1%)
> [cxx]                                             |_fibonacci           : 2.516e-01 sec real,  27896 laps, depth 23 (exclusive:  71.1%)
> [cxx]                                               |_fibonacci         : 7.263e-02 sec real,   9109 laps, depth 24 (exclusive:  79.5%)
> [cxx]                                                 |_fibonacci       : 1.488e-02 sec real,   2048 laps, depth 25 (exclusive:  86.2%)
> [cxx]                                                   |_fibonacci     : 2.052e-03 sec real,    301 laps, depth 26 (exclusive:  91.4%)
> [cxx]                                                     |_fibonacci   : 1.774e-04 sec real,     26 laps, depth 27 (exclusive:  95.2%)
> [cxx]                                                       |_fibonacci : 8.580e-06 sec real,      1 laps, depth 28

> [cxx] main[./test_cxx_overhead]@'test_cxx_overhead.cpp':132           : 4.070e+00 sec user,      1 laps, depth  0 (exclusive:  33.7%)
> [cxx] |_fibonacci                                                     : 2.700e+00 sec user,      1 laps, depth  1 (exclusive:   0.0%)
> [cxx]   |_fibonacci                                                   : 2.700e+00 sec user,      2 laps, depth  2 (exclusive:   0.0%)
> [cxx]     |_fibonacci                                                 : 2.700e+00 sec user,      4 laps, depth  3 (exclusive:   0.0%)
> [cxx]       |_fibonacci                                               : 2.700e+00 sec user,      8 laps, depth  4 (exclusive:   0.0%)
> [cxx]         |_fibonacci                                             : 2.700e+00 sec user,     16 laps, depth  5 (exclusive:   0.0%)
> [cxx]           |_fibonacci                                           : 2.700e+00 sec user,     32 laps, depth  6 (exclusive:   0.0%)
> [cxx]             |_fibonacci                                         : 2.700e+00 sec user,     64 laps, depth  7 (exclusive:   0.0%)
> [cxx]               |_fibonacci                                       : 2.700e+00 sec user,    128 laps, depth  8 (exclusive:   0.0%)
> [cxx]                 |_fibonacci                                     : 2.700e+00 sec user,    256 laps, depth  9 (exclusive:   0.4%)
> [cxx]                   |_fibonacci                                   : 2.690e+00 sec user,    512 laps, depth 10 (exclusive:   0.0%)
> [cxx]                     |_fibonacci                                 : 2.690e+00 sec user,   1024 laps, depth 11 (exclusive:   0.4%)
> [cxx]                       |_fibonacci                               : 2.680e+00 sec user,   2048 laps, depth 12 (exclusive:   0.0%)
> [cxx]                         |_fibonacci                             : 2.680e+00 sec user,   4096 laps, depth 13 (exclusive:   0.7%)
> [cxx]                           |_fibonacci                           : 2.660e+00 sec user,   8192 laps, depth 14 (exclusive:   1.1%)
> [cxx]                             |_fibonacci                         : 2.630e+00 sec user,  16369 laps, depth 15 (exclusive:   1.5%)
> [cxx]                               |_fibonacci                       : 2.590e+00 sec user,  32192 laps, depth 16 (exclusive:   5.4%)
> [cxx]                                 |_fibonacci                     : 2.450e+00 sec user,  58651 laps, depth 17 (exclusive:  11.8%)
> [cxx]                                   |_fibonacci                   : 2.160e+00 sec user,  89846 laps, depth 18 (exclusive:  22.7%)
> [cxx]                                     |_fibonacci                 : 1.670e+00 sec user, 106762 laps, depth 19 (exclusive:  35.9%)
> [cxx]                                       |_fibonacci               : 1.070e+00 sec user,  94184 laps, depth 20 (exclusive:  45.8%)
> [cxx]                                         |_fibonacci             : 5.800e-01 sec user,  60460 laps, depth 21 (exclusive:  50.0%)
> [cxx]                                           |_fibonacci           : 2.900e-01 sec user,  27896 laps, depth 22 (exclusive:  75.9%)
> [cxx]                                             |_fibonacci         : 7.000e-02 sec user,   9109 laps, depth 23 (exclusive:  85.7%)
> [cxx]                                               |_fibonacci       : 1.000e-02 sec user,   2048 laps, depth 24 (exclusive: 100.0%)
> [cxx]                                                 |_fibonacci     : 0.000e+00 sec user,    301 laps, depth 25 (exclusive:   0.0%)
> [cxx]                                                   |_fibonacci   : 0.000e+00 sec user,     26 laps, depth 26 (exclusive:   0.0%)
> [cxx]                                                     |_fibonacci : 0.000e+00 sec user,      1 laps, depth 27
```

> Output is truncated and/or not shown for all components (`real_clock`, `system_clock`, `user_clock`, and `trip_count`)

## Conclusion

Since TiMemory only records information of the functions explicitly specified, you can safely assume that unless
TiMemory is inserted into a function called `> 100,000` times, it won't be adding more than a second of runtime
to the function and **judicious use will probably have zero overhead**.

Therefore, there is a simple rule of thumb:
**do not insert a TiMemory components into simple functions that get invoked extremely frequently**.
