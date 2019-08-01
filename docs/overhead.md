# Overhead

Analysis on a fibonacci calculation determined that one TiMemory "component" adds an average overhead of 3 microseconds (`0.000003 s`) when the component is being inserted into call-graph for the first time. Once a component exists in
the call-graph, the overhead is approximately 1.25 microseconds. For example, in the following:

```c++
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_AUTO_TUPLE(tim::auto_tuple<real_clock>, "[", n, "]");
        return (n < 2) ? n : (fibonacci(n - 2, cutoff) + fibonacci(n - 1, cutoff));
    }
    return fibonacci(n); // standard fibonacci (no timers)
}
```

every single instance is unique and the overhead fibonacci(43, 16) produces 514191 unique timers:

```shell
> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110             : 1.667e+00 sec real, 1 laps
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110             : 2.782e+00 sec real, 1 laps
> [cxx] fibonacci[43]                                                     : 2.782e+00 sec real, 1 laps
> [cxx] |_fibonacci[41]                                                   : 1.033e+00 sec real, 1 laps
> [cxx]   |_fibonacci[39]                                                 : 3.868e-01 sec real, 1 laps
> [cxx]     |_fibonacci[37]                                               : 1.467e-01 sec real, 1 laps
> [cxx]       |_fibonacci[35]                                             : 5.519e-02 sec real, 1 laps
> [cxx]         |_fibonacci[33]                                           : 2.151e-02 sec real, 1 laps
> [cxx]           |_fibonacci[31]                                         : 8.197e-03 sec real, 1 laps
> [cxx]             |_fibonacci[29]                                       : 3.063e-03 sec real, 1 laps
> [cxx]               |_fibonacci[27]                                     : 1.148e-03 sec real, 1 laps
> [cxx]                 |_fibonacci[25]                                   : 4.421e-04 sec real, 1 laps
> [cxx]                   |_fibonacci[23]                                 : 1.718e-04 sec real, 1 laps
> [cxx]                     |_fibonacci[21]                               : 6.159e-05 sec real, 1 laps
> [cxx]                       |_fibonacci[19]                             : 2.116e-05 sec real, 1 laps
> [cxx]                         |_fibonacci[17]                           : 6.281e-06 sec real, 1 laps
> [cxx]                         |_fibonacci[18]                           : 1.172e-05 sec real, 1 laps
> [cxx]                           |_fibonacci[17]                         : 6.146e-06 sec real, 1 laps
> [cxx]                       |_fibonacci[20]                             : 3.766e-05 sec real, 1 laps
> [cxx]                         |_fibonacci[18]                           : 1.156e-05 sec real, 1 laps
> [cxx]                           |_fibonacci[17]                         : 6.183e-06 sec real, 1 laps
> [cxx]                         |_fibonacci[19]                           : 2.318e-05 sec real, 1 laps
> [cxx]                           |_fibonacci[17]                         : 6.166e-06 sec real, 1 laps
> [cxx]                           |_fibonacci[18]                         : 1.319e-05 sec real, 1 laps
> [cxx]                             |_fibonacci[17]                       : 6.180e-06 sec real, 1 laps
> [cxx]                     |_fibonacci[22]                               : 1.072e-04 sec real, 1 laps

...

> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110 : 1.667e+00 sec real [laps: 1]
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110 : 2.782e+00 sec real [laps: 1]
> [cxx] timing difference                                     : 1.115e+00 sec real
> [cxx] average overhead per timer                            : 2.168e-06 sec real
```

However, the following produces only 27 unique timers:

```c++
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_AUTO_TUPLE(tim::auto_tuple<real_clock>, "");
        return (n < 2) ? n : (fibonacci(n - 2, cutoff) + fibonacci(n - 1, cutoff));
    }
    return fibonacci(n); // standard fibonacci (no timers)
}
```

and the overhead is much smaller:

```shell
> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110         : 2.220e+00 sec real, 1 laps
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110         : 2.832e+00 sec real, 1 laps
> [cxx] fibonacci                                                     : 2.832e+00 sec real, 1 laps
> [cxx] |_fibonacci                                                   : 2.832e+00 sec real, 2 laps
> [cxx]   |_fibonacci                                                 : 2.832e+00 sec real, 4 laps
> [cxx]     |_fibonacci                                               : 2.832e+00 sec real, 8 laps
> [cxx]       |_fibonacci                                             : 2.832e+00 sec real, 16 laps
> [cxx]         |_fibonacci                                           : 2.832e+00 sec real, 32 laps
> [cxx]           |_fibonacci                                         : 2.832e+00 sec real, 64 laps
> [cxx]             |_fibonacci                                       : 2.832e+00 sec real, 128 laps
> [cxx]               |_fibonacci                                     : 2.831e+00 sec real, 256 laps
> [cxx]                 |_fibonacci                                   : 2.831e+00 sec real, 512 laps
> [cxx]                   |_fibonacci                                 : 2.830e+00 sec real, 1024 laps
> [cxx]                     |_fibonacci                               : 2.828e+00 sec real, 2048 laps
> [cxx]                       |_fibonacci                             : 2.824e+00 sec real, 4096 laps
> [cxx]                         |_fibonacci                           : 2.815e+00 sec real, 8192 laps
> [cxx]                           |_fibonacci                         : 2.798e+00 sec real, 16369 laps
> [cxx]                             |_fibonacci                       : 2.761e+00 sec real, 32192 laps
> [cxx]                               |_fibonacci                     : 2.660e+00 sec real, 58651 laps
> [cxx]                                 |_fibonacci                   : 2.425e+00 sec real, 89846 laps
> [cxx]                                   |_fibonacci                 : 1.977e+00 sec real, 106762 laps
> [cxx]                                     |_fibonacci               : 1.355e+00 sec real, 94184 laps
> [cxx]                                       |_fibonacci             : 7.419e-01 sec real, 60460 laps
> [cxx]                                         |_fibonacci           : 3.124e-01 sec real, 27896 laps
> [cxx]                                           |_fibonacci         : 9.630e-02 sec real, 9109 laps
> [cxx]                                             |_fibonacci       : 2.064e-02 sec real, 2048 laps
> [cxx]                                               |_fibonacci     : 2.952e-03 sec real, 301 laps
> [cxx]                                                 |_fibonacci   : 2.318e-04 sec real, 26 laps
> [cxx]                                                   |_fibonacci : 8.503e-06 sec real, 1 laps

> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110 : 2.220e+00 sec real [laps: 1]
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110 : 2.832e+00 sec real [laps: 1]
> [cxx] timing difference                                     : 6.116e-01 sec real
> [cxx] average overhead per timer                            : 1.189e-06 sec real
```

The exact performance is specific to the machine and the overhead for a particular machine can be calculated by running the `test_cxx_overhead` example.

Since TiMemory only records information of the functions explicitly specified, you can safely assume that unless
TiMemory is inserted into a function called `> 100,000` times, it won't be adding more than a second of runtime
to the function. Therefore, there is a simple rule of thumb: **don't insert a TiMemory auto-tuple into very simple functions
that get called very frequently**.
