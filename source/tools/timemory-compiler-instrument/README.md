# timemory compiler instrumentation library

The timemory compiler instrumentation library provides the symbols required by the `-finstrument-functions` compiler flag supported by Clang and GCC.

## Description

The `-finstrument-functions` compiler flag inserts functions at the beginning and end of each _compiled_ function (does not apply to externally defined functions), respectively:

```cpp
void __cyg_profile_func_enter(void* this_fn, void* call_site);
void __cyg_profile_func_exit(void* this_fn, void* call_site);
```

The `timemory-compiler-instrument` library invokes the timemory analogues to the above functions
from the `timemory-compiler-instrument-base` library:

```cpp
void timemory_profile_func_enter(void* this_fn, void* call_site);
void timemory_profile_func_exit(void* this_fn, void* call_site);
```

By default, this library instruments a wall-clock timer but can be customized exclusively
via the `TIMEMORY_COMPILER_COMPONENTS` environment variable or synchronized with timemory
instrumentation present in the target code via `TIMEMORY_GLOBAL_COMPONENTS`.
Timemory compiler instrumentation is fully compatible with codes that are instrumented with timemory
but, in order to prevent expensive checks for recursion, the data is collected separately --
resulting in an entirely separate set of outputs.

## Usage

### Build

Timemory provides a `timemory::timemory-compiler-instrument` target in CMake which provides the necessary
libraries and the compiler flags for the best resolution of the instrumented function names.

> When compiling with `-finstrument-functions`, it is highly recommended to also compile with:
> `-g -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls`.
> These flags are automatically provided when using CMake.

#### CMake Example

```cmake
#########################
#
#   C or C++ codes
#
#########################

# Source is not instrumented with timemory
find_package(timemory COMPONENTS compiler-instrumentation)

# Source is instrumented with the timemory library interface
find_package(timemory COMPONENTS c cxx compiler-instrumentation)

#########################
#
#   C++ codes
#
#########################

# Source is instrumented with timemory library and/or template interface
find_package(timemory COMPONENTS cxx compiler-instrumentation)

# Source is instrumented with timemory template interface in header-only mode
find_package(timemory COMPONENTS headers compiler-instrumentation)

# Source is instrumented with timemory macros and wants to disable all macros
find_package(timemory COMPONENTS cxx disable compiler-instrumentation)

# Source is instrumented with timemory and wants (in-source, not compiler)
# instrumentation disabled by default at runtime
find_package(timemory COMPONENTS cxx default-disabled compiler-instrumentation)

#########################
#
#   Using one of above
#
#########################

# some library
add_library(foo SHARED foo.c foo.cpp)

# add in instrumentation, PUBLIC so it propagates to downstream targets
target_link_libraries(foo PUBLIC timemory::timemory)

# some executable
add_executable(bar bar.c)

# add in instrumentation to executable via timemory::timemory publicly linked to foo
target_link_libraries(bar PRIVATE foo)
```

#### Makefile Example

```makefile
CC = gcc
CFLAGS += -finstrument-functions -g -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls
LIBS += timemory-compiler-instrument
OBJ = foo.o

%.o: %.c $(DEPS)
    $(CC) -c -o $@ $< $(CFLAGS)

bar: $(OBJ) bar.c
    $(CC) -o $@ $^ $(CFLAGS) $(LIBS)
```

### Restricting Instrumentation

Compiler instrumentation will generate alot of profiling information -- instrumentation is also done for functions expanded inline in other functions.
In order to reduce this, refer to the compiler documentation for the additional flags: [-finstrument-functions-exclude-file-list=file,file,...](https://gcc.gnu.org/onlinedocs/gcc-4.4.4/gcc/Code-Gen-Options.html) and
[-finstrument-functions-exclude-function-list=sym,sym,...](https://gcc.gnu.org/onlinedocs/gcc-4.4.4/gcc/Code-Gen-Options.html).

## Examples

### Code

```cpp
#include <chrono>
#include <iostream>
#include <mutex>

inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

inline void
consume(long n)
{
    using mutex_t = std::mutex;
    using lock_t  = std::unique_lock<mutex_t>;
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

int
main(int argc, char** argv)
{
    long nwait = (argc > 1) ? atol(argv[1]) : 12;
    int  nitr  = (argc > 2) ? atoi(argv[2]) : 10;

    for(int i = 0; i < nitr; ++i)
    {
        consume(fibonacci(nwait));
    }

    return 0;
}
```

### Expected Output

```console
$ cmake -DTIMEMORY_BUILD_EXAMPLES=ON -DTIMEMORY_BUILD_TOOLS=ON -DTIMEMORY_BUILD_COMPILER_INSTRUMENTATION=ON ...
$ make ex_compiler_instrument
$ TIMEMORY_MAX_WIDTH=60 ./ex_compiler_instrument
[12301]> timemory-compiler-instrument will close after 'main' returns
[12301]> timemory-compiler-instrument: 47 results
[wall]|0> Outputting 'timemory-compiler-instrumentation-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-compiler-instrumentation-output/wall.json'...
[wall]|0> Outputting 'timemory-compiler-instrumentation-output/wall.tree.json'...
[wall]|0> Outputting 'timemory-compiler-instrumentation-output/wall.txt'...

[12301]> timemory-compiler-instrument: finalizing...
$ cat timemory-compiler-instrumentation-output/wall.txt
|--------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                        REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                        |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|
|                            LABEL                             | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|--------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> main                                                     |      1 |      0 | wall   | sec    |  1.544 |  1.544 |  1.544 |  1.544 |  0.000 |    0.0 |
| >>> |_fibonacci(long)                                        |     10 |      1 | wall   | sec    |  0.092 |  0.009 |  0.008 |  0.011 |  0.001 |    0.4 |
| >>>   |_fibonacci(long)                                      |     20 |      2 | wall   | sec    |  0.092 |  0.005 |  0.003 |  0.007 |  0.001 |    0.8 |
| >>>     |_fibonacci(long)                                    |     40 |      3 | wall   | sec    |  0.091 |  0.002 |  0.001 |  0.004 |  0.001 |    1.6 |
| >>>       |_fibonacci(long)                                  |     80 |      4 | wall   | sec    |  0.089 |  0.001 |  0.000 |  0.002 |  0.001 |    3.2 |
| >>>         |_fibonacci(long)                                |    160 |      5 | wall   | sec    |  0.086 |  0.001 |  0.000 |  0.001 |  0.000 |    6.7 |
| >>>           |_fibonacci(long)                              |    320 |      6 | wall   | sec    |  0.081 |  0.000 |  0.000 |  0.001 |  0.000 |   14.1 |
| >>>             |_fibonacci(long)                            |    640 |      7 | wall   | sec    |  0.069 |  0.000 |  0.000 |  0.001 |  0.000 |   30.5 |
| >>>               |_fibonacci(long)                          |   1140 |      8 | wall   | sec    |  0.048 |  0.000 |  0.000 |  0.000 |  0.000 |   53.0 |
| >>>                 |_fibonacci(long)                        |   1280 |      9 | wall   | sec    |  0.023 |  0.000 |  0.000 |  0.000 |  0.000 |   71.8 |
| >>>                   |_fibonacci(long)                      |    740 |     10 | wall   | sec    |  0.006 |  0.000 |  0.000 |  0.000 |  0.000 |   84.5 |
| >>>                     |_fibonacci(long)                    |    200 |     11 | wall   | sec    |  0.001 |  0.000 |  0.000 |  0.000 |  0.000 |   93.4 |
| >>>                       |_fibonacci(long)                  |     20 |     12 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>> |_consume(long)                                          |     10 |      1 | wall   | sec    |  1.452 |  0.145 |  0.145 |  0.146 |  0.000 |   13.1 |
| >>>   |_std::__1::mutex::mutex()                             |     10 |      2 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   84.7 |
| >>>     |_std::__1::mutex::mutex()                           |     10 |      3 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::unique_lock<std::__1::mutex>::unique_lo... |     10 |      2 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   45.2 |
| >>>     |_std::__1::unique_lock<std::__1::mutex>::unique_... |     10 |      3 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   86.9 |
| >>>       |_std::__1::mutex* std::__1::addressof<std::__1... |     10 |      4 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::unique_lock<std::__1::mutex>::unique_lo... |     10 |      2 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   45.5 |
| >>>     |_std::__1::unique_lock<std::__1::mutex>::unique_... |     10 |      3 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   85.7 |
| >>>       |_std::__1::mutex* std::__1::addressof<std::__1... |     10 |      4 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::chrono::duration<long long, std::__1::r... |   2610 |      2 | wall   | sec    |  0.061 |  0.000 |  0.000 |  0.000 |  0.000 |   87.1 |
| >>>     |_std::__1::chrono::duration<long long, std::__1:... |   2610 |      3 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::chrono::time_point<std::__1::chrono::st... |   2610 |      2 | wall   | sec    |  0.864 |  0.000 |  0.000 |  0.001 |  0.000 |   16.9 |
| >>>     |_std::__1::chrono::time_point<std::__1::chrono::... |   2610 |      3 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>     |_std::__1::common_type<std::__1::chrono::duratio... |   2610 |      3 | wall   | sec    |  0.650 |  0.000 |  0.000 |  0.001 |  0.000 |   30.0 |
| >>>       |_std::__1::chrono::duration<long long, std::__... |   5220 |      4 | wall   | sec    |  0.017 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>       |_std::__1::chrono::duration<long long, std::__... |   2610 |      4 | wall   | sec    |  0.375 |  0.000 |  0.000 |  0.000 |  0.000 |   13.3 |
| >>>         |_std::__1::chrono::duration<long long, std::... |   2610 |      5 | wall   | sec    |  0.325 |  0.000 |  0.000 |  0.000 |  0.000 |   30.5 |
| >>>           |_std::__1::enable_if<__is_duration<std::__... |   2610 |      6 | wall   | sec    |  0.218 |  0.000 |  0.000 |  0.000 |  0.000 |   22.8 |
| >>>             |_std::__1::chrono::__duration_cast<std::... |   2610 |      7 | wall   | sec    |  0.168 |  0.000 |  0.000 |  0.000 |  0.000 |   58.5 |
| >>>               |_std::__1::chrono::duration<long long,... |   2610 |      8 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>               |_std::__1::chrono::duration<long long,... |   2610 |      8 | wall   | sec    |  0.062 |  0.000 |  0.000 |  0.000 |  0.000 |   86.4 |
| >>>                 |_std::__1::chrono::duration<long lon... |   2610 |      9 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>           |_std::__1::chrono::duration<long long, std... |   2610 |      6 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>       |_std::__1::chrono::duration<long long, std::__... |   2610 |      4 | wall   | sec    |  0.063 |  0.000 |  0.000 |  0.000 |  0.000 |   86.4 |
| >>>         |_std::__1::chrono::duration<long long, std::... |   2610 |      5 | wall   | sec    |  0.009 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>     |_std::__1::chrono::time_point<std::__1::chrono::... |   2610 |      3 | wall   | sec    |  0.060 |  0.000 |  0.000 |  0.000 |  0.000 |   86.9 |
| >>>       |_std::__1::chrono::time_point<std::__1::chrono... |   2610 |      4 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_bool std::__1::chrono::operator<<std::__1::chrono... |   2610 |      2 | wall   | sec    |  0.327 |  0.000 |  0.000 |  0.000 |  0.000 |   44.8 |
| >>>     |_std::__1::chrono::time_point<std::__1::chrono::... |   5220 |      3 | wall   | sec    |  0.016 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>     |_bool std::__1::chrono::operator<<long long, std... |   2610 |      3 | wall   | sec    |  0.165 |  0.000 |  0.000 |  0.000 |  0.000 |   29.7 |
| >>>       |_std::__1::chrono::__duration_lt<std::__1::chr... |   2610 |      4 | wall   | sec    |  0.116 |  0.000 |  0.000 |  0.000 |  0.000 |   86.5 |
| >>>         |_std::__1::chrono::duration<long long, std::... |   5220 |      5 | wall   | sec    |  0.016 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::unique_lock<std::__1::mutex>::try_lock()   |   2600 |      2 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::unique_lock<std::__1::mutex>::~unique_l... |     20 |      2 | wall   | sec    |  0.001 |  0.000 |  0.000 |  0.000 |  0.000 |   86.8 |
| >>>     |_std::__1::unique_lock<std::__1::mutex>::~unique... |     20 |      3 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|
```