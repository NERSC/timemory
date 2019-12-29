# Getting Started

For C++ projects, basic functionality simply requires including the header path,
e.g. `-I/usr/local/include` if `timemory.hpp` is in `/usr/local/include/timemory/timemory.hpp`.
However, this will not enable additional capabilities such as PAPI, CUPTI, CUDA kernel timing,
extern templates, etc.

In C++ and Python, timemory can be added in one line of code (once the type is declared):

## Supported Languages

### C++

```cpp
using auto_tuple_t = tim::auto_tuple<wall_clock, cpu_clock, peak_rss, user_tuple_bundle>;
void some_function()
{
    TIMEMORY_MARKER(auto_tuple_t, "");
    // ...
}

void init()
{
    // insert the cpu_util component at runtime
    user_tuple_bundle::configure<cpu_util>();
}
```

### Python

```python
from timemory.util import auto_timer

@auto_timer()
def some_function():
    with auto_timer():
        # ...
```

### C

In C, timemory requires only two lines of code

```c
void* timer = TIMEMORY_MARKER("", WALL_CLOCK, SYS_CLOCK, USER_CLOCK, PEAK_RSS, CUDA_EVENT);
// ...
FREE_TIMEMORY_MARKER(timer);
```

When the application terminates, output to text and JSON is automated or controlled via
environment variables.

## Environment Controls

| Environment Variable              | C++ Accessor                           | Value Type     | Default                | Description                                                                                    |
| --------------------------------- | -------------------------------------- | -------------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| TIMEMORY_SUPPRESS_PARSING         | `settings::suppress_parsing()`         | bool           | OFF                    | Disable parsing environment variables for settings                                             |
| TIMEMORY_ENABLED                  | `settings::enabled()`                  | bool           | ON                     | Enable/disable timemory markers                                                                |
| TIMEMORY_AUTO_OUTPUT              | `settings::auto_output()`              | bool           | ON                     | Generate output when application exits                                                         |
| TIMEMORY_COUT_OUTPUT              | `settings::cout_output()`              | bool           | ON                     | Enable/disable output to terminal                                                              |
| TIMEMORY_FILE_OUTPUT              | `settings::file_output()`              | bool           | ON                     | Enable/disable output to files                                                                 |
| TIMEMORY_TEXT_OUTPUT              | `settings::text_output()`              | bool           | ON                     | Enable/disable text file output                                                                |
| TIMEMORY_JSON_OUTPUT              | `settings::json_output()`              | bool           | OFF                    | Enable/disable JSON file output                                                                |
| TIMEMORY_DART_OUTPUT              | `settings::dart_output()`              | bool           | OFF                    | Enable/disable DART measurements (CTest + CDash)                                               |
| TIMEMORY_TIME_OUTPUT              | `settings::time_output()`              | bool           | OFF                    | Enable/disable output folders based on timestamp                                               |
| TIMEMORY_VERBOSE                  | `settings::verbose()`                  | int            | 0                      | Enable/disable verbosity                                                                       |
| TIMEMORY_DEBUG                    | `settings::debug()`                    | bool           | OFF                    | Enable/disable debug output                                                                    |
| TIMEMORY_BANNER                   | `settings::banner()`                   | bool           | ON                     | Enable/disable banner at initialization and finalization                                       |
| TIMEMORY_FLAT_PROFILE             | `settings::flat_profile()`             | bool           | OFF                    | Enable/disable marker nesting                                                                  |
| TIMEMORY_COLLAPSE_THREADS         | `settings::collapse_threads()`         | bool           | ON                     | Enable/disable combining thread-local data                                                     |
| TIMEMORY_MAX_DEPTH                | `settings::max_depth()`                | unsigned short | 65535                  |                                                                                                |
| TIMEMORY_TIME_FORMAT              | `settings::time_format()`              | string         | `"%F_%I.%M_%p"`        | See [strftime](http://man7.org/linux/man-pages/man3/strftime.3.html)                           |
| TIMEMORY_PRECISION                | `settings::precision()`                | short          | component-specific     | Output precision                                                                               |
| TIMEMORY_WIDTH                    | `settings::width()`                    | short          | component-specific     | Output value width                                                                             |
| TIMEMORY_SCIENTIFIC               | `settings::scientific()`               | bool           | OFF                    | Use scientific notation globally                                                               |
| TIMEMORY_TIMING_PRECISION         | `settings::timing_precision()`         | short          | component-specific     |                                                                                                |
| TIMEMORY_TIMING_WIDTH             | `settings::timing_width()`             | short          | component-specific     |                                                                                                |
| TIMEMORY_TIMING_UNITS             | `settings::timing_units()`             | string         | seconds                |                                                                                                |
| TIMEMORY_TIMING_SCIENTIFIC        | `settings::timing_scientific()`        | bool           | OFF                    | Use scientific notation for timing types                                                       |
| TIMEMORY_MEMORY_PRECISION         | `settings::memory_precision()`         | short          | component-specific     |                                                                                                |
| TIMEMORY_MEMORY_WIDTH             | `settings::memory_width()`             | short          | component-specific     |                                                                                                |
| TIMEMORY_MEMORY_UNITS             | `settings::memory_units()`             | string         | MB                     |                                                                                                |
| TIMEMORY_MEMORY_SCIENTIFIC        | `settings::memory_scientific()`        | bool           | OFF                    | Use scientific notation for memory types                                                       |
| TIMEMORY_MPI_INIT                 | `settings::mpi_init()`                 | bool           | ON                     | `timemory_init(int*, char***)` initialized MPI                                                 |
| TIMEMORY_MPI_FINALIZE             | `settings::mpi_finalize()`             | bool           | ON                     | `timemory_finalize()` finalizes MPI                                                            |
| TIMEMORY_MPI_THREAD               | `settings::mpi_thread()`               | bool           | ON                     | Use `MPI_Thread_init` instead of `MPI_Init` if timemory initializes MPI                        |
| TIMEMORY_MPI_THREAD_TYPE          | `settings::mpi_thread_type()`          | string         | `""`                   | `MPI_Thread_init` type: `"single"`, `"serialized"`, `"funneled"`, `"multiple"`                 |
| TIMEMORY_MPI_OUTPUT_PER_RANK      | `settings::mpi_output_per_rank()`      | bool           | OFF                    |                                                                                                |
| TIMEMORY_MPI_OUTPUT_PER_NODE      | `settings::mpi_output_per_node()`      | bool           | OFF                    |                                                                                                |
| TIMEMORY_OUTPUT_PATH              | `settings::output_path()`              | string         | `"timemory-output/"`   | Output folder path                                                                             |
| TIMEMORY_OUTPUT_PREFIX            | `settings::output_prefix()`            | string         | `""`                   | Prefix for output files                                                                        |
| TIMEMORY_DART_TYPE                | `settings::dart_type()`                | string         | `""`                   | Only echo DART measurements for components with this label                                     |
| TIMEMORY_DART_COUNT               | `settings::dart_count()`               | unsigned long  | 1                      | Only echo N measurements per component                                                         |
| TIMEMORY_DART_LABEL               | `settings::dart_label()`               | unsigned long  | 1                      | Use the component label instead of marker tag for DART measurements                            |
| TIMEMORY_CPU_AFFINITY             | `settings::cpu_affinity()`             | bool           | OFF                    | Enable/disable pinning threads to CPUs                                                         |
| TIMEMORY_PAPI_MULTIPLEXING        | `settings::papi_multiplexing()`        | bool           | ON                     | Enable/disable PAPI HW counter multiplexing                                                    |
| TIMEMORY_PAPI_FAIL_ON_ERROR       | `settings::papi_fail_on_error()`       | bool           | OFF                    | Terminate application when PAPI errors occur                                                   |
| TIMEMORY_PAPI_QUIET               | `settings::papi_quiet()`               | bool           | OFF                    | Suppress all warnings/errors                                                                   |
| TIMEMORY_PAPI_EVENTS              | `settings::papi_events()`              | string         | `""`                   | PAPI presets to count, e.g. `"PAPI_TOT_CYC,PAPI_LST_INS"`                                      |
| TIMEMORY_CUDA_EVENT_BATCH_SIZE    | `settings::cuda_event_batch_size()`    | unsigned long  | 5                      |                                                                                                |
| TIMEMORY_NVTX_MARKER_DEVICE_SYNC  | `settings::nvtx_marker_device_sync()`  | bool           | ON                     | NVTX markers call `cudaDeviceSynchronize()` when range is closed                               |
| TIMEMORY_CUPTI_ACTIVITY_LEVEL     | `settings::cupti_activity_level()`     | int            | 1                      | Levels of activity details                                                                     |
| TIMEMORY_CUPTI_ACTIVITY_KINDS     | `settings::cupti_activity_kinds()`     | string         | `""`                   | Specific activity record kinds to trace                                                        |
| TIMEMORY_CUPTI_EVENTS             | `settings::cupti_events()`             | string         | `""`                   | CUPTI events                                                                                   |
| TIMEMORY_CUPTI_METRICS            | `settings::cupti_metrics()`            | string         | `""`                   | CUPTI metrics                                                                                  |
| TIMEMORY_CUPTI_DEVICE             | `settings::cupti_device()`             | int            | 0                      | Device to enable CUPTI on                                                                      |
| TIMEMORY_ROOFLINE_MODE            | `settings::roofline_mode()`            | string         | op                     | `op` for CPU is FLOPs and HW counters for GPU. `ai` is load/stores for CPU and tracing for GPU |
| TIMEMORY_ROOFLINE_MODE_CPU        | `settings::roofline_mode_cpu()`        | string         | op                     |                                                                                                |
| TIMEMORY_ROOFLINE_MODE_GPU        | `settings::roofline_mode_gpu()`        | string         | op                     |                                                                                                |
| TIMEMORY_ROOFLINE_EVENTS_CPU      | `settings::roofline_events_cpu()`      | string         | `""`                   | Extra HW counter types for CPU roofline                                                        |
| TIMEMORY_ROOFLINE_EVENTS_GPU      | `settings::roofline_events_gpu()`      | string         | `""`                   | Extra HW counter types for GPU roofline                                                        |
| TIMEMORY_ROOFLINE_TYPE_LABELS     | `settings::roofline_type_labels()`     | bool           | OFF                    | Encode data types into labels                                                                  |
| TIMEMORY_ROOFLINE_TYPE_LABELS_CPU | `settings::roofline_type_labels_cpu()` | bool           | OFF                    |                                                                                                |
| TIMEMORY_ROOFLINE_TYPE_LABELS_GPU | `settings::roofline_type_labels_gpu()` | bool           | OFF                    |                                                                                                |
| TIMEMORY_INSTRUCTION_ROOFLINE     | `settings::instruction_roofline()`     | bool           | OFF                    | Enable instruction roofline support for GPU                                                    |
| TIMEMORY_ERT_NUM_THREADS          | `settings::ert_num_threads()`          | unsigned long  | 0                      | Number of threads to use for Empirical Roofline Toolkit                                        |
| TIMEMORY_ERT_NUM_THREADS_CPU      | `settings::ert_num_threads_cpu()`      | unsigned long  | # of cores             |                                                                                                |
| TIMEMORY_ERT_NUM_THREADS_GPU      | `settings::ert_num_threads_gpu()`      | unsigned long  | 1                      |                                                                                                |
| TIMEMORY_ERT_NUM_STREAMS          | `settings::ert_num_streams()`          | unsigned long  | 1                      | Number of (GPU) streams to use for Empirical Roofline Toolkit                                  |
| TIMEMORY_ERT_GRID_SIZE            | `settings::ert_grid_size()`            | unsigned long  | computed               | GPU number of blocks                                                                           |
| TIMEMORY_ERT_BLOCK_SIZE           | `settings::ert_block_size()`           | unsigned long  | 1024                   | GPU number of threads-per-block                                                                |
| TIMEMORY_ERT_ALIGNMENT            | `settings::ert_alignment()`            | unsigned long  | data-type dependent    |                                                                                                |
| TIMEMORY_ERT_MIN_WORKING_SIZE     | `settings::ert_min_working_size()`     | unsigned long  | architecture dependent |                                                                                                |
| TIMEMORY_ERT_MIN_WORKING_SIZE_CPU | `settings::ert_min_working_size_cpu()` | unsigned long  | 64                     |                                                                                                |
| TIMEMORY_ERT_MIN_WORKING_SIZE_GPU | `settings::ert_min_working_size_gpu()` | unsigned long  | 10 MB                  |                                                                                                |
| TIMEMORY_ERT_MAX_DATA_SIZE        | `settings::ert_max_data_size()`        | unsigned long  | architecture dependent |                                                                                                |
| TIMEMORY_ERT_MAX_DATA_SIZE_CPU    | `settings::ert_max_data_size_cpu()`    | unsigned long  | cache-size dependent   |                                                                                                |
| TIMEMORY_ERT_MAX_DATA_SIZE_GPU    | `settings::ert_max_data_size_gpu()`    | unsigned long  | 500 MB                 |                                                                                                |
| TIMEMORY_ERT_SKIP_OPS             | `settings::ert_skip_ops()`             | string         | `""`                   | Skip unrolling FLOPs of these sizes, e.g. (`"4,8"`) in `ops_main<2, 4, 8, 16>(...)`            |
| TIMEMORY_ALLOW_SIGNAL_HANDLER     | `settings::allow_signal_handler()`     | bool           | ON                     |                                                                                                |
| TIMEMORY_ENABLE_SIGNAL_HANDLER    | `settings::enable_signal_handler()`    | bool           | OFF                    |                                                                                                |
| TIMEMORY_ENABLE_ALL_SIGNALS       | `settings::enable_all_signals()`       | bool           | OFF                    |                                                                                                |
| TIMEMORY_DISABLE_ALL_SIGNALS      | `settings::disable_all_signals()`      | bool           | OFF                    |                                                                                                |
| TIMEMORY_NODE_COUNT               | `settings::node_count()`               | int            | 0                      | Explicitly configure the number of nodes                                                       |
| TIMEMORY_DESTRUCTOR_REPORT        | `settings::destructor_report()`        | bool           | OFF                    | `auto_{tuple,list,hybrid}` print at destruction                                                |
| TIMEMORY_PYTHON_EXE               | `settings::python_exe()`               | string         | `"python"`             | Path to python executable when plotting from C++                                               |
| TIMEMORY_UPCXX_INIT               | `settings::upcxx_init()`               | bool           | ON                     | `timemory_init(int*, char***)` initializes UPC++                                               |
| TIMEMORY_UPCXX_FINALIZE           | `settings::upcxx_finalize()`           | bool           | ON                     | `timemory_finalize()` finalizes UPC++                                                          |

> NOTE: To configure timemory to default to `OFF`, define `-DTIMEMORY_DEFAULT_ENABLED=false` during application compilation

## Example

```c++
#include <timemory/timemory.hpp>

using namespace tim::component;

using wall_tuple_t = tim::auto_tuple<wall_clock>;
using auto_tuple_t = tim::auto_tuple<wall_clock, cpu_clock,
                                     cpu_util, peak_rss>;
using comp_tuple_t = typename auto_tuple_t::component_type;

long
fibonacci(long n)
{
    TIMEMORY_BASIC_MARKER(wall_tuple_t, "");
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

int
main(int argc, char** argv)
{
    tim::settings::banner()            = false;
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

    tim::timemory_finalize();
}
```

Compile:

```console
g++ -O3 -pthread -I/usr/local example.cpp -o example
```

Output:

```console
#--------------------- tim::manager initialized [0][0] ---------------------#


fibonacci(10) = 55
>>>  fibonacci(10) :     0.000104 sec wall,     0.000000 sec cpu,    0.0 % cpu_util,        0.000 KB peak_rss [laps: 1]

fibonacci(11) = 89
>>>  fibonacci(11) :     0.000106 sec wall,     0.000000 sec cpu,    0.0 % cpu_util,        0.000 KB peak_rss [laps: 1]

fibonacci(12) = 144
>>>  fibonacci(12) :     0.000159 sec wall,     0.000000 sec cpu,    0.0 % cpu_util,        0.000 KB peak_rss [laps: 1]

>>>  overall timer :     0.000696 sec wall,     0.000000 sec cpu,    0.0 % cpu_util,        0.000 KB peak_rss [laps: 1]

[wall]|0> Outputting 'timemory-example-output/wall.txt'...

>>> overall timer                       :     0.000696 sec wall,   1 laps, depth  0 (exclusive:  47.1%)
>>> |_fibonacci(10)                     :     0.000104 sec wall,   1 laps, depth  1 (exclusive:   8.9%)
>>>   |_fibonacci                       :     0.000095 sec wall,   1 laps, depth  2 (exclusive:   1.1%)
>>>     |_fibonacci                     :     0.000093 sec wall,   2 laps, depth  3 (exclusive:   2.0%)
>>>       |_fibonacci                   :     0.000092 sec wall,   4 laps, depth  4 (exclusive:   4.9%)
>>>         |_fibonacci                 :     0.000087 sec wall,   8 laps, depth  5 (exclusive:  36.9%)
>>>           |_fibonacci               :     0.000055 sec wall,  16 laps, depth  6 (exclusive:  17.9%)
>>>             |_fibonacci             :     0.000045 sec wall,  32 laps, depth  7 (exclusive:  34.9%)
>>>               |_fibonacci           :     0.000029 sec wall,  52 laps, depth  8 (exclusive:  49.9%)
>>>                 |_fibonacci         :     0.000015 sec wall,  44 laps, depth  9 (exclusive:  47.6%)
>>>                   |_fibonacci       :     0.000008 sec wall,  16 laps, depth 10 (exclusive:  98.8%)
>>>                     |_fibonacci     :     0.000000 sec wall,   2 laps, depth 11
>>> |_fibonacci(11)                     :     0.000106 sec wall,   1 laps, depth  1 (exclusive:   4.0%)
>>>   |_fibonacci                       :     0.000101 sec wall,   1 laps, depth  2 (exclusive:   0.7%)
>>>     |_fibonacci                     :     0.000101 sec wall,   2 laps, depth  3 (exclusive:   1.3%)
>>>       |_fibonacci                   :     0.000099 sec wall,   4 laps, depth  4 (exclusive:   2.6%)
>>>         |_fibonacci                 :     0.000097 sec wall,   8 laps, depth  5 (exclusive:   5.0%)
>>>           |_fibonacci               :     0.000092 sec wall,  16 laps, depth  6 (exclusive:  10.3%)
>>>             |_fibonacci             :     0.000083 sec wall,  32 laps, depth  7 (exclusive:  25.5%)
>>>               |_fibonacci           :     0.000062 sec wall,  62 laps, depth  8 (exclusive:  52.8%)
>>>                 |_fibonacci         :     0.000029 sec wall,  84 laps, depth  9 (exclusive:  68.3%)
>>>                   |_fibonacci       :     0.000009 sec wall,  58 laps, depth 10 (exclusive:  77.5%)
>>>                     |_fibonacci     :     0.000002 sec wall,  18 laps, depth 11 (exclusive:  95.6%)
>>>                       |_fibonacci   :     0.000000 sec wall,   2 laps, depth 12
>>> |_fibonacci(12)                     :     0.000159 sec wall,   1 laps, depth  1 (exclusive:   2.4%)
>>>   |_fibonacci                       :     0.000155 sec wall,   1 laps, depth  2 (exclusive:   0.4%)
>>>     |_fibonacci                     :     0.000154 sec wall,   2 laps, depth  3 (exclusive:   0.8%)
>>>       |_fibonacci                   :     0.000153 sec wall,   4 laps, depth  4 (exclusive:   1.6%)
>>>         |_fibonacci                 :     0.000151 sec wall,   8 laps, depth  5 (exclusive:   3.3%)
>>>           |_fibonacci               :     0.000146 sec wall,  16 laps, depth  6 (exclusive:   6.6%)
>>>             |_fibonacci             :     0.000136 sec wall,  32 laps, depth  7 (exclusive:  13.8%)
>>>               |_fibonacci           :     0.000117 sec wall,  64 laps, depth  8 (exclusive:  29.3%)
>>>                 |_fibonacci         :     0.000083 sec wall, 114 laps, depth  9 (exclusive:  49.4%)
>>>                   |_fibonacci       :     0.000042 sec wall, 128 laps, depth 10 (exclusive:  75.5%)
>>>                     |_fibonacci     :     0.000010 sec wall,  74 laps, depth 11 (exclusive:  80.9%)
>>>                       |_fibonacci   :     0.000002 sec wall,  20 laps, depth 12 (exclusive:  95.7%)
>>>                         |_fibonacci :     0.000000 sec wall,   2 laps, depth 13

[cpu]|0> Outputting 'timemory-example-output/cpu.txt'...

>>> overall timer   :     0.000000 sec cpu, 1 laps, depth 0 (exclusive:   0.0%)
>>> |_fibonacci(10) :     0.000000 sec cpu, 1 laps, depth 1
>>> |_fibonacci(11) :     0.000000 sec cpu, 1 laps, depth 1
>>> |_fibonacci(12) :     0.000000 sec cpu, 1 laps, depth 1

[cpu_util]|0> Outputting 'timemory-example-output/cpu_util.txt'...

>>> overall timer   :    0.0 % cpu_util, 1 laps, depth 0 (exclusive:   0.0%)
>>> |_fibonacci(10) :    0.0 % cpu_util, 1 laps, depth 1
>>> |_fibonacci(11) :    0.0 % cpu_util, 1 laps, depth 1
>>> |_fibonacci(12) :    0.0 % cpu_util, 1 laps, depth 1

[peak_rss]|0> Outputting 'timemory-example-output/peak_rss.txt'...

>>> overall timer   :        0.000 KB peak_rss, 1 laps, depth 0 (exclusive:   0.0%)
>>> |_fibonacci(10) :        0.000 KB peak_rss, 1 laps, depth 1
>>> |_fibonacci(11) :        0.000 KB peak_rss, 1 laps, depth 1
>>> |_fibonacci(12) :        0.000 KB peak_rss, 1 laps, depth 1

[metadata::manager::finalize]> Outputting 'timemory-example-output/metadata.json'...


#---------------------- tim::manager destroyed [0][0] ----------------------#
```
