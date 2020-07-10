# Python Bindings

The timemory python interface is generated via the PyBind11 library. The combination of these two template-based
libraries provides a feature-rich interface which combines the flexibility of python with the performance of C++.

## Desciption

The python interface provides several pre-configured scoped components in bundles and profiler sub-packages which can be used as decorators or context-managers. The timemory settings be controlled either directly via the `settings` sub-package or by using environment variables. The `component` subpackage contains the individual components that can be used for custom instrumentation. Further, the `hardware_counters` sub-package provides API interface to accessing hardware counters (requires PAPI and/or CUDA support). The `mpi` subpackage provides bindings to timemory's MPI support. The generic data `plotting` (such as plotting instrumentation graphs) and `roofline` plotting are available in subsequent sub-packages.

## Contents

```bash
$ python -c "import timemory; help(timemory)"

PACKAGE CONTENTS
    api (package)
    bundle (package)
    common
    component (package)
    ert (package)
    hardware_counters (package)
    libpytimemory
    mpi (package)
    mpi_support (package)
    options
    plotting (package)
    profiler (package)
    region (package)
    roofline (package)
    settings (package)
    signals
    trace (package)
    test (package)
    units
    util (package)

CLASSES
    pybind11_builtins.pybind11_object(builtins.object)
        timemory.libpytimemory.auto_timer
        timemory.libpytimemory.component_bundle
        timemory.libpytimemory.manager
        timemory.libpytimemory.rss_usage
        timemory.libpytimemory.timer
```

## Usage

```python
import timemory
# timemory components, decorators, bundles, roofline
# instrumented or profiled user code
timemory.finalize()
```

## Examples

The following code snippets demonstrate a few key features and components packed by the timemory python package.

### Settings

```python
import timemory

# set verbose output to 1
timemory.settings.verbose = 1
# disable timemory debug prints
timemory.settings.debug = False
# set output data format output to json
timemory.settings.json_output = True
# disable mpi_thread mode
timemory.settings.mpi_thread  = False
# enable timemory dart output
timemory.settings.dart_output = True
timemory.settings.dart_count = 1
# disable timemory banner
timemory.settings.banner = False
```

### Environment Variables

```python
import os

# enable timemory flat profile mode
os.environ["TIMEMORY_FLAT_PROFILE"] = "ON"
# enable timemory timeline profile mode
os.environ["TIMEMORY_TIMELINE_PROFILE"] = "ON"
# parse the environment variable updates within timemory
timemory.settings.parse()
```

### Decorators

```python
import timemory
import timemory.util.marker as marker, auto_timer

@marker(["trip_count", "peak_rss"])
def fibonacci_instrumented(n):
    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)

@auto_timer()
def fibonacci_timed(n):
    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)
```

### Context Managers

```python
import timemory
from timemory.profiler import profile

def main():
    with profile(["wall_clock", "peak_rss"], flat=False, timeline=False):
        ans = fibonacci(n=3)
```

### Library Bindings

This is a sample of the python bindings to the timemory library interface available to C, C++, and Fortran.
Using these libraries calls in python code which calls C/C++/Fortran code also instrumented with
timemory will enable the same set of performance analysis components to span all the languages.

```python
import timemory

# initialize
timemory.trace.init("wall_clock", False, "tracing")

# insert a trace
timemory.trace.push("consume_time")
ans = work_milliseconds(1000)
timemory.trace.pop("consume_time")

# insert another trace point
timemory.trace.push("sleeping")
ans = sleep_milliseconds(1000)
timemory.trace.pop("sleeping")

# insert a region
timemory.region.push("work_region")
for i in range(10):
    ans = work_milliseconds(1000)
timemory.region.pop("work_region")

# finalize
timemory.trace.finalize()
```

### Individual Components

Sample usage:

```python
import time
import timemory
from  timemory.component import WallClock

# instantiate wall clock component
wc = WallClock("wall")
# start the clock
wc.start()
# sleep
time.sleep(2)
# stop the clock
wc.stop()
# get data
result = wc.get()
#finalize timemory
timemory.finalize()
```

Sample help page:

```python
>>> help(timemory.component)
NAME
    timemory.libpytimemory.component - Stand-alone classes for the components.
    Unless push() and pop() are called on these objects, they will not store any data in the
    timemory call-graph (if applicable)

CLASSES
    pybind11_builtins.pybind11_object(builtins.object)
        AllineaMap
        Caliper
        CaliperConfig
        CaliperLoopMarker
        CpuClock
        CpuRooflineDpFlops
        CpuRooflineFlops
        CpuRooflineSpFlops
        CpuUtil
        CraypatCounters
        CraypatFlushBuffer
        CraypatHeapStats
        CraypatRecord
        CraypatRegion
        CudaEvent
        CudaProfiler
        CuptiActivity
        CuptiCounters
        CurrentPeakRss
        GperftoolsCpuProfiler
        GperftoolsHeapProfiler
        GpuRooflineDpFlops
        GpuRooflineFlops
        GpuRooflineHpFlops
        GpuRooflineSpFlops
        KernelModeTime
        LikwidMarker
        LikwidNvmarker
        MallocGotcha
        MonotonicClock
        MonotonicRawClock
        NumIoIn
        NumIoOut
        NumMajorPageFaults
        NumMinorPageFaults
        NvtxMarker
        OmptHandle
        PageRss
        PapiArray
        PapiVector
        PeakRss
        PriorityContextSwitch
        ProcessCpuClock
        ProcessCpuUtil
        ReadBytes
        SysClock
        TauMarker
        ThreadCpuClock
        ThreadCpuUtil
        TripCount
        UserClock
        UserGlobalBundle
        UserListBundle
        UserModeTime
        UserMpipBundle
        UserOmptBundle
        UserTupleBundle
        VirtualMemory
        VoluntaryContextSwitch
        VtuneEvent
        VtuneFrame
        VtuneProfiler
        WallClock
        WrittenBytes
        id

    class AllineaMap(pybind11_builtins.pybind11_object)
     |  not available
     |

    # ... etc ...

    class WrittenBytes(pybind11_builtins.pybind11_object)
     |  Physical I/O writes
     |
     |  Method resolution order:
     |      WrittenBytes
     |      pybind11_builtins.pybind11_object
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __add__(...)
     |      __add__(self: timemory.libpytimemory.component.WrittenBytes, arg0: timemory.libpytimemory.component.WrittenBytes) -> timemory.libpytimemory.component.WrittenBytes
     |
     |  __iadd__(...)
     |      __iadd__(self: timemory.libpytimemory.component.WrittenBytes, arg0: timemory.libpytimemory.component.WrittenBytes) -> timemory.libpytimemory.component.WrittenBytes
     |
     |  __init__(...)
     |      __init__(*args, **kwargs)
     |      Overloaded function.
     |
     |      1. __init__(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Creates component
     |
     |      2. __init__(self: timemory.libpytimemory.component.WrittenBytes, arg0: str) -> None
     |
     |      Creates component with a label
     |
     |  __isub__(...)
     |      __isub__(self: timemory.libpytimemory.component.WrittenBytes, arg0: timemory.libpytimemory.component.WrittenBytes) -> timemory.libpytimemory.component.WrittenBytes
     |
     |      Subtract rhs from lhs
     |
     |  __repr__(...)
     |      __repr__(self: timemory.libpytimemory.component.WrittenBytes) -> str
     |
     |      String representation
     |
     |  __sub__(...)
     |      __sub__(self: timemory.libpytimemory.component.WrittenBytes, arg0: timemory.libpytimemory.component.WrittenBytes) -> timemory.libpytimemory.component.WrittenBytes
     |
     |  get(...)
     |      get(self: timemory.libpytimemory.component.WrittenBytes) -> List[float[2]]
     |
     |      Get the current value
     |
     |  hash(...)
     |      hash(self: timemory.libpytimemory.component.WrittenBytes) -> int
     |
     |      Get the current hash
     |
     |  key(...)
     |      key(self: timemory.libpytimemory.component.WrittenBytes) -> str
     |
     |      Get the identifier
     |
     |  laps(...)
     |      laps(self: timemory.libpytimemory.component.WrittenBytes) -> int
     |
     |      Get the number of laps
     |
     |  mark_begin(...)
     |      mark_begin(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Mark an begin point
     |
     |  mark_end(...)
     |      mark_end(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Mark an end point
     |
     |  measure(...)
     |      measure(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Take a measurement
     |
     |  pop(...)
     |      pop(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Pop off the call-graph
     |
     |  push(...)
     |      push(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Push into the call-graph
     |
     |  rekey(...)
     |      rekey(self: timemory.libpytimemory.component.WrittenBytes, arg0: str) -> None
     |
     |      Change the identifier
     |
     |  reset(...)
     |      reset(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Reset the values
     |
     |  start(...)
     |      start(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Start measurement
     |
     |  stop(...)
     |      stop(self: timemory.libpytimemory.component.WrittenBytes) -> None
     |
     |      Stop measurement
     |
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |
     |  configure(...) from builtins.PyCapsule
     |      configure(*args, **kwargs) -> None
     |
     |      Configure the tool
     |
     |  description(...) from builtins.PyCapsule
     |      description() -> str
     |
     |      Get the description for the type
     |
     |  label(...) from builtins.PyCapsule
     |      label() -> str
     |
     |      Get the label for the type
     |
     |  record(...) from builtins.PyCapsule
     |      record() -> List[int[2]]
     |
     |      Get the record of a measurement
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  available
     |
     |  has_value
     |
     |  ----------------------------------------------------------------------
     |  Static methods inherited from pybind11_builtins.pybind11_object:
     |
     |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
     |      Create and return a new object.  See help(type) for accurate signature.

    class id(pybind11_builtins.pybind11_object)
     |  Component enumerations for timemory module
     |
     |  Members:
     |
     |    allinea_map : Controls the AllineaMAP sampler
     |
     |    caliper : Generic forwarding of markers to Caliper instrumentation
     |
     |    caliper_config : Caliper configuration manager
     |
     |    caliper_loop_marker : Variant of caliper_marker with support for loop marking
     |
     |    cpu_clock : Total CPU time spent in both user- and kernel-mode
     |
     |    cpu_roofline_dp_flops : Model used to provide performance relative to the peak possible performance  on a CPU architecture.
     |
     |    cpu_roofline_flops : Model used to provide performance relative to the peak possible performance on a CPU architecture.
     |
     |    cpu_roofline_sp_flops : Model used to provide performance relative to the peak  possible performance on a CPU architecture.
     |
     |    cpu_util : Percentage of CPU-clock time divided by wall-clock time
     |
     |    craypat_counters : Names and value of any counter events that have been set to count on the hardware category
     |
     |    craypat_flush_buffer : Writes all the recorded contents in the data buffer. Returns the number of bytes flushed
     |
     |    craypat_heap_stats : Undocumented by 'pat_api.h'
     |
     |    craypat_record : Toggles CrayPAT recording on calling thread
     |
     |    craypat_region : Adds region labels to CrayPAT output
     |
     |    cuda_event : Records the time interval between two points in a CUDA stream. Less accurate than 'cupti_activity' for kernel timing
     |
     |    cuda_profiler : Control switch for a CUDA profiler running on the application
     |
     |    cupti_activity : Wall-clock execution timing for the CUDA API
     |
     |    cupti_counters : Hardware counters for the CUDA API
     |
     |    current_peak_rss : Absolute value of high-water mark of memory allocation in RAM
     |
     |    gperftools_cpu_profiler : Control switch for gperftools CPU profiler
     |
     |    gperftools_heap_profiler : Control switch for the gperftools heap profiler
     |
     |    gpu_roofline_dp_flops : Model used to provide performance relative to the peak possible performance on a GPU architecture.
     |
     |    gpu_roofline_flops : Model used to provide performance relative to the peak possible performance on a GPU architecture.
     |
     |    gpu_roofline_hp_flops : GPU Roofline tim::cuda::half2 Counters
     |
     |    gpu_roofline_sp_flops : Model used to provide performance relative to the peak  possible performance on a GPU architecture.
     |
     |    kernel_mode_time : CPU time spent executing in kernel mode (via rusage)
     |
     |    likwid_marker : LIKWID perfmon (CPU) marker forwarding
     |
     |    likwid_nvmarker : LIKWID nvmon (GPU) marker forwarding
     |
     |    malloc_gotcha : GOTCHA wrapper for memory allocation functions
     |
     |    monotonic_clock : Wall-clock timer which will continue to increment even while the system is asleep
     |
     |    monotonic_raw_clock : Wall-clock timer unaffected by frequency or time adjustments in system time-of-day clock
     |
     |    num_io_in : Number of times the filesystem had to perform input
     |
     |    num_io_out : Number of times the filesystem had to perform output
     |
     |    num_major_page_faults : Number of page faults serviced that required I/O activity
     |
     |    num_minor_page_faults : Number of page faults serviced without any I/O activity via 'reclaiming' a page frame from the list of pages awaiting reallocation
     |
     |    nvtx_marker : Generates high-level region markers for CUDA profilers
     |
     |    ompt_handle : Control switch for enabling/disabling OpenMP tools defined by the tim::api::native_tag tag
     |
     |    page_rss : Amount of memory allocated in pages of memory. Unlike peak_rss, value will fluctuate as memory is freed/allocated
     |
     |    papi_array : Fixed-size array of PAPI HW counters
     |
     |    papi_vector : Dynamically allocated array of PAPI HW counters
     |
     |    peak_rss : Measures changes in the high-water mark for the amount of memory allocated in RAM. May fluctuate if swap is enabled
     |
     |    priority_context_switch : Number of context switch due to higher priority process becoming runnable or because the current process exceeded its time slice)
     |
     |    process_cpu_clock : CPU-clock timer for the calling process (all threads)
     |
     |    process_cpu_util : Percentage of CPU-clock time divided by wall-clock time for calling process (all threads)
     |
     |    read_bytes : Physical I/O reads
     |
     |    sys_clock : CPU time spent in kernel-mode
     |
     |    tau_marker : Forwards markers to TAU instrumentation (via Tau_start and Tau_stop)
     |
     |    thread_cpu_clock : CPU-clock timer for the calling thread
     |
     |    thread_cpu_util : Percentage of CPU-clock time divided by wall-clock time for calling thread
     |
     |    trip_count : Counts number of invocations
     |
     |    user_clock : CPU time spent in user-mode
     |
     |    user_global_bundle : Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion
     |
     |    user_list_bundle : Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion
     |
     |    user_mode_time : CPU time spent executing in user mode (via rusage)
     |
     |    user_mpip_bundle : Generic bundle of components designed for runtime configuration by a user  via environment variables and/or direct insertion
     |
     |    user_ompt_bundle : Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion
     |
     |    user_tuple_bundle : Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion
     |
     |    virtual_memory : Records the change in virtual memory
     |
     |    voluntary_context_switch : Number of context switches due to a process voluntarily giving up the processor before its time slice was completed
     |
     |    vtune_event : Creates events for Intel profiler running on the application
     |
     |    vtune_frame : Creates frames for Intel profiler running on the application
     |
     |    vtune_profiler : Control switch for Intel profiler running on the application
     |
     |    wall_clock : Real-clock timer (i.e. wall-clock timer)
     |
     |    written_bytes : Physical I/O writes
     |
     |  Method resolution order:
     |      id
     |      pybind11_builtins.pybind11_object
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __and__ = (...)
     |      (self: object, arg0: object) -> object
     |
     |  __eq__ = (...)
     |      (self: object, arg0: object) -> bool
     |
     |  __ge__ = (...)
     |      (self: object, arg0: object) -> bool
     |
     |  __getstate__ = (...)
     |      (self: object) -> int_
     |
     |  __gt__ = (...)
     |      (self: object, arg0: object) -> bool
     |
     |  __hash__ = (...)
     |      (self: object) -> int_
     |
     |  __init__(...)
     |      __init__(self: timemory.libpytimemory.component.id, arg0: int) -> None
     |
     |  __int__(...)
     |      __int__(self: timemory.libpytimemory.component.id) -> int
     |
     |  __invert__ = (...)
     |      (self: object) -> object
     |
     |  __le__ = (...)
     |      (self: object, arg0: object) -> bool
     |
     |  __lt__ = (...)
     |      (self: object, arg0: object) -> bool
     |
     |  __ne__ = (...)
     |      (self: object, arg0: object) -> bool
     |
     |  __or__ = (...)
     |      (self: object, arg0: object) -> object
     |
     |  __rand__ = (...)
     |      (self: object, arg0: object) -> object
     |
     |  __repr__ = (...)
     |      (self: handle) -> str
     |
     |  __ror__ = (...)
     |      (self: object, arg0: object) -> object
     |
     |  __rxor__ = (...)
     |      (self: object, arg0: object) -> object
     |
     |  __setstate__ = (...)
     |      (self: timemory.libpytimemory.component.id, arg0: int) -> None
     |
     |  __xor__ = (...)
     |      (self: object, arg0: object) -> object
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __members__
     |
     |  name
     |      (self: handle) -> str
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  allinea_map = id.allinea_map
     |
     |  caliper = id.caliper
     |
     |  caliper_config = id.caliper_config
     |
     |  caliper_loop_marker = id.caliper_loop_marker
     |
     |  cpu_clock = id.cpu_clock
     |
     |  cpu_roofline_dp_flops = id.cpu_roofline_dp_flops
     |
     |  cpu_roofline_flops = id.cpu_roofline_flops
     |
     |  cpu_roofline_sp_flops = id.cpu_roofline_sp_flops
     |
     |  cpu_util = id.cpu_util
     |
     |  craypat_counters = id.craypat_counters
     |
     |  craypat_flush_buffer = id.craypat_flush_buffer
     |
     |  craypat_heap_stats = id.craypat_heap_stats
     |
     |  craypat_record = id.craypat_record
     |
     |  craypat_region = id.craypat_region
     |
     |  cuda_event = id.cuda_event
     |
     |  cuda_profiler = id.cuda_profiler
     |
     |  cupti_activity = id.cupti_activity
     |
     |  cupti_counters = id.cupti_counters
     |
     |  current_peak_rss = id.current_peak_rss
     |
     |  gperftools_cpu_profiler = id.gperftools_cpu_profiler
     |
     |  gperftools_heap_profiler = id.gperftools_heap_profiler
     |
     |  gpu_roofline_dp_flops = id.gpu_roofline_dp_flops
     |
     |  gpu_roofline_flops = id.gpu_roofline_flops
     |
     |  gpu_roofline_hp_flops = id.gpu_roofline_hp_flops
     |
     |  gpu_roofline_sp_flops = id.gpu_roofline_sp_flops
     |
     |  kernel_mode_time = id.kernel_mode_time
     |
     |  likwid_marker = id.likwid_marker
     |
     |  likwid_nvmarker = id.likwid_nvmarker
     |
     |  malloc_gotcha = id.malloc_gotcha
     |
     |  monotonic_clock = id.monotonic_clock
     |
     |  monotonic_raw_clock = id.monotonic_raw_clock
     |
     |  num_io_in = id.num_io_in
     |
     |  num_io_out = id.num_io_out
     |
     |  num_major_page_faults = id.num_major_page_faults
     |
     |  num_minor_page_faults = id.num_minor_page_faults
     |
     |  nvtx_marker = id.nvtx_marker
     |
     |  ompt_handle = id.ompt_handle
     |
     |  page_rss = id.page_rss
     |
     |  papi_array = id.papi_array
     |
     |  papi_vector = id.papi_vector
     |
     |  peak_rss = id.peak_rss
     |
     |  priority_context_switch = id.priority_context_switch
     |
     |  process_cpu_clock = id.process_cpu_clock
     |
     |  process_cpu_util = id.process_cpu_util
     |
     |  read_bytes = id.read_bytes
     |
     |  sys_clock = id.sys_clock
     |
     |  tau_marker = id.tau_marker
     |
     |  thread_cpu_clock = id.thread_cpu_clock
     |
     |  thread_cpu_util = id.thread_cpu_util
     |
     |  trip_count = id.trip_count
     |
     |  user_clock = id.user_clock
     |
     |  user_global_bundle = id.user_global_bundle
     |
     |  user_list_bundle = id.user_list_bundle
     |
     |  user_mode_time = id.user_mode_time
     |
     |  user_mpip_bundle = id.user_mpip_bundle
     |
     |  user_ompt_bundle = id.user_ompt_bundle
     |
     |  user_tuple_bundle = id.user_tuple_bundle
     |
     |  virtual_memory = id.virtual_memory
     |
     |  voluntary_context_switch = id.voluntary_context_switch
     |
     |  vtune_event = id.vtune_event
     |
     |  vtune_frame = id.vtune_frame
     |
     |  vtune_profiler = id.vtune_profiler
     |
     |  wall_clock = id.wall_clock
     |
     |  written_bytes = id.written_bytes
     |
     |  ----------------------------------------------------------------------
     |  Static methods inherited from pybind11_builtins.pybind11_object:
     |
     |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
     |      Create and return a new object.  See help(type) for accurate signature.

DATA
    allinea_map = id.allinea_map
    caliper = id.caliper
    caliper_config = id.caliper_config
    caliper_loop_marker = id.caliper_loop_marker
    cpu_clock = id.cpu_clock
    cpu_roofline_dp_flops = id.cpu_roofline_dp_flops
    cpu_roofline_flops = id.cpu_roofline_flops
    cpu_roofline_sp_flops = id.cpu_roofline_sp_flops
    cpu_util = id.cpu_util
    craypat_counters = id.craypat_counters
    craypat_flush_buffer = id.craypat_flush_buffer
    craypat_heap_stats = id.craypat_heap_stats
    craypat_record = id.craypat_record
    craypat_region = id.craypat_region
    cuda_event = id.cuda_event
    cuda_profiler = id.cuda_profiler
    cupti_activity = id.cupti_activity
    cupti_counters = id.cupti_counters
    current_peak_rss = id.current_peak_rss
    gperftools_cpu_profiler = id.gperftools_cpu_profiler
    gperftools_heap_profiler = id.gperftools_heap_profiler
    gpu_roofline_dp_flops = id.gpu_roofline_dp_flops
    gpu_roofline_flops = id.gpu_roofline_flops
    gpu_roofline_hp_flops = id.gpu_roofline_hp_flops
    gpu_roofline_sp_flops = id.gpu_roofline_sp_flops
    kernel_mode_time = id.kernel_mode_time
    likwid_marker = id.likwid_marker
    likwid_nvmarker = id.likwid_nvmarker
    malloc_gotcha = id.malloc_gotcha
    monotonic_clock = id.monotonic_clock
    monotonic_raw_clock = id.monotonic_raw_clock
    num_io_in = id.num_io_in
    num_io_out = id.num_io_out
    num_major_page_faults = id.num_major_page_faults
    num_minor_page_faults = id.num_minor_page_faults
    nvtx_marker = id.nvtx_marker
    ompt_handle = id.ompt_handle
    page_rss = id.page_rss
    papi_array = id.papi_array
    papi_vector = id.papi_vector
    peak_rss = id.peak_rss
    priority_context_switch = id.priority_context_switch
    process_cpu_clock = id.process_cpu_clock
    process_cpu_util = id.process_cpu_util
    read_bytes = id.read_bytes
    sys_clock = id.sys_clock
    tau_marker = id.tau_marker
    thread_cpu_clock = id.thread_cpu_clock
    thread_cpu_util = id.thread_cpu_util
    trip_count = id.trip_count
    user_clock = id.user_clock
    user_global_bundle = id.user_global_bundle
    user_list_bundle = id.user_list_bundle
    user_mode_time = id.user_mode_time
    user_mpip_bundle = id.user_mpip_bundle
    user_ompt_bundle = id.user_ompt_bundle
    user_tuple_bundle = id.user_tuple_bundle
    virtual_memory = id.virtual_memory
    voluntary_context_switch = id.voluntary_context_switch
    vtune_event = id.vtune_event
    vtune_frame = id.vtune_frame
    vtune_profiler = id.vtune_profiler
    wall_clock = id.wall_clock
    written_bytes = id.written_bytes

FILE
    (built-in)
```
