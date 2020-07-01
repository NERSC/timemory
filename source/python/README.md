# timemory python bindings

PYBIND11 based bindings for the timemory Python package. 

## Desciption

The timemory python package provides several pre-configured scoped components in bundles and profiler sub-pacakages that can be used as decorators or context-managers. The timemory settings be controlled either directly via the `settings` sub-package or by using environment variables. The `component` subpackage contains the individual components that can be used for custom instrumentation. Further, the `hardware_counters` sub-package provides API interface to accessing hardware counters (requires PAPI and CUDA support). The `mpi` subpackage provides bindings to timemory's MPI support. The generic data `plotting` (such as plotting instrumentation graphs) and `roofline` plotting are available in subsequent sub-packages. The C++/Python bindings have been implemented using PYBIND11.

## Contents

```bash
$ python -c "import timemory; help(timemory)"

PACKAGE CONTENTS
    api (package)
    bundle (package)
    common
    component (package)
    ert (package)
    gperftools (package)
    hardware_counters (package)
    libpytimemory
    mpi (package)
    mpi_support (package)
    options
    plotting (package)
    profiler (package)
    roofline (package)
    settings (package)
    signals
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
import json
import os
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

### Individual Components
```python
import time
import timemory
from  timemory.component import WallClock

# instantiate wall clock component
wc = WallClock("wall")
#start the clock
wc.start()
#sleep
time.sleep(2)
# stop the clock
wc.stop()
# get data
result = wc.get()
#finalize timemory
timemory.finalize()
```

### Tracing

```python
import timemory

# initialize
timemory.timemory_trace_init("wall_clock", False, "timemory_tracing")

# insert a trace
timemory.timemory_push_trace("consume_time")
ans = work_milliseconds(1000)
timemory.timemory_pop_trace("consume_time")

# insert another trace point
timemory.timemory_push_trace("sleeping")
ans = sleep_milliseconds(1000)
timemory.timemory_pop_trace("sleeping")

# insert a region
timemory.timemory_push_region("work_region")
for i in range(10):
    ans = work_milliseconds(1000)
timemory.timemory_pop_region("work_region")

# finalize
timemory.timemory_trace_finalize()
```