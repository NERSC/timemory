# About

Timemory is very _lightweight_, _cross-language_ timing, resource usage, and hardware counter utility
for reporting timing, resource usage, and hardware counters for the CPU and GPU.

Timemory is implemented as a generic C++11 template library but supports implementation in C, C++, CUDA, and Python codes.
The design goal of timemory is to enable "always-on" performance analysis that can be standard part of the source code
with a negligible amount of overhead.

Timemory is not intended to replace profiling tools such as Intel's VTune, GProf, etc. -- instead,
it complements them by enabling one to verify timing and memory usage without the overhead of the profiler.

## Credits

Timemory is actively maintained by NERSC at Lawrence Berkeley National Laboratory

| Name               |                                        Affiliation                                        |                    GitHub                     |
| ------------------ | :---------------------------------------------------------------------------------------: | :-------------------------------------------: |
| Jonathan R. Madsen | [NERSC](https://www.nersc.gov/about/nersc-staff/application-performance/jonathan-madsen/) |    [jrmadsen](https://github.com/jrmadsen)    |
| Yunsong Wang       |       [NERSC](https://www.nersc.gov/about/nersc-staff/nesap-postdocs/yunsong-wang/)       | [PointKernel](https://github.com/PointKernel) |

## Contributions

Timemory encourages contributions via GitHub pull-requests.
