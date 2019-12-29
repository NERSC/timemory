# About

Timemory is a modular API for performance measurements and analysis with a very lightweight overhead.
If timemory does not support a particular measurement type or analysis method, user applications
can easily create their own component that accomplishes the desired task.

Timemory is implemented as a generic C++11 template library but supports implementation
in C, C++, CUDA, and Python codes.
The design goal of timemory is to create an easy-to-use framework for generating
performance measurements and analysis methods which are extremely flexible
with respect to both how the data is stored/accumulated and which methods the measurement
or analysis supports. In order to keep the overhead as low as reasonable achievable,
a significant amount of logic is evaluated at compile-time. As a result, applications
which directly utilize the C++ template interface tend to see increases in compilation
time, binary size (especially when debug info is included), and compiler memory usage.
If this aspect of timemory impedes productivity, the best course of action is to
utilize the library interface.

## Credits

Timemory is actively developed by NERSC at Lawrence Berkeley National Laboratory

| Name               |                                        Affiliation                                        |                    GitHub                     |
| ------------------ | :---------------------------------------------------------------------------------------: | :-------------------------------------------: |
| Jonathan R. Madsen | [NERSC](https://www.nersc.gov/about/nersc-staff/application-performance/jonathan-madsen/) |    [jrmadsen](https://github.com/jrmadsen)    |
| Yunsong Wang       |       [NERSC](https://www.nersc.gov/about/nersc-staff/nesap-postdocs/yunsong-wang/)       | [PointKernel](https://github.com/PointKernel) |

## Contributions

Timemory encourages contributions via GitHub pull-requests.
