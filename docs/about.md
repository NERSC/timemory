# About

Timemory is a modular API for performance measurements and analysis with a very lightweight overhead.
If timemory does not support a particular measurement type or analysis method, user applications
can easily create their own component that accomplishes the desired task.

Timemory is implemented as a generic C++14 template library but supports implementation
in C, C++, Fortran, CUDA, and Python codes.
The design goal of timemory is to create an easy-to-use framework for generating
performance measurements and analysis methods which are extremely flexible
with respect to both how the data is stored/accumulated and which methods the measurement
or analysis supports.

For a more extensive introduction and more information about the long-term goals, visit the
[README.md](https://github.com/NERSC/timemory/blob/develop/README.md) on GitHub.

## Credits

Timemory is actively developed by NERSC at Lawrence Berkeley National Laboratory

| Name               |                                        Affiliation                                        |                    GitHub                     |
| ------------------ | :---------------------------------------------------------------------------------------: | :-------------------------------------------: |
| Jonathan R. Madsen | [NERSC](https://www.nersc.gov/about/nersc-staff/application-performance/jonathan-madsen/) |    [jrmadsen](https://github.com/jrmadsen)    |
| Yunsong Wang       |       [NERSC](https://www.nersc.gov/about/nersc-staff/nesap-postdocs/yunsong-wang/)       | [PointKernel](https://github.com/PointKernel) |
| Muhammad Haseeb    |                   [NERSC](https://sites.google.com/a/fiu.edu/mhaseeb/)                    |  [mhaseeb123](https://github.com/mhaseeb123)  |

## Contributions

Timemory encourages contributions via GitHub pull-requests.
For more information about contributing new components, please read the
[CONTRIBUTING](https://github.com/NERSC/timemory/blob/develop/CONTRIBUTING.md)
document on GitHub.
