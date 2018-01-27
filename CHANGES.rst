TiMemory
========

Release: TiMemory 1.1b0
~~~~~~~~~~~~~~~~~~~~~~~

- Author: Jonathan R. Madsen
- Date:   Fri Jan 26 17:24:42 2018 -0800
    
    - Updated documentation for TiMemory 1.1b0
    - added rss_usage decorator
    - made a base class for the decorators
    - update the setup.py to 1.1b0
    - +=, -=, +, -, current, and peak methods to RSS in Python
    - updated timemory_test.py
    - restructured submodules: originally all submodules were under util, now only the decorators live there
    - new submodules are: options, mpi_support, plotting, util, and signals
    - timemory.options: I/O options, formerly timemory.util.options
    - timemory.plotting: plotting utilities, formerly timemory.util.plot
    - timemory.util: decorators, formerly all-encompassing submodule
    - timemory.signals: signal enumeration, new submodule
    - timemory.mpi_support: report MPI information, new submodule
    - added new RSS capability (+=, -= usage)
    - added Python RSS interface
    - added signals interface
