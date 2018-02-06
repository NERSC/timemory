TiMemory
========

Release: TiMemory 1.1.8
~~~~~~~~~~~~~~~~~~~~~~~

Author: Jonathan R. Madsen
Date:   Tue Feb 6 05:12:56 2018 -0800

    - Large restructuring to fix submodule nesting issue
      
        - Python > 3.1 now allows: "from timemory.util import rss_usage"  
        - requires importlib.util
        - not available in older versions
          
    - Better C++ auto_timer tagging and second option
      
         - TIMEMORY_AUTO_TIMER (func@'file':line)
         - TIMEMORY_AUTO_TIMER_SIMPLE (func)
         - TIMEMORY_AUTO_TIMER_SIMPLE was the old TIMEMORY_AUTO_TIMER
           
    - Squashed bugs + I/O and test improvements
    - Excluded non-displayed timers (i.e. falling below minimum) from setting the output width
    - Improved MPI detection          
    - Included tests in installation --> timemory.tests.run(pattern="")
    - timemory.plotting routines have improved handling of bar graphs to help eliminate hidden graphs in the overlay

Release: TiMemory 1.1.7
~~~~~~~~~~~~~~~~~~~~~~~

- Author: Jonathan R. Madsen
- Date:   Wed Jan 31 14:28:19 2018 -0800

    - I/O fix for RSS to report negative values (i.e. deallocation)

Release: TiMemory 1.1.5
~~~~~~~~~~~~~~~~~~~~~~~

- Author: Jonathan R. Madsen
- Date:   Mon Jan 29 18:46:09 2018 -0800

    - Backported CMake to support older version of CMake (previous min: 3.1.3, new min: 2.8.12)
  
Release: TiMemory 1.1.3
~~~~~~~~~~~~~~~~~~~~~~~

- Author: Jonathan R. Madsen
- Date:   Mon Jan 29 18:46:09 2018 -0800

    - added `timemory.set_exit_action(...)` capability for defining a function to handle the exit of the application due to a signal being raised (e.g. SIGHUP, SIGINT, SIGABRT)

Release: TiMemory 1.1.2
~~~~~~~~~~~~~~~~~~~~~~~

- Author: Jonathan R. Madsen
- Date:   Mon Jan 29 16:20:06 2018 -0800

    - removed Python 'cmake' requirement in 'setup.py

Release: TiMemory 1.1.1
~~~~~~~~~~~~~~~~~~~~~~~

- Author: Jonathan R. Madsen
- Date:   Mon Jan 29 15:00:12 2018 -0800

    - Added 'report_at_exit' parameter to auto_timer decorator
    - Added added_args flag for auto_timer decorator
    - Fixed I/O output bug
    - Added setup.cfg
    - Fixed auto_timer decorator issue with self.key, self.is_class, and self.add_args


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
