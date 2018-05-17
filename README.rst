TiMemory
========

C / C++ / Python Timing + Memory Utilities including auto-timers and temporary memory calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|Build Status| |Build status|

`TiMemory on GitHub (Source
code) <https://github.com/jrmadsen/TiMemory>`__

`TiMemory General Documentation (GitHub
Pages) <https://jrmadsen.github.io/TiMemory>`__

`TiMemory Source Code Documentation
(Doxygen) <https://jrmadsen.github.io/TiMemory/doxy/index.html>`__

`TiMemory Testing Dashboard
(CDash) <http://jonathan-madsen.info/cdash/public/index.php?project=TiMemory>`__

`TiMemory Release
Notes <https://jrmadsen.github.io/TiMemory/ReleaseNotes.html>`__

TiMemory's design is aimed at routine ("everyday") timing and memory analysis that can be standard part of the source code.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TiMemory is a very *lightweight*, *cross-language* timing and memory
utility. It support implementation in C, C++, and Python and is easily
imported into CMake projects.

TiMemory is Lightweight and Fast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analysis on a fibonacci calculation determined that each TiMemory
"auto-timer" adds an average overhead of 9 microseconds (``0.000009 s``)
without memory measurements and 16 microseconds (``0.000016 s``) with
memory measurements. This performance is specific to the machine and the
overhead for a particular machine can be calculated by running the
``test_cxx_overhead`` example.

Since TiMemory only records information of the functions explicitly
specified, you can safely assume that unless TiMemory is inserted into a
function called ``> 100,000`` times, it won't be adding more than a
second of runtime to the function. Therefore, there is a simple rule of
thumb: don't insert a TiMemory auto-timer into very simple functions
that get called very frequently.

TiMemory is not intended to replace profiling tools such as Intel's
VTune, GProf, etc. -- instead, it complements them by enabling one to
verify timing and memory usage without the overhead of the profiler.

TiMemory is Cross-Language: C, C++, and Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is very common for Python projects to implement expensive routines in
C or C++. Implementing a TiMemory auto-timer in any combination of these
languages will produce one combined report for all the languages
(provided each language links to the same library). However, this is a
feature of TiMemory. TiMemory can be used in standalone C, C++, or
Python projects.

TiMemory is thread-safe with minimal locking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All TiMemory auto-timers reference a timer storaged in a thread-local
singleton "manager" class. It is never recommended to directly create a
"manager" instance. Instead, call the static function
``tim::manager::instance()`` or if the master thread instance is
desired: ``tim::manager::master_instance()``. It is generally safe to
delete the master instance at the end of the application. The master
instance is a raw pointer in C++. The only locking that occurs within
TiMemory is on the destruction of a non-master-thread instance of the
manager -- which is automatically done via the use of a shared\_ptr in
C++ when the thread exits.

TiMemory supports MPI
~~~~~~~~~~~~~~~~~~~~~

If a project uses MPI, TiMemory will combined the reports from all the
MPI ranks when a report is requested.

TiMemory has built-in timing and memory plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The results from TiMemory can be serialized to JSON and the JSON output
can be used to produce timing and memory performance plots via the
standalone ``timemory-plotter`` or ``timemory.plotting`` Python module

TiMemory reports temporary memory usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory reports permit determination of temporary memory usage until the
"high-water mark" of memory allocation is reached

TiMemory can be used from the command-line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UNIX systems provide ``timem`` executable that works like ``time``. On
all systems, ``pytimem`` is provided.

::

  $ timem bash -c "for i in {1..5}; do sleep 1; done"

  > [bash] total execution time  : 5.052 wall, 0.010 user + 0.010 system = 0.020 cpu (  0.4%) [sec], 2.2 peak rss [MB]

  $ pytimem bash -c "for i in {1..5}; do sleep 1; done"

  > [bash] total execution time  : 5.043 wall, 0.010 user + 0.020 system = 0.030 cpu (  0.6%) [sec], 2.1 peak rss [MB]

``pytimem`` has the benefit of being able to define a
``timemory_json_handler`` module in the CWD that handles the
serialization data (such as submitting the data to a server). Here is an
example ``timemory_json_handler.py``:

.. code:: python

  #!/usr/bin/env python

  def receive(args, json_obj):
      print('\n{} received json object for "{}"\n'.format(__file__, args))

      print('json dictionary\n{}'.format(json_obj))

TiMemory has environment controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  TIMEMORY\_VERBOSE
-  TIMEMORY\_DISABLE\_TIMER\_MEMORY
-  TIMEMORY\_NUM\_THREADS\_ENV
-  TIMEMORY\_NUM\_THREADS
-  TIMEMORY\_ENABLE
-  TIMEMORY\_TIMING\_FORMAT
-  TIMEMORY\_TIMING\_PRECISION
-  TIMEMORY\_TIMING\_WIDTH
-  TIMEMORY\_TIMING\_UNITS
-  TIMEMORY\_TIMING\_SCIENTIFIC
-  TIMEMORY\_MEMORY\_FORMAT
-  TIMEMORY\_MEMORY\_PRECISION
-  TIMEMORY\_MEMORY\_WIDTH
-  TIMEMORY\_MEMORY\_UNITS
-  TIMEMORY\_MEMORY\_SCIENTIFIC
-  TIMEMORY\_TIMING\_MEMORY\_FORMAT
-  TIMEMORY\_TIMING\_MEMORY\_PRECISION
-  TIMEMORY\_TIMING\_MEMORY\_WIDTH
-  TIMEMORY\_TIMING\_MEMORY\_UNITS
-  TIMEMORY\_TIMING\_MEMORY\_SCIENTIFIC

.. |Build Status| image:: https://travis-ci.org/jrmadsen/TiMemory.svg?branch=master
   :target: https://travis-ci.org/jrmadsen/TiMemory
.. |Build status| image:: https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1?svg=true
   :target: https://ci.appveyor.com/project/jrmadsen/timemory
