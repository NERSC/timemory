# Getting Started

## Dynamic Instrumentation

Dynamic instrumentation (instrumentation without any modifications to the source code) is only
available on Linux and requires the Dyninst package. It is recommended to install Dyninst
via Spack. For further information, please see documentation on [timemory-run](../tools/timemory-run.md).

## Manual Instrumentation

### Python

Timemory provides an extensive suite of Python utilities. Users are encouraged to
use the built-in `help(...)` manual pages from the Python interpreter for the
most extensive details. The `timemory.util` submodule provides decorators and
context managers to generic bundles of components. The `timemory.profiler`
submodule provides an implementation which instruments every Python interpreter
call in the scope of the profiling instance. In general, components can be
specified through lists/tuples of strings (use `timemory-avail -s` to see the string IDs of
the components) or the `timemory.component` enumeration values. Timemory
also provides stand-alone Python classes for each component in `timemory.components`
(note the `"s"` at the end). The stand-alone Python classes behave slightly
differently in that they do not implicitly interact with the persistent timemory
storage classes which track call-stack hierarchy and therefore require a
an invocation of `push()` before `start()` is invoked and an invocation of
`pop()` after `stop()` is invoked in order to show up correctly in the call-stack
tracing. In the absence of a `push()` and `pop()` operation, these classes
map to the underlying invocation of the tool with no overhead.

## Environment Controls

The vast majority of the environment variables can be viewed using the `timemory-avail` executable with the `-S/--settings` option.
Additionally, the `<OUTPUT_PATH>/metadata.json` file will record all the environment variables during the simulation. In particular,
some dynamically generated environment variables for components and variadic bundlers appear in the `metadata.json` file
and not in the `timemory-avail` output.
