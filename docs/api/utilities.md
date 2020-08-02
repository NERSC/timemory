# Utilities

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2
```

## Quirks

Quirks are used to slightly tweak the default behavior of component bundlers.
Quirks can be included as template parameters and the property of the quirk will
be applied to all instances of that component bundle,
e.g. `component_tuple<wall_clock, quirk::flat_scope>` will cause all instances
of that bundler to propagate the flat-storage specification to the wall-clock
component regardless of the global setting. Quirks can also be applied
per-component bundle instance within a `tim::quirk::config<...>` specification, e.g.
an `auto_list` traditionally invokes `start()` on a `component_list` in the constructor
and thus, any attempts to activate the components in the list after construction are
ignored. Thus, if a special initialization case is desired for a particular instance
of `auto_list`, the `quirk::explicit_start` can be added to suppress this behavior so
that initialization can be performed before manually invoking `start()`.

```cpp
namespace quirk = tim::quirk;

using foo_t   = tim::auto_list<wall_clock, cpu_clock>;
using quirk_t = quirk::config<quirk::explicit_start>;

void Func1(bool condition)
{
    auto f = foo_t("Func1", quirk_t{});
    if(condition)
    {
        f.initialize<wall_clock>();
        f.disable<cpu_clock>();
        f.start();
    }
    // ...
}

void Func2()
{
    auto f = foo_t("Func2");
    // ...
}
```

```eval_rst
.. doxygenstruct:: tim::quirk::config
.. doxygenstruct:: tim::quirk::auto_start
.. doxygenstruct:: tim::quirk::auto_stop
.. doxygenstruct:: tim::quirk::explicit_start
.. doxygenstruct:: tim::quirk::explicit_stop
.. doxygenstruct:: tim::quirk::exit_report
.. doxygenstruct:: tim::quirk::no_init
.. doxygenstruct:: tim::quirk::no_store
.. doxygenstruct:: tim::quirk::tree_scope
.. doxygenstruct:: tim::quirk::flat_scope
.. doxygenstruct:: tim::quirk::timeline_scope
```
