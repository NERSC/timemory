# C++ Operations

> Namespace: `tim::operation`

These structs are invoked by the variadic template bundlers when performing
operations on components. The constructor of the class is invoked with a
reference to the component and any additional arguments either provided by
the bundler or passed in by the user.

```cpp
component_tuple<Foo, Bar> handle("example");

handle.start();
for(int i = 0; i < N; ++i)
{
    handle.store(i);
    // ...
}
handle.stop();
```

will roughly translate to:

```cpp
component_tuple<Foo, Bar> handle("example");

Foo& f = std::get<0>(handle.m_data);
Bar& b = std::get<1>(handle.m_data);

// constructor
auto ... = operation::set_prefix<Foo>(f, "example");
auto ... = operation::set_prefix<Bar>(b, "example");

// handle.start()
auto ... = operation::start<Foo>(f);
auto ... = operation::start<Bar>(b);

for(int i = 0; i < N; ++i)
{
    // handle.store(i)
    auto ... = operation::store<Foo>(f, i);
    auto ... = operation::store<Bar>(b, i);
    // ...
}
// handle.stop()
auto ... = operation::stop<Foo>(f);
auto ... = operation::stop<Bar>(b);
```

and, after the optimization by the compiler, those temporary objects
will be eliminated and only the contents of happen inside the constructor (if any)
will remain. This can be verified by executing the `nm` utility and searching for
any symbols in the binary with the name `operation`:

```console
nm --demangle <EXE or LIB> | grep operation
```

```eval_rst
.. doxygenstruct:: tim::operation::init
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::init_storage
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::fini_storage
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::construct
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::set_prefix
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::set_scope
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::push_node
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::pop_node
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::record
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::reset
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::measure
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::sample
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::compose
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::is_running
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::set_started
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::start
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::priority_start
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::standard_start
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::delayed_start
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::set_stopped
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::stop
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::priority_stop
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::standard_stop
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::delayed_stop
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::mark
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::mark_begin
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::mark_end
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::store
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::audit
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::plus
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::minus
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::multiply
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::divide
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::get
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::get_data
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::get_labeled_data
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::base_printer
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::print
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::print_header
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::print_statistics
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::print_storage
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::add_secondary
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::add_statistics
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::serialization
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::echo_measurement
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::copy
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::assemble
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::derive
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::cache
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::fini
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::finalize::get
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::finalize::mpi_get
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::finalize::upc_get
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::finalize::dmp_get
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::finalize::print
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::finalize::merge
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::finalize::flamegraph
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::generic_deleter
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::generic_counter
    :members:
    :undoc-members:
.. doxygenstruct:: tim::operation::generic_operator
    :members:
    :undoc-members:
```
