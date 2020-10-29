# Memory Management

## Manager Class

The `tim::manager` class is a thread-local singleton which handles the memory management for each thread.
It is stored as a `std::shared_ptr` which automatically deletes itself when the thread exits.
When the thread-local singleton for each component storage class is created, it increments the reference
count for this class to ensure that it exists while any storage classes are allocated. When the
storage singletons are created, it registers functors with the manager for destorying itself.

```eval_rst
.. doxygenclass:: tim::manager
    :members:
    :undoc-members:
```

## Storage Class

The `tim::storage` class is a thread-local singleton which handles the call-graph and persistent
data accumulation for each component. It is stored as a `std::unique_ptr` which automatically deletes
itself when the thread exits. On the non-primary thread, destruction of the singleton merges it's
call-graph data into the storage singleton on the primary thread. Initialization and finalization
of the storage class is the ONLY time that thread synchronization and inter-process communication
occurs. This characteristic enables timemory storage to arbitrarily scale to any number of threads and/or
processes without performance degradation. If you want to information of the state of the call-graph,
the `tim::storage<T>` is the structure to do so, e.g. the current size of the call-graph, a serialization
of the current process- and thread-specific, etc. Invoking the `get()` member function will return
the data for the current thread on worker threads and invoking the `get()` member function on the primary
thread will return the data for _all_ the threads. Invoking `mpi_get()` will aggregate the results
across all MPI processes, `upc_get()` will aggregate the results across all the UPC++ results, and
`dmp_get()` (dmp == distributed memory parallelism) will aggregate all the results across MPI and UPC++
processes.

```eval_rst
.. doxygenclass:: tim::base::storage
    :members:
    :undoc-members:
.. doxygenclass:: tim::storage
    :members:
    :outline:
.. doxygenclass:: tim::impl::storage
    :members:
    :outline:
```
