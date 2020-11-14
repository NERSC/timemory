# Components

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2

   components/gotcha
   components/roofline
   components/custom_components
```

This is an overview of the components available in timemory. For detailed info on the
member functions, etc. please refer to the [Doxygen](doxygen).

Components are categorized into some general subsections and then sorted alphabetically.
In general, which member function are present are not that important as long
as you use the variadic component bundlers -- these handle ignoring
trying to call `start()` on a component is the component does not have
a `start()` member function but is bundled alongside other components
which do (that the `start()` was intended for).

## Component Metadata

```eval_rst
.. doxygenstruct:: tim::component::enumerator
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::metadata
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::properties
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::static_properties
   :members:
   :undoc-members:
```

## Timing Components

```eval_rst
.. doxygenstruct:: tim::component::cpu_clock

.. doxygenstruct:: tim::component::cpu_util

.. doxygenstruct:: tim::component::kernel_mode_time

.. doxygenstruct:: tim::component::monotonic_clock

.. doxygenstruct:: tim::component::monotonic_raw_clock

.. doxygenstruct:: tim::component::process_cpu_clock

.. doxygenstruct:: tim::component::process_cpu_util

.. doxygenstruct:: tim::component::system_clock

.. doxygenstruct:: tim::component::thread_cpu_clock

.. doxygenstruct:: tim::component::thread_cpu_util

.. doxygenstruct:: tim::component::user_clock

.. doxygenstruct:: tim::component::user_mode_time

.. doxygenstruct:: tim::component::wall_clock
```

## Resource Usage Components

```eval_rst
.. doxygenstruct:: tim::component::current_peak_rss

.. doxygenstruct:: tim::component::num_io_in

.. doxygenstruct:: tim::component::num_io_out

.. doxygenstruct:: tim::component::num_major_page_faults

.. doxygenstruct:: tim::component::num_minor_page_faults

.. doxygenstruct:: tim::component::page_rss

.. doxygenstruct:: tim::component::peak_rss

.. doxygenstruct:: tim::component::priority_context_switch

.. doxygenstruct:: tim::component::virtual_memory

.. doxygenstruct:: tim::component::voluntary_context_switch
```

## I/O Components

```eval_rst
.. doxygenstruct:: tim::component::read_bytes

.. doxygenstruct:: tim::component::read_char

.. doxygenstruct:: tim::component::written_bytes

.. doxygenstruct:: tim::component::written_char
```

## User Bundle Components

Timemory provides the `user_bundle` component as a generic component bundler
that the user can use to insert components at runtime. This component is
heavily used when mapping timemory to languages other than C++. Timemory
implements many specialization of this template class for various tools.
For example, `user_mpip_bundle` is the bundle used by the MPI wrappers,
`user_profiler_bundle` is used by the Python function profiler,
`user_trace_bundle` is used by the dynamic instrumentation tool `timemory-run` and
the Python line tracing profiler, etc. These specialization are
all individually configurable and it is recommended that applications create
their own specialization specific to their project -- this will ensure that
the desired set of components configured by your application will not be
affected by a third-party library configuring their own set of components.

The general design is that each user-bundle:

- Has their own unique environment variable for exclusive configuration, usually `"TIMEMORY_<LABEL>_COMPONENTS"`, e.g.:
  - `"TIMEMORY_TRACE_COMPONENTS"` for `user_trace_bundle`
  - `"TIMEMORY_MPIP_COMPONENTS"` for `user_mpip_components`
- If the unique environment variable is set, only the components in the variable are used
  - Thus making the bundle uniquely configurable
- If the unique environment variable is *not* set, it searches one or more _backup_ environment variables, the last of which being `"TIMEMORY_GLOBAL_COMPONENTS"`
  - Thus, if no specific environment variables are set, all user bundles collect the components specified in `"TIMEMORY_GLOBAL_COMPONENTS"`
- If the unique environment variable is set to `"none"`, it terminates searching the backup environment variables
  - Thus, `"TIMEMORY_GLOBAL_COMPONENTS"` can be set but the user can suppress a specific bundle from being affected by this configuration
- If the unique environment variable contains `"fallthrough"`, it will continue adding the components specified by the backup environment variables
  - Thus, the components specified in `"TIMEMORY_GLOBAL_COMPONENTS"` and `"TIMEMORY_<LABEL>_COMPONENTS"` will be added

```eval_rst
.. doxygenstruct:: tim::component::user_bundle

.. doxygentypedef:: tim::component::user_global_bundle

.. doxygentypedef:: tim::component::user_kokkosp_bundle

.. doxygentypedef:: tim::component::user_mpip_bundle

.. doxygentypedef:: tim::component::user_ncclp_bundle

.. doxygentypedef:: tim::component::user_ompt_bundle

.. doxygentypedef:: tim::component::user_profiler_bundle

.. doxygentypedef:: tim::component::user_trace_bundle

```

## Third-Party Interface Components

```eval_rst
.. doxygenstruct:: tim::component::allinea_map

.. doxygenstruct:: tim::component::caliper_marker

.. doxygenstruct:: tim::component::caliper_config

.. doxygenstruct:: tim::component::caliper_loop_marker

.. doxygenstruct:: tim::component::craypat_counters

.. doxygenstruct:: tim::component::craypat_flush_buffer

.. doxygenstruct:: tim::component::craypat_heap_stats

.. doxygenstruct:: tim::component::craypat_record

.. doxygenstruct:: tim::component::craypat_region

.. doxygenstruct:: tim::component::gperftools_cpu_profiler

.. doxygenstruct:: tim::component::gperftools_heap_profiler

.. doxygenstruct:: tim::component::likwid_marker

.. doxygenstruct:: tim::component::likwid_nvmarker

.. doxygenstruct:: tim::component::ompt_handle

.. doxygenstruct:: tim::component::tau_marker

.. doxygenstruct:: tim::component::vtune_event

.. doxygenstruct:: tim::component::vtune_frame

.. doxygenstruct:: tim::component::vtune_profiler

```

## Hardware Counter Components

```eval_rst
.. doxygenstruct:: tim::component::papi_tuple

.. doxygenstruct:: tim::component::papi_rate_tuple

.. doxygenstruct:: tim::component::papi_array

.. doxygenstruct:: tim::component::papi_vector

```

## Miscellaneous Components

```eval_rst
.. doxygenstruct:: tim::component::cpu_roofline

.. doxygentypedef:: tim::component::cpu_roofline_dp_flops

.. doxygentypedef:: tim::component::cpu_roofline_flops

.. doxygentypedef:: tim::component::cpu_roofline_sp_flops

```

## GPU Components

```eval_rst
.. doxygenstruct:: tim::component::cuda_event

.. doxygenstruct:: tim::component::cupti_activity

.. doxygenstruct:: tim::component::cupti_counters

.. doxygenstruct:: tim::component::cupti_profiler

.. doxygenstruct:: tim::component::gpu_roofline

.. doxygentypedef:: tim::component::gpu_roofline_dp_flops

.. doxygentypedef:: tim::component::gpu_roofline_flops

.. doxygentypedef:: tim::component::gpu_roofline_hp_flops

.. doxygentypedef:: tim::component::gpu_roofline_sp_flops

.. doxygenstruct:: tim::component::nvtx_marker
   :members:
```

## Data Tracking Components

```eval_rst
.. doxygenstruct:: tim::component::data_tracker
   :members:
.. doxygentypedef:: tim::component::data_tracker_integer

.. doxygentypedef:: tim::component::data_tracker_unsigned

.. doxygentypedef:: tim::component::data_tracker_floating
```

## Function Wrapping Components

```eval_rst
.. doxygenstruct:: tim::component::gotcha
   :members:
.. doxygenstruct:: tim::component::malloc_gotcha
   :members:
```

## Base Components

```eval_rst
.. doxygenstruct:: tim::component::base
   :members:
   :undoc-members:
.. doxygenstruct:: tim::component::empty_base
   :members:
   :undoc-members:
```
