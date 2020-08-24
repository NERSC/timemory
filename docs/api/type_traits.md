# Type-Traits

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2
```

Type-traits are used to specialize behavior at compile-time. In general, timemory
tries to avoid specialization in the core library (when possible) so that users are
not restricted for specializing downstream. Please note, specializations may be ignored
or cause compilation errors if extern templates are used. Ignoring specializations commonly happen in
output routines like `tim::operation::print<T>` where the value of the specialization
is used at runtime and the body of the output routine is not actually instantiated in the
user code.

```eval_rst
.. doxygenstruct:: tim::trait::api_components
.. doxygenstruct:: tim::trait::api_input_archive
.. doxygenstruct:: tim::trait::api_output_archive
.. doxygenstruct:: tim::trait::array_serialization
.. doxygenstruct:: tim::trait::base_has_accum
.. doxygenstruct:: tim::trait::base_has_last
.. doxygenstruct:: tim::trait::cache
.. doxygenstruct:: tim::trait::collects_data
.. doxygenstruct:: tim::trait::custom_label_printing
.. doxygenstruct:: tim::trait::custom_laps_printing
.. doxygenstruct:: tim::trait::custom_serialization
.. doxygenstruct:: tim::trait::custom_unit_printing
.. doxygenstruct:: tim::trait::data
.. doxygenstruct:: tim::trait::derivation_types
.. doxygenstruct:: tim::trait::echo_enabled
.. doxygenstruct:: tim::trait::file_sampler
.. doxygenstruct:: tim::trait::flat_storage
.. doxygenstruct:: tim::trait::generates_output
.. doxygenstruct:: tim::trait::implements_storage
.. doxygenstruct:: tim::trait::input_archive
.. doxygenstruct:: tim::trait::is_available
.. doxygenstruct:: tim::trait::is_component
.. doxygenstruct:: tim::trait::is_gotcha
.. doxygenstruct:: tim::trait::is_memory_category
.. doxygenstruct:: tim::trait::is_timing_category
.. doxygenstruct:: tim::trait::is_user_bundle
.. doxygenstruct:: tim::trait::iterable_measurement
.. doxygenstruct:: tim::trait::output_archive
.. doxygenstruct:: tim::trait::permissive_statistics
.. doxygenstruct:: tim::trait::pretty_json
.. doxygenstruct:: tim::trait::python_args
.. doxygenstruct:: tim::trait::record_statistics
.. doxygenstruct:: tim::trait::report_mean
.. doxygenstruct:: tim::trait::report_metric_name
.. doxygenstruct:: tim::trait::report_self
.. doxygenstruct:: tim::trait::report_statistics
.. doxygenstruct:: tim::trait::report_sum
.. doxygenstruct:: tim::trait::report_units
.. doxygenstruct:: tim::trait::report_values
.. doxygenstruct:: tim::trait::requires_json
.. doxygenstruct:: tim::trait::requires_prefix
.. doxygenstruct:: tim::trait::runtime_enabled
.. doxygenstruct:: tim::trait::sampler
.. doxygenstruct:: tim::trait::secondary_data
.. doxygenstruct:: tim::trait::statistics
.. doxygenstruct:: tim::trait::supports_args
.. doxygenstruct:: tim::trait::supports_custom_record
.. doxygenstruct:: tim::trait::supports_flamegraph
.. doxygenstruct:: tim::trait::thread_scope_only
.. doxygenstruct:: tim::trait::units
.. doxygenstruct:: tim::trait::uses_memory_units
.. doxygenstruct:: tim::trait::uses_percent_units
.. doxygenstruct:: tim::trait::uses_timing_units
```
