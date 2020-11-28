# C++ Type-Traits

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

## Component Implementation

```eval_rst
.. doxygenstruct:: tim::trait::is_available
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::python_args
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::runtime_enabled
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::api_components
   :members:
   :undoc-members:
```

## Type Identification

```eval_rst
```

## Base Class Modifications

```eval_rst
.. doxygenstruct:: tim::trait::base_has_accum
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::base_has_last
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::dynamic_base
   :members:
   :undoc-members:
```

## Data Sharing

```eval_rst
.. doxygenstruct:: tim::trait::cache
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::derivation_types
   :members:
   :undoc-members:
```

## Data Collection

```eval_rst
.. doxygenstruct:: tim::trait::sampler
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::file_sampler
   :members:
   :undoc-members:
```

## Feature Support

```eval_rst
.. doxygenstruct:: tim::trait::supports_custom_record
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::supports_flamegraph
   :members:
   :undoc-members:
```

## Archive Serialization

```eval_rst
.. doxygenstruct:: tim::trait::api_input_archive
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::api_output_archive
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::input_archive
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::output_archive
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::pretty_archive
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::requires_json
   :members:
   :undoc-members:
```

## Units and Formatting

```eval_rst
.. doxygenstruct:: tim::trait::is_memory_category
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::is_timing_category
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::uses_memory_units
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::uses_timing_units
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::uses_percent_units
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::units
   :members:
   :undoc-members:
```

## Output Reporting

```eval_rst
.. doxygenstruct:: tim::trait::report
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::report_count
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::report_depth
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::report_metric_name
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::report_units
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::report_sum
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::report_mean
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::report_statistics
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::report_self
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::custom_label_printing
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::custom_serialization
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::custom_unit_printing
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::echo_enabled
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::iterable_measurement
   :members:
   :undoc-members:
```

## Statistics

```eval_rst
.. doxygenstruct:: tim::trait::statistics
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::record_statistics
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::permissive_statistics
   :members:
   :undoc-members:
```

## Storage

```eval_rst
.. doxygenstruct:: tim::trait::uses_storage
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::uses_value_storage
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::tree_storage
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::flat_storage
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::timeline_storage
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::thread_scope_only
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::data
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::secondary_data
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::collects_data
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::generates_output
   :members:
   :undoc-members:
```

## Deprecated

These type-traits are either:

- Removed from the source code entirely
- Automatically detected
- Migrated to concepts

```eval_rst
.. doxygenstruct:: tim::trait::is_component
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::is_gotcha
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::is_user_bundle
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::record_max
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::array_serialization
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::requires_prefix
   :members:
   :undoc-members:
.. doxygenstruct:: tim::trait::supports_args
   :members:
   :undoc-members:
```
