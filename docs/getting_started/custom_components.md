# Custom Components

Timemory supports user-specifed components. A custom component must inherit from the static polymorphic base class:
`tim::component::base`. This class provides the integration into the API and requires a minimum of two types:

1. component type (i.e. itself) [`required`]
2. the data type being stored [`required`]
3. A variadic list of policy classes [`optional`]

```cpp
// generic static polymorphic base class
template <typename _Tp, typename value_type = int64_t, typename... _Policies>
struct base;

// create cpu_clock component and use default data type (int64_t)
struct cpu_clock
: public base<cpu_clock>
{
    // ...
};

// create cpu_util component and specify data type as pair of 64-bit integers
struct cpu_util
: public base<cpu_util, std::pair<int64_t, int64_t>>
{
    // ...
};

// create papi_tuple component and specify data type as an array of long long types
// with the thread_init and thread_finalize policies
template <int... EventTypes>
struct papi_tuple
: public base<
              papi_tuple<EventTypes...>,                    // self type
              std::array<long long, sizeof...(EventTypes)>, // data type
              policy::thread_init, policy::thread_finalize  // policies
             >
{
    // ...
};
```

## Type Traits

Type traits are provided to customize how operations on the data type are handled. For example, adding two `real_clock`
components together is a simple `a += b` operation but adding two `peak_rss` components is better defined as a
max operation: `a = max(a, b)` since `+=` a "peak" memory value could return a result larger than the total amount of
memory on the system. Additionally, certain components are not available on certain systems, e.g. components that
provide data via the POSIX rusage struct on Windows.

> Namespace: `tim::trait`

| Type Trait                     | Description                                                                            | Default Setting   |
| ------------------------------ | -------------------------------------------------------------------------------------- | ----------------- |
| **`is_available`**             | Specify the availablity of the component's implementation                              | `std::true_type`  |
| **`record_max`**               | Specify that arithmetic operations should use comparisions                             | `std::false_type` |
| **`array_serialization`**      | Specify the component stores data as an array                                          | `std::false_type` |
| **`external_output_handling`** | Specify the component handles it's own output (if any)                                 | `std::false_type` |
| **`requires_prefix`**          | Specify the component requires a `prefix` member variable to be set before `start()`   | `std::false_type` |
| **`custom_label_printing`**    | Specify the component provides its own labeling for output (typically multiple labels) | `std::false_type` |
| **`custom_unit_printing`**     | Specify the component provides its own units for output (typically multiple units)     | `std::false_type` |
| **`custom_laps_printing`**     | Specify the component provides its own "lap" printing for output                       | `std::false_type` |
| **`start_priority`**           | Specify the component should be started before non-prioritized components              | `std::false_type` |
| **`stop_priority`**            | Specify the component should be stopped before non-prioritized components              | `std::false_type` |
| **`is_timing_category`**       | Designates the width and precision should apply environment-specified timing settings  | `std::false_type` |
| **`is_memory_category`**       | Designates the width and precision should apply environment-specified memory settings  | `std::false_type` |
| **`uses_timing_units`**        | Designates the width and precision should apply environment-specified timing units     | `std::false_type` |
| **`uses_memory_units`**        | Designates the width and precision should apply environment-specified memory units     | `std::false_type` |
| **`requires_json`**            | Specify the component should always output the JSON file format                        | `std::false_type` |
| **`supports_args`**            | Specifies whether a components `mark_begin` and `mark_end` support a set of arguments  | `std::false_type` |
| **`supports_custom_record`**   | Specifies a type supports changing the record() static function per-instance           | `std::false_type` |
| **`iterable_measurement`**     | Specifies that `get()` member function returns an iterable type (e.g. vector)          | `std::false_type` |

> `tim::trait::array_serialization` trait causes invocation of `_array` variants of `label`, `descript`, `display_unit`, and `unit` function calls,
> e.g. `label_array()`, and implies `trait::iterable_measurement`

### Type Traits Example

```cpp
template <typename _Tp> struct is_available : std::true_type { };

// on Windows, stack_rss is not available because stack_rss requires POSIX rusage struct
#if defined(_WIN32) || defined(_WIN64)
template <> struct is_available<component::stack_rss> : std::false_type { };
#endif
```

### The `is_available` Trait

`tim::trait::is_available` is a very important type-trait. Marking a component as not available results in this type
being filtered out of the variadic wrapper classes `auto_tuple`, `auto_list`, `component_tuple`, and `component_list`.
This means that with the above example, on Windows, the `example_real_stack_type` type below will implement the
same behavor as the `example_real_type` type:

```cpp
using example_real_type       = tim::auto_tuple< real_clock >;
using example_real_stack_type = tim::auto_tuple< real_clock, stack_rss >;
// On Windows, above type will filter out stack_rss at compile time
```

In addition to a performance benefit (no operations on a component that does not provide anything) and a much
cleaner code base (significant reduction of `#ifdef` blocks), this can be used to enable "add-on" features for
performance analysis that won't be included in production code, e.g. including `cpu_roofline_dp_flops` in a specification
but not enabling the PAPI backend in a production code.

## Policies

Policy classes are provided to enable static functionality that may be critical to the functionality of a
component but are not required by most components. Policy classes are declared in the base class specification
and their inclusion requires the definition of an associated static member function.

> Namespace: `tim::policy`

| Policy                             | Associated Static Member Function                     | Invocation Context                             |
| ---------------------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| **`tim::policy::global_init`**     | `void invoke_global_init()`                           | Initial creation of component within a process |
| **`tim::policy::global_finalize`** | `void invoke_global_finalize()`                       | Termination of process (application cleanup)   |
| **`tim::policy::thread_init`**     | `void invoke_thread_init()`                           | Initial creation of component within a thread  |
| **`tim::policy::thread_finalize`** | `void invoke_thread_finalize()`                       | Termination of thread (thread cleanup)         |
| **`tim::policy::serialization`**   | `template <typename Archive> void invoke_serialize()` | Serialization to JSON                          |

In general, the `global_init` policy is used to define an operation that occur prior to recording any component of this
type, e.g. parse environment of any relevant settings; the `global_finalize` policy is used to define an operation that
occurs after all the recording is completed for a process, e.g. calculate the peak values for a roofline; the
`thread_init` policy is used to define an operation that occurs prior to recording any component of this type within that
thread of execution, e.g. start PAPI hardware counters for thread; the `thread_finalize` policy is used to define
an operation that occurs after all the recording is completed within that thread of execution, e.g. stop PAPI hardware
counters for thread; the `serialization` policy is used add additional information to the JSON serialization, e.g.
adding the peak data to a roofline component.

## Custom Component Example

| Required              | Type                      | Description                                                             |
| --------------------- | ------------------------- | ----------------------------------------------------------------------- |
| `value_type`          | Alias/typedef             | Data type used for storage                                              |
| `base_type`           | Alias/typedef             | Base type specification                                                 |
| `precision`           | Constant integer          | Default output precision                                                |
| `width`               | Constant integer          | Default output width                                                    |
| `format_flags`        | `std::ios_base::fmtflags` | Bitset of formatting flags                                              |
| `unit()`              | numerical value           | Units of data type                                                      |
| `label()`             | string                    | Short-hand description without spaces                                   |
| `descript()`          | string                    | Full description of component                                           |
| `display_unit()`      | string                    | String representation of `unit()`                                       |
| `record()`            | static function           | How to measure/record data type of component (must return `value_type`) |
| `get() const`         | const member function     | Returns the current measurement                                         |
| `get_display() const` | const member function     | Returns the current measurement in format desired for reporting         |
| `start()`             | member function           | Defines operation when component recording is started                   |
| `stop()`              | member function           | Defines operation when component recording is stopped                   |

```cpp
struct real_clock : public base<real_clock, int64_t>
{
    // [required] alias to the type of data being recorded is stored
    using value_type = int64_t;

    // [required] alias to base type that implements a lot of the functionality
    using base_type  = base<real_clock, value_type>;

    // this is specific to this particular component
    using ratio_t    = std::nano;

    // [required] these are formatting guidelines
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    // [required] these handle units conversions and descriptions
    static int64_t     unit() { return units::sec; }
    static std::string label() { return "real"; }
    static std::string descript() { return "wall time"; }
    static std::string display_unit() { return "sec"; }

    // [required] this defines how to record a value for the component
    static value_type  record()
    {
        return tim::get_clock_real_now<int64_t, ratio_t>();
    }

    // [required] this defines how to get the relevant data from the component and can
    // return any type
    float get() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }

    // [required] this defines how the value is represented in '<<' and can return any type
    float compute_display() const
    {
        return this->get();
    }

    // [required] this defines what happens when the components is started
    void start()
    {
        base_type::set_started();
        value = record();
    }

    // [required] this defines what happens when the components is stopped
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        base_type::set_stopped();
    }
};
```
