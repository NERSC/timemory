//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file timemory/mpl/type_traits.hpp
 * \headerfile timemory/mpl/type_traits.hpp "timemory/mpl/type_traits.hpp"
 * These are the definitions of type-traits used by timemory and should be defined
 * separately from the class so that they can be queried without including the
 * definition of the component
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/tpls/cereal/archives.hpp"
//
#include <type_traits>

//======================================================================================//
//
//                                 Type Traits
//
//======================================================================================//

namespace tim
{
template <typename...>
class component_tuple;

template <typename T>
struct statistics;

class manager;

namespace trait
{
//--------------------------------------------------------------------------------------//
//
template <typename TraitT>
inline std::string
as_string()
{
    constexpr bool _val = TraitT::value;
    return (_val) ? "true" : "false";
}

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::base_has_accum
/// \brief trait that signifies that a component has an accumulation value
///
template <typename T>
struct base_has_accum : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::base_has_last
/// \brief trait that signifies that a component has an "last" value which may be
/// different than the "value" value
///
template <typename T>
struct base_has_last : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_available
/// \brief trait that signifies that an implementation (e.g. PAPI) is available
///
template <typename T>
struct is_available : TIMEMORY_DEFAULT_AVAILABLE
{};

template <typename T>
struct is_available<T*> : is_available<std::remove_pointer_t<T>>
{};

template <typename T>
using is_available_t = typename is_available<T>::type;

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::data
/// \brief trait to specify the value type of a component before the definition of
/// the component
///
template <typename T>
struct data
{
    using type       = T;
    using value_type = type_list<>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::component_apis
/// \brief trait to specify the APIs that the component is logically a part of
///
template <typename T>
struct component_apis
{
    using type = type_list<>;
};

template <typename T>
using component_apis_t = typename component_apis<T>::type;

//--------------------------------------------------------------------------------------//

template <>
struct runtime_enabled<void>
{
    // GET specialization if component is available
    static bool get() { return get_runtime_value(); }

    // SET specialization if component is available
    static void set(bool val) { get_runtime_value() = val; }

private:
    static bool& get_runtime_value()
    {
        static bool _instance = TIMEMORY_DEFAULT_ENABLED;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::runtime_enabled
/// \brief trait that signifies that an implementation is enabled at runtime. The
/// value returned from get() is for the specific setting for the type, the
/// global settings (type: void) and the specific settings for it's APIs
///
template <typename T>
struct runtime_enabled
{
    // type-list of APIs that are runtime configurable
    using api_type_list =
        get_true_types_t<concepts::is_runtime_configurable, component_apis_t<T>>;

    // GET specialization if component is available
    template <typename U = T>
    TIMEMORY_HOT static inline enable_if_t<is_available<U>::value, bool> get()
    {
        return (runtime_enabled<void>::get() && get_runtime_value() &&
                apply<runtime_enabled>{}(api_type_list{}));
    }

    // SET specialization if component is available
    template <typename U = T>
    TIMEMORY_HOT static inline enable_if_t<is_available<U>::value, void> set(bool val)
    {
        get_runtime_value() = val;
    }

    // GET specialization if component is NOT available
    template <typename U = T>
    static inline enable_if_t<!is_available<U>::value, bool> get()
    {
        return false;
    }

    // SET specialization if component is NOT available
    template <typename U = T>
    static inline enable_if_t<!is_available<U>::value, void> set(bool)
    {}

private:
    TIMEMORY_HOT static bool& get_runtime_value()
    {
        static bool _instance = is_available<T>::value;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::record_max
/// \brief trait that signifies that updating w.r.t. another instance should
/// be a max of the two instances
/// \deprecated This is no longer used. Overload the operators for +=, -=, etc. to obtain
/// previous functionality.
//
template <typename T>
struct record_max : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::array_serialization
/// \brief trait that signifies that data is an array type
/// \deprecated { This trait is no longer used as array types are determined by other
/// means }
///
template <typename T>
struct array_serialization : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::requires_prefix
/// \brief trait that signifies that a component requires the prefix to be set right after
/// construction. Types with this trait must contain a member string variable named
/// prefix
/// \deprecated { This trait is no longer used as this property is determined by other
/// means }
///
template <typename T>
struct requires_prefix : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::custom_label_printing
/// \brief trait that signifies that a component will handle printing the label(s)
///
template <typename T>
struct custom_label_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::custom_unit_printing
/// \brief trait that signifies that a component will handle printing the units(s)
///
template <typename T>
struct custom_unit_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::custom_laps_printing
/// \brief trait that signifies that a component will handle printing the laps(s)
///
template <typename T>
struct custom_laps_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::start_priority
/// \brief trait that designates whether there is a priority when starting the type w.r.t.
/// other types.
///
template <typename T>
struct start_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::stop_priority
/// \brief trait that designates whether there is a priority when stopping the type w.r.t.
/// other types.
///
template <typename T>
struct stop_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_timing_category
/// \brief trait that designates the width and precision should follow formatting settings
/// related to timing measurements
///
template <typename T>
struct is_timing_category : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_memory_category
/// \brief trait that designates the width and precision should follow formatting settings
/// related to memory measurements
///
template <typename T>
struct is_memory_category : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_timing_units
/// \brief trait that designates the units should follow unit settings related to timing
/// measurements
///
template <typename T>
struct uses_timing_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_memory_units
/// \brief trait that designates the units should follow unit settings related to memory
/// measurements
///
template <typename T>
struct uses_memory_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::uses_percent_units
/// \brief trait that designates the units are a percentage
///
template <typename T>
struct uses_percent_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::requires_json
/// \brief trait that designates a type should always print a JSON output
///
template <typename T>
struct requires_json : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_component
/// \brief trait that designates the type is a timemory component
///
template <typename T>
struct is_component : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::api_components
/// \brief trait that designates components in an API (tim::api)
///
template <typename T, typename Tag>
struct api_components
{
    using type = type_list<>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_gotcha
/// \brief trait that designates the type is a gotcha
/// \deprecated{ This is being migrated to a concept }
///
template <typename T>
struct is_gotcha : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::is_user_bundle
/// \brief trait that designates the type is a user-bundle
/// \deprecated{ This is being migrated to a concept }
///
template <typename T>
struct is_user_bundle : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::component_value_type
/// \brief trait that can be used to override the evaluation of the \ref
/// tim::trait::collects_data trait. It checks to see if \ref tim::trait::data is
/// specialized and, if not, evaluates to \code{.cpp} typename T::value_type \endcode
///
template <typename T>
struct component_value_type<T, true>
{
    using type = T;
    static constexpr bool decl_value_v =
        !(std::is_same<type_list<>, typename data<T>::value_type>::value);
    using value_type = std::conditional_t<(decl_value_v), typename data<T>::value_type,
                                          typename T::value_type>;
};

template <typename T>
struct component_value_type<T, false>
{
    using type       = T;
    using value_type = void;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::collects_data
/// \brief trait that specifies or determines if a component collects any data. Default
/// behavior is to check if the component is available and extract the type and value
/// fields from \ref tim::trait::component_value_type
///
template <typename T>
struct collects_data
{
    using type = T;
    using value_type =
        typename component_value_type<T, is_available<T>::value>::value_type;
    static constexpr bool value =
        (!std::is_same<value_type, void>::value &&
         !std::is_same<value_type, type_list<>>::value && is_available<T>::value);
    static_assert(std::is_void<value_type>::value != value,
                  "Error value_type is void and value is true");
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::supports_args
/// \brief trait that designates the type supports calling a function with a certain
/// set of argument types (passed via a tuple).
/// \deprecated This is legacy code and support for calling a function with given
/// arguments is automatically determined.
///
template <typename T, typename Tuple>
struct supports_args : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::supports_custom_record
/// \brief trait that designates the type supports changing the record() static function
/// per-instance
///
template <typename T>
struct supports_custom_record : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::iterable_measurement
/// \brief trait that signifies that get() returns an iterable type
///
template <typename T>
struct iterable_measurement : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::secondary_data
/// \brief trait that signifies that secondary data resembling the original data
/// exists but should be another node entry in the graph. These types
/// must provide a get_secondary() member function and that member function
/// must return a pair-wise iterable container, e.g. std::map, of types:
///     - std::string
///     - value_type
///
template <typename T>
struct secondary_data : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::thread_scope_only
/// \brief trait that signifies the component only has relevant values if it is not
/// collapsed into the master thread
///
template <typename T>
struct thread_scope_only : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::custom_serialization
/// \brief trait that signifies the component will be providing it's own split load(...)
/// and store(...) for serialization so do not provide one in the base class
///
template <typename T>
struct custom_serialization : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::record_statistics
/// \brief trait that signifies the component will calculate min/max/stddev
///
template <typename T>
struct record_statistics : default_record_statistics_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::statistics
/// \brief trait that specifies the data type of the statistics
///
template <typename T>
struct statistics
{
    using type = std::tuple<>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::permissive_statistics
/// \brief trait that will suppress compilation error in
/// \code{.cpp} operation::add_statistics<Component> \endcode if the data type passed does
/// not match \code{.cpp} statistics<Component>::type \endcode
///
template <typename T>
struct permissive_statistics : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::sampler
/// \brief trait that signifies the component supports sampling
///
template <typename T>
struct sampler : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::file_sampler
/// \brief trait that signifies the component samples a measurement from a file
///
template <typename T>
struct file_sampler : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::units
/// \brief trait the designates the units
///
template <typename T>
struct units
{
    using type         = int64_t;
    using display_type = std::string;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::echo_enabled
/// \brief trait that configures echo_measurement usage
///
template <typename T>
struct echo_enabled : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::pretty_json
/// \brief trait that configures whether JSON output uses pretty print. If set to
/// false_type then the JSON will be compact
///
template <typename T>
struct pretty_json : std::false_type
{};

/// \struct tim::trait::api_input_archive
/// \brief trait that configures the default input archive type for an entire API
/// specification, e.g. TIMEMORY_API (which is \code struct tim::api::native_tag
/// \endcode). The input archive format of individual components is determined from the
/// derived \ref tim::trait::input_archive
///
template <typename Api>
struct api_input_archive
{
    using type = TIMEMORY_INPUT_ARCHIVE;
};

/// \struct tim::trait::api_output_archive
/// \brief trait that configures the default output archive type for an entire API
/// specification, e.g. TIMEMORY_API (which is \code struct tim::api::native_tag
/// \endcode). The output archive format of individual components is determined from the
/// derived \ref tim::trait::output_archive
///
template <typename Api>
struct api_output_archive
{
    using default_type = TIMEMORY_OUTPUT_ARCHIVE;

    static constexpr bool is_default_v = std::is_same<default_type, type_list<>>::value;
    static constexpr bool is_pretty_v  = pretty_json<void>::value;

    using type =
        conditional_t<(is_default_v),
                      conditional_t<(is_pretty_v), cereal::PrettyJSONOutputArchive,
                                    cereal::MinimalJSONOutputArchive>,
                      default_type>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::input_archive
/// \brief trait that configures output archive type
///
template <typename T, typename Api>
struct input_archive
{
    using type = typename api_input_archive<Api>::type;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::output_archive
/// \brief trait that configures output archive type
///
template <typename T, typename Api>
struct output_archive
{
    using api_type = typename api_output_archive<Api>::type;

    using minimal_type = cereal::MinimalJSONOutputArchive;
    using pretty_type  = cereal::PrettyJSONOutputArchive;

    static constexpr bool is_pretty_v =
        (pretty_json<T>::value && pretty_json<void>::value);

    static constexpr bool is_json = (std::is_same<api_type, pretty_type>::value ||
                                     std::is_same<api_type, minimal_type>::value);

    using type =
        conditional_t<(is_json), conditional_t<(is_pretty_v), pretty_type, api_type>,
                      api_type>;
};

template <>
struct output_archive<manager, api::native_tag>
{
    using type = cereal::BaseJSONOutputArchive<cereal::PrettyJsonWriter>;
};

template <typename Api>
struct output_archive<manager, Api> : output_archive<manager, api::native_tag>
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::flat_storage
/// \brief trait that configures type to always flat_storage the call-tree
///
template <typename T>
struct flat_storage : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_sum
/// \brief trait that configures type to not report the accumulated value (useful if
/// meaningless). Only applies to text output.
///
template <typename T>
struct report_sum : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_mean
/// \brief trait that configures type to not report the mean value (useful if
/// meaningless). Only applies to text output.
///
template <typename T>
struct report_mean : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_values
/// \brief trait that allows runtime configuration of reporting certain types of values
/// (used in roofline). Only applies to text output.
///
template <typename T>
struct report_values
{
    using value_type = std::tuple<bool, bool>;

    static bool sum() { return std::get<0>(get_runtime_value()); }
    static void sum(bool val) { std::get<0>(get_runtime_value()) = val; }
    static bool mean() { return std::get<1>(get_runtime_value()); }
    static void mean(bool val) { std::get<1>(get_runtime_value()) = val; }

private:
    static value_type& get_runtime_value()
    {
        static value_type _instance{ report_sum<T>::value, report_mean<T>::value };
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_self
/// \brief trait that configures type to not report the % self field (useful if
/// meaningless). Only applies to text output.
///
template <typename T>
struct report_self : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_metric_name
/// \brief trait that configures type to not report the "METRIC" column, useful if
/// redundant). Only applies to text output.
///
template <typename T>
struct report_metric_name : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_units
/// \brief trait that configures type to not report the "UNITS" column (useful if always
/// empty). Only applies to text output.
///
template <typename T>
struct report_units : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::report_statistics
/// \brief trait that configures type to not report the "UNITS" column (useful if always
/// empty). Only applies to text output and does NOT affect whether statistics are
/// accumulated. For disabling statistics completely, see \ref
/// tim::trait::record_statistics and \ref tim::policy::record_statistics.
///
template <typename T>
struct report_statistics : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::supports_flamegraph
/// \brief trait that designates a type supports flamegraph output
///
template <typename T>
struct supports_flamegraph : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::derivation_types
/// \brief trait that designates the type supports calling assemble and derive member
/// functions with these types. Specializations MUST be structured as a
/// tim::type_list<...> of tim::type_list<...> where each inner type_list entry is
/// the list of component types required to perform a derivation.
/// \code{.cpp}
/// template <>
/// struct derivation_types<cpu_util>
/// {
///     // can derive its data when present alongside wall_clock + cpu_clock and/or
///     // wall_clock + user_clock + system_clock
///     using type = type_list<
///         type_list<wall_clock, cpu_clock>,
///         type_list<wall_clock, user_clock, system_clock>
///     >;
/// };
/// \endcode
///
template <typename T>
struct derivation_types : false_type
{
    static constexpr size_t size = 0;
    using type                   = std::tuple<type_list<>>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::python_args
/// \brief trait that designates the type supports these arguments from python.
/// Specializations MUST be structured as a tim::type_list<...> of tim::type_list<...>.
/// The first argument is a \ref TIMEMORY_OPERATION enumerated type and for each
/// inner \ref tim::type_list, a python member function for the stand-alone component
/// will be generated with those arguments. E.g. to create a custom store member function
/// accepting integer:
/// \code{.py}
/// foo = timemory.component.CaliperLoopMarker("example")
/// foo.start()
/// for i in range(10):
///     foo.store(i)    # store member function accepting integer
///     # ...
/// foo.stop()
/// \endcode
/// The type-trait specification would look like this:
/// \code{.cpp}
/// template <>
/// struct python_args<TIMEMORY_STORE, component::caliper_loop_marker>
/// {
///     using type = type_list<size_t>;
/// };
/// \endcode
///
template <int OpT, typename T>
struct python_args
{
    using type = type_list<type_list<>>;
};

//--------------------------------------------------------------------------------------//

template <int OpT, typename T>
using python_args_t = typename python_args<OpT, T>::type;

//--------------------------------------------------------------------------------------//
/// \struct tim::trait::cache
/// \brief trait that specifies the intermediate data type that will hold the relevant
/// data required by the component. This is useful for when multiple components
/// read different parts of the same file (e.g. /proc/<PID>/io) or an API
/// reports data in a larger data structure than the scope of the component (e.g. rusage)
/// but multiple components require access to this data structure
///
template <typename T>
struct cache
{
    using type = null_type;
};

//--------------------------------------------------------------------------------------//
//
//      determines if output is generated
//
//--------------------------------------------------------------------------------------//
/// \struct tim::trait::generates_output
/// \brief trait used to evaluate whether a component value type produces a useable value
template <typename T, typename V>
struct generates_output
{
    using value_type            = V;
    static constexpr bool value = (!concepts::is_null_type<value_type>::value);
};

template <typename T>
struct generates_output<T, void>
{
    using value_type            = void;
    static constexpr bool value = false;
};

template <typename T>
struct generates_output<T, null_type>
{
    using value_type            = null_type;
    static constexpr bool value = false;
};

template <typename T>
struct generates_output<T, type_list<>>
{
    // this is default evaluation from trait::data<T>::value_type
    using value_type            = typename T::value_type;
    static constexpr bool value = (!concepts::is_null_type<value_type>::value);
};

//--------------------------------------------------------------------------------------//
//
//      determines if storage should be implemented
//
//--------------------------------------------------------------------------------------//
/// \struct tim::trait::implements_storage
/// \brief This trait is used to determine whether the (expensive) instantiation of the
/// storage class happens
template <typename T, typename V>
struct implements_storage
{
    using value_type               = V;
    static constexpr bool avail_v  = trait::is_available<T>::value;
    static constexpr bool output_v = trait::generates_output<T, value_type>::value;
    static constexpr bool value    = (avail_v && output_v);
};

template <typename T>
struct implements_storage<T, type_list<>>
{
    using value_type               = typename T::value_type;
    static constexpr bool avail_v  = trait::is_available<T>::value;
    static constexpr bool output_v = trait::generates_output<T, value_type>::value;
    static constexpr bool value    = (avail_v && output_v);
};

template <typename T>
struct implements_storage<T, void>
{
    using value_type            = void;
    static constexpr bool value = false;
};

template <typename T>
struct implements_storage<T, null_type>
{
    using value_type            = null_type;
    static constexpr bool value = false;
};

//--------------------------------------------------------------------------------------//
}  // namespace trait
}  // namespace tim

//======================================================================================//
//
//                              Specifications
//
//======================================================================================//

#if !defined(TIMEMORY_DEFINE_CONCRETE_TRAIT)
#    define TIMEMORY_DEFINE_CONCRETE_TRAIT(TRAIT, COMPONENT, VALUE)                      \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct TRAIT<COMPONENT> : VALUE                                                  \
        {};                                                                              \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_DEFINE_TEMPLATE_TRAIT)
#    define TIMEMORY_DEFINE_TEMPLATE_TRAIT(TRAIT, COMPONENT, VALUE, TYPE)                \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TYPE T>                                                                \
        struct TRAIT<COMPONENT<T>> : VALUE                                               \
        {};                                                                              \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_DEFINE_VARIADIC_TRAIT)
#    define TIMEMORY_DEFINE_VARIADIC_TRAIT(TRAIT, COMPONENT, VALUE, TYPE)                \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TYPE... T>                                                             \
        struct TRAIT<COMPONENT<T...>> : VALUE                                            \
        {};                                                                              \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//
