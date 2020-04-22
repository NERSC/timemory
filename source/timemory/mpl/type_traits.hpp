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
#include "timemory/mpl/types.hpp"
#include "timemory/utility/serializer.hpp"
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
/// trait that signifies that a component has an accumulation value
///
template <typename T>
struct base_has_accum : true_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component has an "last" value which may be different
/// than the "value" value
///
template <typename T>
struct base_has_last : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that an implementation (e.g. PAPI) is available
///
template <typename T>
struct is_available : TIMEMORY_DEFAULT_AVAILABLE
{};

template <typename T>
struct is_available<T*> : is_available<std::remove_pointer_t<T>>
{};

//--------------------------------------------------------------------------------------//
/// \class data
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
/// trait that signifies that an implementation is enabled at runtime
///
template <typename T>
struct runtime_enabled
{
    static bool get() { return get_runtime_value(); }
    static void set(bool val) { get_runtime_value() = val; }

private:
    static bool& get_runtime_value()
    {
        static bool _instance = is_available<T>::value;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that updating w.r.t. another instance should
/// be a max of the two instances
//
template <typename T>
struct record_max : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that data is an array type
///
template <typename T>
struct array_serialization : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component requires the prefix to be set right after
/// construction. Types with this trait must contain a member string variable named
/// prefix
///
template <typename T>
struct requires_prefix : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component handles it's label when printing
///
template <typename T>
struct custom_label_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component includes it's units when printing
///
template <typename T>
struct custom_unit_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component includes it's laps when printing
///
template <typename T>
struct custom_laps_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority when starting the type w.r.t.
/// other types.
///
template <typename T>
struct start_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority when stopping the type w.r.t.
/// other types.
///
template <typename T>
struct stop_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the width and precision should follow env specified
/// timing settings
///
template <typename T>
struct is_timing_category : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the width and precision should follow env specified
/// memory settings
///
template <typename T>
struct is_memory_category : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units should follow env specified timing settings
///
template <typename T>
struct uses_timing_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units should follow env specified memory settings
///
template <typename T>
struct uses_memory_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units are a percentage
///
template <typename T>
struct uses_percent_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates a type should always print a JSON output
///
template <typename T>
struct requires_json : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type is a gotcha... ONLY gotcha should set to TRUE!
///
template <typename T>
struct is_gotcha : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type is a user-bundle... ONLY user-bundles should be TRUE!
///
template <typename T>
struct is_user_bundle : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type supports calling a function with a certain
/// set of argument types (passed via a tuple)
///
template <typename T, bool>
struct component_value_type;

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

template <typename T>
struct collects_data
{
    using type = T;
    using value_type =
        typename component_value_type<T, is_available<T>::value>::value_type;
    static constexpr bool value =
        (!std::is_same<value_type, void>::value &&
         !std::is_same<value_type, void*>::value &&
         !std::is_same<value_type, type_list<>>::value && is_available<T>::value);
    static_assert(std::is_void<value_type>::value != value,
                  "Error value_type is void and value is true");
};

//--------------------------------------------------------------------------------------//
/// trait that designates the type supports calling a function with a certain
/// set of argument types (passed via a tuple)
///
template <typename T, typename Tuple>
struct supports_args : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type supports changing the record() static function
/// per-instance
///
template <typename T>
struct supports_custom_record : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that get() returns an iterable type
///
template <typename T>
struct iterable_measurement : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that secondary data resembling the original data
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
/// trait that signifies the component only has relevant values if it is not collapsed
/// into the master thread
///
template <typename T>
struct thread_scope_only : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component will be providing a separate split load(...) and
/// store(...) for serialization so the base class should not provide a generic
/// serialize(...) function
///
template <typename T>
struct custom_serialization : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component will accumulate a min/max
///
template <typename T>
struct record_statistics : default_record_statistics_type
{};

//--------------------------------------------------------------------------------------//
/// trait that specifies the data type of the statistics
///
template <typename T>
struct statistics
{
    using type = std::tuple<>;
};

//--------------------------------------------------------------------------------------//
/// trait that will suppress compilation error in operation::add_statistics<Component>
/// if the data type passed does not match statistics<Component>::type
///
template <typename T>
struct permissive_statistics : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component support sampling
///
template <typename T>
struct sampler : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component samples a measurement from a file
///
template <typename T>
struct file_sampler : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait the designates the units
///
template <typename T>
struct units
{
    using type         = int64_t;
    using display_type = std::string;
};

//--------------------------------------------------------------------------------------//
/// trait the configures echo_measurement usage
///
template <typename T>
struct echo_enabled : true_type
{};

//--------------------------------------------------------------------------------------//
/// trait the configures whether JSON output uses pretty print. If set to false_type
/// then the JSON will be compact
///
template <typename T>
struct pretty_json : std::false_type
{};

template <typename Api>
struct api_input_archive
{
    using type = TIMEMORY_INPUT_ARCHIVE;
};

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
/// trait the configures output archive type
///
template <typename T, typename Api>
struct input_archive
{
    using type = typename api_input_archive<Api>::type;
};

//--------------------------------------------------------------------------------------//
/// trait the configures output archive type
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
/// trait the configures type to always flat_storage the call-tree
///
template <typename T>
struct flat_storage : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait the configures type to not report the accumulated value (useful if meaningless)
///
template <typename T>
struct report_sum : true_type
{};

//--------------------------------------------------------------------------------------//
/// trait the configures type to not report the mean value (useful if meaningless)
///
template <typename T>
struct report_mean : true_type
{};

//--------------------------------------------------------------------------------------//
/// trait that allows runtime configuration of reporting certain types of values
/// (used in roofline)
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
/// trait that designates a type supports flamegraph output
///
template <typename T>
struct supports_flamegraph : false_type
{};

//--------------------------------------------------------------------------------------//

template <typename TraitT>
inline std::string
as_string()
{
    constexpr bool _val = TraitT::value;
    return (_val) ? "true" : "false";
}

//--------------------------------------------------------------------------------------//
/// trait that designates the type supports calling assemble and derive member functions
/// with these types. Specializations MUST be structured as a std::tuple<...> of
/// tim::type_list<...>
///
template <typename T>
struct derivation_types : false_type
{
    static constexpr size_t size = 0;
    using type                   = std::tuple<type_list<>>;
};

//--------------------------------------------------------------------------------------//
}  // namespace trait
}  // namespace tim

//======================================================================================//
//
//                              Derived helper traits
//
//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//      determines if output is generated
//
//--------------------------------------------------------------------------------------//

template <typename T, typename V>
struct generates_output
{
    using value_type            = V;
    static constexpr bool value = (!(std::is_same<V, void>::value));
};

template <typename T>
struct generates_output<T, type_list<>>
{
    using V                     = typename T::value_type;
    using value_type            = V;
    static constexpr bool value = (!(std::is_same<V, void>::value));
};

template <typename T>
struct generates_output<T, void>
{
    using V                     = void;
    using value_type            = V;
    static constexpr bool value = false;
};

//--------------------------------------------------------------------------------------//
//
//      determines if storage should be implemented
//
//--------------------------------------------------------------------------------------//

template <typename T, typename V>
struct implements_storage
{
    using value_type               = V;
    static constexpr bool avail_v  = trait::is_available<T>::value;
    static constexpr bool output_v = generates_output<T, V>::value;
    static constexpr bool value    = (avail_v && output_v);
};

template <typename T>
struct implements_storage<T, type_list<>>
{
    using V                        = typename T::value_type;
    using value_type               = V;
    static constexpr bool avail_v  = trait::is_available<T>::value;
    static constexpr bool output_v = generates_output<T, V>::value;
    static constexpr bool value    = (avail_v && output_v);
};

template <typename T>
struct implements_storage<T, void>
{
    using V                     = void;
    using value_type            = V;
    static constexpr bool value = false;
};

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
