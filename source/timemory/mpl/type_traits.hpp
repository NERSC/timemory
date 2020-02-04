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

#include "timemory/components/types.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/serializer.hpp"

#include <type_traits>

//======================================================================================//
//
//                                 Type Traits
//
//======================================================================================//

namespace tim
{
template <typename _Tp>
struct statistics;

class manager;

namespace trait
{
//--------------------------------------------------------------------------------------//
/// this is a helper trait
///
template <typename>
struct sfinae_true : true_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that an implementation (e.g. PAPI) is available
///
template <typename _Tp>
struct is_available : true_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that an implementation (e.g. PAPI) is available
///
template <typename _Tp>
struct runtime_enabled
{
    static bool get() { return get_runtime_value(); }
    static void set(bool val) { get_runtime_value() = val; }

private:
    static bool& get_runtime_value()
    {
        static bool _instance = is_available<_Tp>::value;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that updating w.r.t. another instance should
/// be a max of the two instances
//
template <typename _Tp>
struct record_max : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that data is an array type
///
template <typename _Tp>
struct array_serialization : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component requires the prefix to be set right after
/// construction. Types with this trait must contain a member string variable named
/// prefix
///
template <typename _Tp>
struct requires_prefix : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component handles it's label when printing
///
template <typename _Tp>
struct custom_label_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component includes it's units when printing
///
template <typename _Tp>
struct custom_unit_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component includes it's laps when printing
///
template <typename _Tp>
struct custom_laps_printing : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority when starting the type w.r.t.
/// other types.
///
template <typename _Tp>
struct start_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority when stopping the type w.r.t.
/// other types.
///
template <typename _Tp>
struct stop_priority : std::integral_constant<int, 0>
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the width and precision should follow env specified
/// timing settings
///
template <typename _Tp>
struct is_timing_category : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the width and precision should follow env specified
/// memory settings
///
template <typename _Tp>
struct is_memory_category : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units should follow env specified timing settings
///
template <typename _Tp>
struct uses_timing_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units should follow env specified memory settings
///
template <typename _Tp>
struct uses_memory_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units are a percentage
///
template <typename _Tp>
struct uses_percent_units : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates a type should always print a JSON output
///
template <typename _Tp>
struct requires_json : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type is a gotcha... ONLY gotcha should set to TRUE!
///
template <typename _Tp>
struct is_gotcha : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type supports calling a function with a certain
/// set of argument types (passed via a tuple)
///
template <typename _Tp, typename _Tuple>
struct supports_args : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type supports changing the record() static function
/// per-instance
///
template <typename _Tp>
struct supports_custom_record : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that get() returns an iterable type
///
template <typename _Tp>
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
template <typename _Tp>
struct secondary_data : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component only has relevant values if it is not collapsed
/// into the master thread
///
template <typename _Tp>
struct thread_scope_only : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component will be providing a separate split load(...) and
/// store(...) for serialization so the base class should not provide a generic
/// serialize(...) function
///
template <typename _Tp>
struct split_serialization : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component will accumulate a min/max
///
template <typename _Tp>
struct record_statistics : default_record_statistics_type
{};

//--------------------------------------------------------------------------------------//
/// trait that specifies the data type of the statistics
///
template <typename _Tp>
struct statistics
{
    using type = std::tuple<>;
};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component support sampling
///
template <typename _Tp>
struct sampler : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies the component samples a measurement from a file
///
template <typename _Tp>
struct file_sampler : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait the designates the units
///
template <typename _Tp>
struct units
{
    using type         = int64_t;
    using display_type = std::string;
};

//--------------------------------------------------------------------------------------//
/// trait the configures echo_measurement usage
///
template <typename _Tp>
struct echo_enabled : true_type
{};

//--------------------------------------------------------------------------------------//
/// trait the configures whether JSON output uses pretty print. If set to false_type
/// then the JSON will be compact
///
template <typename T>
struct pretty_json : std::true_type
{};

//--------------------------------------------------------------------------------------//
/// trait the configures output archive type
///
template <typename _Tp>
struct input_archive
{
    using type    = cereal::JSONInputArchive;
    using pointer = std::shared_ptr<type>;

    static pointer get(std::istream& is) { return std::make_shared<type>(is); }
};

//--------------------------------------------------------------------------------------//
/// trait the configures output archive type
///
template <typename _Tp>
struct output_archive
{
    using subtype =
        conditional_t<(!pretty_json<_Tp>::value || !pretty_json<void>::value) &&
                          !std::is_same<_Tp, manager>::value,
                      cereal::MinimalJsonWriter, cereal::PrettyJsonWriter>;
    using type        = cereal::BaseJSONOutputArchive<subtype>;
    using pointer     = std::shared_ptr<type>;
    using option_type = typename type::Options;
    using indent_type = typename option_type::IndentChar;

    static unsigned int& precision()
    {
        static unsigned int value = 16;
        return value;
    }
    static unsigned int& indent_length()
    {
        static unsigned int value = 2;
        return value;
    }
    static indent_type& indent_char()
    {
        static indent_type value = indent_type::space;
        return value;
    }

    static pointer get(std::ostream& os)
    {
        constexpr auto spacing = option_type::IndentChar::space;
        //  Option args: precision, spacing, indent size
        option_type opts(precision(), spacing, indent_length());
        return std::make_shared<type>(os, opts);
    }
};

//--------------------------------------------------------------------------------------//
/// explicit specialization for PrettyJSONOutputArchive
///
template <>
struct output_archive<cereal::PrettyJSONOutputArchive>
{
    using subtype     = cereal::PrettyJsonWriter;
    using type        = cereal::PrettyJSONOutputArchive;
    using pointer     = std::shared_ptr<type>;
    using option_type = typename type::Options;
    using indent_type = typename option_type::IndentChar;

    static unsigned int& precision()
    {
        static unsigned int value = 16;
        return value;
    }
    static unsigned int& indent_length()
    {
        static unsigned int value = 2;
        return value;
    }
    static indent_type& indent_char()
    {
        static indent_type value = indent_type::space;
        return value;
    }

    static pointer get(std::ostream& os)
    {
        //  Option args: precision, spacing, indent size
        option_type opts(precision(), indent_type(), indent_length());
        return std::make_shared<type>(os, opts);
    }
};

//--------------------------------------------------------------------------------------//
/// explicit specialization for MinimalJSONOutputArchive
///
template <>
struct output_archive<cereal::MinimalJSONOutputArchive>
{
    using subtype     = cereal::MinimalJsonWriter;
    using type        = cereal::MinimalJSONOutputArchive;
    using pointer     = std::shared_ptr<type>;
    using option_type = typename type::Options;
    using indent_type = typename option_type::IndentChar;

    static unsigned int& precision()
    {
        static unsigned int value = 16;
        return value;
    }
    static unsigned int& indent_length()
    {
        static unsigned int value = 0;
        return value;
    }
    static indent_type& indent_char()
    {
        static indent_type value = indent_type::space;
        return value;
    }

    static pointer get(std::ostream& os)
    {
        //  Option args: precision, spacing, indent size
        //  The last two options are meaningless for the minimal writer
        option_type opts(precision(), indent_type(), indent_length());
        return std::make_shared<type>(os, opts);
    }
};

//--------------------------------------------------------------------------------------//
/// trait the configures type to always flat_storage the call-tree
///
template <typename _Tp>
struct flat_storage : false_type
{};

//--------------------------------------------------------------------------------------//
/// trait the configures type to not report the accumulated value (useful if meaningless)
///
template <typename _Tp>
struct report_sum : true_type
{};

//--------------------------------------------------------------------------------------//
/// trait the configures type to not report the mean value (useful if meaningless)
///
template <typename _Tp>
struct report_mean : true_type
{};

//--------------------------------------------------------------------------------------//

template <typename _Trait>
inline std::string
as_string()
{
    constexpr bool _val = _Trait::value;
    return (_val) ? "true" : "false";
}

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

template <typename _Tp, typename _Vp>
struct generates_output
{
    static constexpr bool value = (!(std::is_same<_Vp, void>::value));
};

//--------------------------------------------------------------------------------------//
//
//      determines if storage should be implemented
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Vp>
struct implements_storage
{
    static constexpr bool value =
        (trait::is_available<_Tp>::value && !(std::is_same<_Vp, void>::value));
};

}  // namespace tim

//======================================================================================//
//
//                              Specifications
//
//======================================================================================//

//--------------------------------------------------------------------------------------//
//
//                              RECORD MAX
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::peak_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::stack_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::data_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::virtual_memory, true_type)

//--------------------------------------------------------------------------------------//
//
//                              REPORT SUM
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(report_sum, component::current_peak_rss, false_type)

//--------------------------------------------------------------------------------------//
//
//                              REPORT MEAN
//
//--------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------//
//
//                              SAMPLER
//
//--------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------//
//
//                              FILE SAMPLER
//
//--------------------------------------------------------------------------------------//

#if defined(_LINUX) || (defined(_UNIX) && !defined(_MACOS))

TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::data_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::written_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::virtual_memory, true_type)

#endif

//--------------------------------------------------------------------------------------//
//
//                              START PRIORITY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, component::cuda_event,
                               priority_constant<128>)

//--------------------------------------------------------------------------------------//
//
//                              STOP PRIORITY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, component::cuda_event,
                               priority_constant<-128>)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::written_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::current_peak_rss,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::cupti_counters, true_type)
TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_unit_printing, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LABEL PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::written_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::cupti_counters,
                               true_type)
TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_label_printing, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LAPS PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_laps_printing, component::trip_count, true_type)

//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_TEMPLATE_TRAIT(array_serialization, component::papi_array, true_type,
                               size_t)
TIMEMORY_DEFINE_VARIADIC_TRAIT(array_serialization, component::papi_tuple, true_type, int)
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::cupti_counters, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::written_bytes, true_type)

//--------------------------------------------------------------------------------------//
//
//                              THREAD SCOPE ONLY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(thread_scope_only, component::thread_cpu_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(thread_scope_only, component::thread_cpu_util, true_type)

//--------------------------------------------------------------------------------------//
//
//                              REQUIRES JSON
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(requires_json, component::cpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_json, component::cpu_roofline_flops, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_json, component::cpu_roofline_sp_flops, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_json, component::cpu_roofline_dp_flops, true_type)

//--------------------------------------------------------------------------------------//
//
//                              SUPPORTS CUSTOM RECORD
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(supports_custom_record, component::cpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_custom_record, component::cpu_roofline_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_custom_record, component::cpu_roofline_sp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_custom_record, component::cpu_roofline_dp_flops,
                               true_type)

TIMEMORY_DEFINE_VARIADIC_TRAIT(supports_custom_record, component::gpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_custom_record, component::gpu_roofline_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_custom_record, component::gpu_roofline_hp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_custom_record, component::gpu_roofline_sp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_custom_record, component::gpu_roofline_dp_flops,
                               true_type)

//--------------------------------------------------------------------------------------//
//
//                              ITERABLE MEASUREMENT
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(iterable_measurement, component::gpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(iterable_measurement, component::gpu_roofline_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(iterable_measurement, component::gpu_roofline_hp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(iterable_measurement, component::gpu_roofline_sp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(iterable_measurement, component::gpu_roofline_dp_flops,
                               true_type)

//--------------------------------------------------------------------------------------//
//
//                              SPLIT SERIALIZATION
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(split_serialization, component::gpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(split_serialization, component::gpu_roofline_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(split_serialization, component::gpu_roofline_hp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(split_serialization, component::gpu_roofline_sp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(split_serialization, component::gpu_roofline_dp_flops,
                               true_type)

//--------------------------------------------------------------------------------------//
//
//                              SECONDARY DATA
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::cupti_activity, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::cupti_counters, true_type)
TIMEMORY_DEFINE_VARIADIC_TRAIT(secondary_data, component::gpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::gpu_roofline_flops, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::gpu_roofline_hp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::gpu_roofline_sp_flops,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::gpu_roofline_dp_flops,
                               true_type)

//--------------------------------------------------------------------------------------//
//
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::caliper, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::nvtx_marker, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::gperf_heap_profiler, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::likwid_perfmon, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::likwid_nvmon, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::vtune_event, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::vtune_frame, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::tau_marker, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::malloc_gotcha, true_type)

//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
//      PAPI
//
#if !defined(TIMEMORY_USE_PAPI)
TIMEMORY_DEFINE_TEMPLATE_TRAIT(is_available, component::papi_array, false_type, size_t)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::papi_tuple, false_type, int)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::cpu_roofline, false_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_sp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_dp_flops, false_type)
#endif

//
//      CUDA
//
#if !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cuda_event, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cuda_profiler, false_type)
#endif

//
//      CUDA and NVTX
//
#if !defined(TIMEMORY_USE_NVTX) || !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::nvtx_marker, false_type)
#endif

//
//      CUDA and CUPTI
//
#if !defined(TIMEMORY_USE_CUPTI) || !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_counters, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_activity, false_type)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::gpu_roofline, false_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_hp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_sp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_dp_flops, false_type)
#endif

//
//      CALIPER
//
#if !defined(TIMEMORY_USE_CALIPER)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::caliper, false_type)
#endif

//
//      GPERF and GPERF_HEAP_PROFILER
//
#if !defined(TIMEMORY_USE_GPERF) && !defined(TIMEMORY_USE_GPERF_HEAP_PROFILER)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gperf_heap_profiler, false_type)
#endif

//
//      GPERF AND GPERF_CPU_PROFILER
//
#if !defined(TIMEMORY_USE_GPERF) && !defined(TIMEMORY_USE_GPERF_CPU_PROFILER)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gperf_cpu_profiler, false_type)
#endif

//
//      LIKWID
//
#if !defined(TIMEMORY_USE_LIKWID)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::likwid_perfmon, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::likwid_nvmon, false_type)
#else
#    if !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::likwid_nvmon, false_type)
#    endif
#endif

//
//      VTUNE
//
#if !defined(TIMEMORY_USE_VTUNE)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::vtune_event, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::vtune_frame, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::vtune_profiler, false_type)
#endif

//
//      TAU
//
#if !defined(TIMEMORY_USE_TAU)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::tau_marker, false_type)
#endif

//
//      WINDOWS (non-UNIX)
//
#if !defined(_UNIX)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::stack_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::data_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_in, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_out, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_major_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_minor_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_recv, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_sent, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_signals, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_swap, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::read_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::written_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::virtual_memory, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_mode_time, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::kernel_mode_time, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::current_peak_rss, false_type)

#endif

//
//      UNIX
//
#if defined(UNIX)

/// \param TIMEMORY_USE_UNMAINTAINED_RUSAGE
/// \brief This macro enables the globally disable rusage structures that are
/// unmaintained by the Linux kernel and are zero on macOS
///
#    if !defined(TIMEMORY_USE_UNMAINTAINED_RUSAGE)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::stack_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::data_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_swap, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_recv, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_sent, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_signals, false_type)

#        if defined(_MACOS)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_in, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_out, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::read_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::written_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::virtual_memory, false_type)

#        endif
#    endif  // !defined(TIMEMORY_USE_UNMAINTAINED_RUSAGE)

#endif

//======================================================================================//

namespace tim
{
namespace trait
{
//--------------------------------------------------------------------------------------//
//
//                              GOTCHA
//
//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_GOTCHA
//
#if !defined(TIMEMORY_USE_GOTCHA)

template <size_t _N, typename _Comp, typename _Diff>
struct is_available<component::gotcha<_N, _Comp, _Diff>> : false_type
{};

#endif  // TIMEMORY_USE_GOTCHA

template <size_t _N, typename _Comp, typename _Diff>
struct is_gotcha<component::gotcha<_N, _Comp, _Diff>> : true_type
{};

// start gotchas later
template <size_t _N, typename _Comp, typename _Diff>
struct start_priority<component::gotcha<_N, _Comp, _Diff>> : priority_constant<256>
{};

// stop gotchas early
template <size_t _N, typename _Comp, typename _Diff>
struct stop_priority<component::gotcha<_N, _Comp, _Diff>> : priority_constant<-256>
{};

//--------------------------------------------------------------------------------------//
//
//                              User-bundle
//
//--------------------------------------------------------------------------------------//

template <size_t _Idx, typename _Type>
struct requires_prefix<component::user_bundle<_Idx, _Type>> : true_type
{};

//--------------------------------------------------------------------------------------//
}  // namespace trait
}  // namespace tim

//======================================================================================//

#include "timemory/mpl/bits/type_traits.hpp"

//======================================================================================//
