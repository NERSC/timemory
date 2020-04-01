// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * \file timemory/components/ompt/types.hpp
 * \brief Declare the ompt component types
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
//
#include "timemory/components/ompt/backends.hpp"

//======================================================================================//
//
// TIMEMORY_DECLARE_TEMPLATE_COMPONENT(ompt_handle, typename Api = api::native_tag)
namespace tim
{
namespace component
{
template <typename Api = api::native_tag>
struct ompt_handle;
using ompt_native_handle = ompt_handle<api::native_tag>;
}  // namespace component
}  // namespace tim
//
// TIMEMORY_COMPONENT_ALIAS(ompt_native_handle, ompt_handle<api::native_tag>)
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_OMPT)
//
namespace tim
{
namespace trait
{
//
template <typename Api>
struct is_available<component::ompt_handle<Api>> : false_type
{};
//
template <>
struct is_available<component::ompt_native_handle> : false_type
{};
//
}  // namespace trait
}  // namespace tim
//
#endif
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(ompt_handle<api::native_tag>, OMPT_HANDLE, "ompt_handle",
                                 "ompt", "ompt_handle", "openmp", "openmp_tools")
//
//======================================================================================//
//
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/utility.hpp"
//
//======================================================================================//
//
namespace tim
{
namespace openmp
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Api = api::native_tag>
struct context_handler;
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api = api::native_tag>
struct callback_connector;
//
//--------------------------------------------------------------------------------------//
//
template <typename Enumeration>
static std::string
get_unknown_identifier(Enumeration eid)
{
    using type = Enumeration;
    auto&& ret =
        apply<std::string>::join("-", "unspecialized-enumeration",
                                 demangle<type>().c_str(), static_cast<int>(eid));
    return std::move(ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Enumeration>
struct identifier
{
    using type = Enumeration;
    static std::string get(type eid) { return get_unknown_identifier(eid); }
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct identifier<ompt_callbacks_t>
{
    using type      = ompt_callbacks_t;
    using key_map_t = std::unordered_map<int, std::string>;

    static std::string get(type eid)
    {
        static key_map_t _instance = {
            { ompt_callback_thread_begin, "thread_begin" },
            { ompt_callback_thread_end, "thread_end" },
            { ompt_callback_parallel_begin, "parallel_begin" },
            { ompt_callback_parallel_end, "parallel_end" },
            { ompt_callback_task_create, "task_create" },
            { ompt_callback_task_schedule, "task_schedule" },
            { ompt_callback_implicit_task, "implicit_task" },
            { ompt_callback_target, "target" },
            { ompt_callback_target_data_op, "target_data_op" },
            { ompt_callback_target_submit, "target_submit" },
            { ompt_callback_control_tool, "control_tool" },
            { ompt_callback_device_initialize, "device_initialize" },
            { ompt_callback_device_finalize, "device_finalize" },
            { ompt_callback_device_load, "device_load" },
            { ompt_callback_device_unload, "device_unload" },
            { ompt_callback_sync_region_wait, "sync_region_wait" },
            { ompt_callback_mutex_released, "mutex_released" },
            { ompt_callback_task_dependences, "task_dependences" },
            { ompt_callback_task_dependence, "task_dependence" },
            { ompt_callback_work, "work" },
            { ompt_callback_master, "master" },
            { ompt_callback_target_map, "target_map" },
            { ompt_callback_sync_region, "sync_region" },
            { ompt_callback_lock_init, "lock_init" },
            { ompt_callback_lock_destroy, "lock_destroy" },
            { ompt_callback_mutex_acquire, "mutex_acquire" },
            { ompt_callback_mutex_acquired, "mutex_acquired" },
            { ompt_callback_nest_lock, "nest_lock" },
            { ompt_callback_flush, "flush" },
            { ompt_callback_cancel, "cancel" },
        };

        auto itr = _instance.find(eid);
        return (itr == _instance.end()) ? get_unknown_identifier(eid) : itr->second;
    }
};
//
//--------------------------------------------------------------------------------------//
//
namespace mode
{
/// \class openmp::mode::begin_callback
/// \brief This is the beginning of a paired callback
struct begin_callback
{};
/// \class openmp::mode::end_callback
/// \brief This is the end of a paired callback
struct end_callback
{};
/// \class openmp::mode::measure_callback
/// \brief This is a sampling callback
struct measure_callback
{};
/// \class openmp::mode::endpoint_callback
/// \brief This is a callback whose first argument designates an endpoint
struct endpoint_callback
{};
}  // namespace mode
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Connector, typename Mode, typename... Args>
struct ompt_wrapper
{
    // using result_type    = ReturnType;
    using args_type      = std::tuple<Args...>;
    using component_type = Components;

    static void callback(Args... args) { Connector(Mode{}, args...); }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace openmp
}  // namespace tim
//
//======================================================================================//
//
#if !defined(TIMEMORY_OMPT_LINKAGE)
//
#    if defined(TIMEMORY_OMPT_SOURCE)
//
#        define TIMEMORY_OMPT_LINKAGE(...) extern "C" __VA_ARGS__
//
#    elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_OMPT_EXTERN)
//
#        define TIMEMORY_OMPT_LINKAGE(...) extern "C" __VA_ARGS__
//
#    else
//
#        define TIMEMORY_OMPT_LINKAGE(...) extern "C" __VA_ARGS__
//
#    endif
//
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_OMPT)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OMPT_LINKAGE(int)
ompt_initialize(ompt_function_lookup_t lookup, ompt_data_t* tool_data);
//
TIMEMORY_OMPT_LINKAGE(ompt_start_tool_result_t*)
ompt_start_tool(unsigned int omp_version, const char* runtime_version);
//
TIMEMORY_OMPT_LINKAGE(void)
ompt_finalize(ompt_data_t* tool_data);
//
//--------------------------------------------------------------------------------------//
//
#endif
//
//--------------------------------------------------------------------------------------//
//
