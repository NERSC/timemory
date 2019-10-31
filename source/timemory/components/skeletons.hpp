// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

/** \file components/skeletons.hpp
 * \headerfile components/skeletons.hpp "timemory/components/skeletons.hpp"
 *
 * These provide fake types for heavyweight types w.r.t. templates. In general,
 * if a component is templated or contains a lot of code, create a skeleton
 * and in \ref timemory/components/types.hpp use an #ifdef to provide the skeleton
 * instead. Also, make sure the component file is not directly included.
 * If the type uses callbacks, emulate the callbacks here.
 *
 */

#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "timemory/ert/types.hpp"

// clang-format off
namespace tim { namespace device { struct cpu; struct gpu; } }
// clang-format on

//======================================================================================//
//
namespace tim
{
namespace component
{
namespace skeleton
{
//--------------------------------------------------------------------------------------//

struct base
{};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct cuda
{};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct nvtx
{};

//--------------------------------------------------------------------------------------//

template <typename _Kind>
struct cupti_activity
{
    using activity_kind_t   = _Kind;
    using kind_vector_type  = std::vector<activity_kind_t>;
    using get_initializer_t = std::function<kind_vector_type()>;

    static get_initializer_t& get_initializer()
    {
        static auto              _lambda   = []() { return kind_vector_type{}; };
        static get_initializer_t _instance = _lambda;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct cupti_counters
{
    // short-hand for vectors
    using string_t = std::string;
    using strvec_t = std::vector<string_t>;
    /// a tuple of the <devices, events, metrics>
    using tuple_type = std::tuple<int, strvec_t, strvec_t>;
    /// function for setting all of device, metrics, and events
    using get_initializer_t = std::function<tuple_type()>;

    static get_initializer_t& get_initializer()
    {
        static auto              _lambda   = []() -> tuple_type { return tuple_type{}; };
        static get_initializer_t _instance = _lambda;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct gpu_roofline
{
    using device_t       = device::cpu;
    using clock_type     = real_clock;
    using ert_data_t     = ert::exec_data;
    using ert_params_t   = ert::exec_params;
    using ert_data_ptr_t = std::shared_ptr<ert_data_t>;

    // short-hand for variadic expansion
    template <typename _Tp>
    using ert_config_type = ert::configuration<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_counter_type = ert::counter<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_executor_type = ert::executor<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_callback_type = ert::callback<ert_executor_type<_Tp>>;

    // variadic expansion for ERT types
    using ert_config_t   = std::tuple<ert_config_type<_Types>...>;
    using ert_counter_t  = std::tuple<ert_counter_type<_Types>...>;
    using ert_executor_t = std::tuple<ert_executor_type<_Types>...>;
    using ert_callback_t = std::tuple<ert_callback_type<_Types>...>;

    static ert_config_t& get_finalizer()
    {
        static ert_config_t _instance;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <size_t N>
struct papi_array
{
    using event_list        = std::vector<int>;
    using get_initializer_t = std::function<event_list()>;
    static get_initializer_t& get_initializer()
    {
        static get_initializer_t _instance = []() { return event_list{}; };
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <int... _Types>
struct papi_tuple
{};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct cpu_roofline
{
    using device_t       = device::cpu;
    using clock_type     = real_clock;
    using ert_data_t     = ert::exec_data;
    using ert_params_t   = ert::exec_params;
    using ert_data_ptr_t = std::shared_ptr<ert_data_t>;

    // short-hand for variadic expansion
    template <typename _Tp>
    using ert_config_type = ert::configuration<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_counter_type = ert::counter<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_executor_type = ert::executor<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_callback_type = ert::callback<ert_executor_type<_Tp>>;

    // variadic expansion for ERT types
    using ert_config_t   = std::tuple<ert_config_type<_Types>...>;
    using ert_counter_t  = std::tuple<ert_counter_type<_Types>...>;
    using ert_executor_t = std::tuple<ert_executor_type<_Types>...>;
    using ert_callback_t = std::tuple<ert_callback_type<_Types>...>;

    static ert_config_t& get_finalizer()
    {
        static ert_config_t _instance;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <size_t _N, typename... _Types>
struct gotcha
{
    using config_t          = void;
    using get_initializer_t = std::function<config_t()>;
    static get_initializer_t& get_initializer()
    {
        static get_initializer_t _instance = []() {};
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct caliper
{};

//--------------------------------------------------------------------------------------//

}  // namespace skeleton

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Vp, typename... _Policies>
struct base;

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Policies>
struct base<_Tp, skeleton::base, _Policies...>
{
    static constexpr bool implements_storage_v = false;
    using Type                                 = _Tp;
    using value_type                           = void;
};

//--------------------------------------------------------------------------------------//

}  // namespace component
}  // namespace tim

#if !defined(TIMEMORY_USE_GOTCHA)

#    if !defined(TIMEMORY_C_GOTCHA)
#        define TIMEMORY_C_GOTCHA(...)
#    endif

#    if !defined(TIMEMORY_DERIVED_GOTCHA)
#        define TIMEMORY_DERIVED_GOTCHA(...)
#    endif

#    if !defined(TIMEMORY_CXX_GOTCHA)
#        define TIMEMORY_CXX_GOTCHA(...)
#    endif

#    if !defined(TIMEMORY_CXX_MEMFUN_GOTCHA)
#        define TIMEMORY_CXX_MEMFUN_GOTCHA(...)
#    endif

#    if !defined(TIMEMORY_C_GOTCHA_TOOL)
#        define TIMEMORY_C_GOTCHA_TOOL(...)
#    endif

#    if !defined(TIMEMORY_CXX_GOTCHA_TOOL)
#        define TIMEMORY_CXX_GOTCHA_TOOL(...)
#    endif

#endif
