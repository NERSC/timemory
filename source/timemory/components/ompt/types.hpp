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

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/data_tracker/types.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/ompt/backends.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"

TIMEMORY_DECLARE_TEMPLATE_COMPONENT(user_bundle, size_t Idx, typename Tag = TIMEMORY_API)
//
TIMEMORY_BUNDLE_INDEX(ompt_bundle_idx, 11110)
//
TIMEMORY_COMPONENT_ALIAS(user_ompt_bundle,
                         user_bundle<ompt_bundle_idx, project::timemory>)
//
//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SET_COMPONENT_API(component::user_ompt_bundle, tpls::openmp, category::external,
                           os::agnostic)
//
namespace tim
{
namespace component
{
template <typename Api = TIMEMORY_API>
struct ompt_handle;

template <typename Api>
struct ompt_data_tracker;

struct ompt_target_data_tag
{};

using ompt_native_handle       = ompt_handle<TIMEMORY_API>;
using ompt_native_data_tracker = ompt_data_tracker<TIMEMORY_API>;
//
using ompt_data_tracker_t = data_tracker<int64_t, ompt_target_data_tag>;
//
}  // namespace component
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_OMPT)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::openmp, false_type)
#endif
//
namespace tim
{
namespace trait
{
//
//--------------------------------------------------------------------------------------//
/// trait for configuring OMPT components
///
template <typename T>
struct ompt_handle
{
    using type =
        component_tuple<component::user_ompt_bundle, component::ompt_native_data_tracker>;
};
//
#if !defined(TIMEMORY_USE_OMPT)
//
template <typename Api>
struct is_available<component::ompt_handle<Api>> : false_type
{};
//
template <>
struct is_available<component::ompt_native_handle> : false_type
{};
//
template <typename Api>
struct is_available<component::ompt_data_tracker<Api>> : false_type
{};
//
template <>
struct is_available<component::ompt_native_data_tracker> : false_type
{};
//
#endif
//
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(ompt_handle<TIMEMORY_API>, TIMEMORY_OMPT_HANDLE,
                                 "ompt_handle", "ompt", "ompt_handle", "openmp",
                                 "openmp_tools")
//
//======================================================================================//
//
#include "timemory/mpl/apply.hpp"
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
namespace mode
{
/// \struct tim::openmp::mode::begin_callback
/// \brief This is the beginning of a paired callback
struct begin_callback
{};
/// \struct tim::openmp::mode::end_callback
/// \brief This is the end of a paired callback
struct end_callback
{};
/// \struct tim::openmp::mode::store_callback
/// \brief This is a callback that just stores some data
struct store_callback
{};
/// \struct tim::openmp::mode::endpoint_callback
/// \brief This is a callback whose first argument designates an endpoint
struct endpoint_callback
{};
}  // namespace mode
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::openmp::context_handler
/// \brief this struct provides the methods through which a unique identifier and
/// a label are generated for each OMPT callback.
///
template <typename Api = TIMEMORY_API>
struct context_handler;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::openmp::callback_connector
/// \brief this struct provides the routines through which timemory components
/// are applied to the callbacks
///
template <typename Components, typename Api = TIMEMORY_API>
struct callback_connector;
//
//--------------------------------------------------------------------------------------//
//
/// \fn void openmp::user_context_callback(Handler& h, std::string& key, Args... args)
/// \brief These functions can be specialized an overloaded for quick access
/// to the the openmp callbacks. The first function (w/ string) is invoked by every
/// openmp callback. The other versions (w/ mode) is invoked depending on how
/// each callback is configured
///
template <typename Handler, typename... Args>
void
user_context_callback(Handler& handle, std::string& key, Args... args)
{
    consume_parameters(handle, key, args...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Handler, typename... Args>
void
user_context_callback(Handler& handle, mode::begin_callback, Args... args)
{
    auto functor = [&args...](Tp* c) {
        c->construct(args...);
        c->store(args...);
        c->start();
        c->audit(args...);
    };
    handle.template construct<Tp>(functor);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Handler, typename... Args>
void
user_context_callback(Handler& handle, mode::end_callback, Args... args)
{
    auto functor = [&args...](Tp* c) {
        c->audit(args...);
        c->stop();
    };
    handle.template destroy<Tp>(functor);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Handler, typename... Args>
void
user_context_callback(Handler& handle, mode::store_callback, Args... args)
{
    Tp c(handle.key());
    c.store(args...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Handler, typename... Args>
void
user_context_callback(Handler&              handle, mode::endpoint_callback,
                      ompt_scope_endpoint_t endp, Args... args)
{
    if(endp == ompt_scope_begin)
    {
        auto functor = [&endp, &args...](Tp* c) {
            c->construct(endp, args...);
            c->store(endp, args...);
            c->start();
            c->audit(endp, args...);
        };
        handle.template construct<Tp>(functor);
    }
    else if(endp == ompt_scope_end)
    {
        auto functor = [&endp, &args...](Tp* c) {
            c->audit(endp, args...);
            c->stop();
        };
        handle.template destroy<Tp>(functor);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Handler, typename Arg, typename... Args>
void
user_context_callback(Handler& handle, mode::endpoint_callback, Arg arg,
                      ompt_scope_endpoint_t endp, Args... args)
{
    if(endp == ompt_scope_begin)
    {
        auto functor = [&arg, &endp, &args...](Tp* c) {
            c->construct(endp, args...);
            c->store(endp, args...);
            c->start();
            c->audit(arg, endp, args...);
        };
        handle.template construct<Tp>(functor);
    }
    else if(endp == ompt_scope_end)
    {
        auto functor = [&arg, &endp, &args...](Tp* c) {
            c->audit(arg, endp, args...);
            c->stop();
        };
        handle.template destroy<Tp>(functor);
    }
}
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::openmp::ompt_wrapper
/// \brief this struct provides the static callback function for OMPT which creates
/// a temporary instance of the connector, e.g. \ref openmp::callback_connector,
/// whose constructor creates a temporary instance of the context handler
/// in order to create a unique identifier and a label and then instruments the callback
/// based on the \tparam Mode template parameter.
///
template <typename Components, typename Connector, typename Mode, typename... Args>
struct ompt_wrapper
{
    using args_type      = std::tuple<Mode, Args...>;
    using component_type = Components;

    static void callback(Args... args)
    {
        constexpr auto can_ctor = std::is_constructible<Connector, Mode, Args...>::value;
        static_assert(can_ctor,
                      "Error! Cannot construct the connector with given arguments");
        Connector(Mode{}, args...);
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace openmp
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_OMPT)
//
//--------------------------------------------------------------------------------------//

extern "C" int
ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                ompt_data_t* tool_data);

extern "C" ompt_start_tool_result_t*
ompt_start_tool(unsigned int omp_version, const char* runtime_version);

extern "C" void
ompt_finalize(ompt_data_t* tool_data);

//--------------------------------------------------------------------------------------//
//
#endif
//
//--------------------------------------------------------------------------------------//
//
