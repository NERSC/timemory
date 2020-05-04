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
#include "timemory/mpl/type_traits.hpp"
//
#include "timemory/components/data_tracker/types.hpp"
#include "timemory/components/ompt/backends.hpp"

//======================================================================================//
//
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(user_bundle, size_t Idx,
                                    typename Tag = api::native_tag)
//
TIMEMORY_BUNDLE_INDEX(ompt_bundle_idx, 11110)
//
TIMEMORY_COMPONENT_ALIAS(user_ompt_bundle, user_bundle<ompt_bundle_idx, api::native_tag>)
//
//======================================================================================//
//
namespace tim
{
namespace component
{
template <typename Api = api::native_tag>
struct ompt_handle;

template <typename Api>
struct ompt_data_tracker;

struct ompt_target_data_op_tag
{};
struct ompt_target_data_map_tag
{};
struct ompt_target_data_submit_tag
{};

using ompt_native_handle         = ompt_handle<api::native_tag>;
using ompt_native_data_tracker   = ompt_data_tracker<api::native_tag>;
using ompt_data_op_tracker_t     = data_tracker<int64_t, ompt_target_data_op_tag>;
using ompt_data_map_tracker_t    = data_tracker<int64_t, ompt_target_data_map_tag>;
using ompt_data_submit_tracker_t = data_tracker<int64_t, ompt_target_data_submit_tag>;

}  // namespace component
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
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
TIMEMORY_PROPERTY_SPECIALIZATION(ompt_handle<api::native_tag>, OMPT_HANDLE, "ompt_handle",
                                 "ompt", "ompt_handle", "openmp", "openmp_tools")
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
/// \class openmp::mode::measure_callback
/// \brief This is a sampling callback
struct store_callback
{};
/// \class openmp::mode::endpoint_callback
/// \brief This is a callback whose first argument designates an endpoint
struct endpoint_callback
{};
}  // namespace mode
//
//--------------------------------------------------------------------------------------//
//
/// \class openmp::context_handler
/// \brief this struct provides the methods through which a unique identifier and
/// a label are generated for each OMPT callback.
///
template <typename Api = api::native_tag>
struct context_handler;
//
//--------------------------------------------------------------------------------------//
//
/// \class openmp::callback_connector
/// \brief this struct provides the routines through which timemory components
/// are applied to the callbacks
///
template <typename Components, typename Api = api::native_tag>
struct callback_connector;
//
//--------------------------------------------------------------------------------------//
//
/// \fn openmp::user_context_callback
/// \brief this is a dummy implementation which the user can use to customize the labeling
/// or identifier. The \param id argument is the unique hash associated with
/// callback, the \param key argument is the label passed to timemory. All remaining
/// arguments should be specialized to the particular callback.
///
template <typename Handler, typename... Args>
void
user_context_callback(Handler& handle, size_t& id, std::string& key, Args... args)
{
    consume_parameters(handle, id, key, args...);
}
//
//--------------------------------------------------------------------------------------//
//
/// \class openmp::ompt_wrapper
/// \brief this struct provides the static callback function for OMPT which creates
/// a temporary instance of the connector, e.g. \ref openmp::callback_connector,
/// whose constructor creates a temporary instance of the context handler
/// in order to create a unique identifier and a label and then instruments the callback
/// based on the \param Mode template parameter.
///
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
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_OMPT)
//
//--------------------------------------------------------------------------------------//

extern "C" int
ompt_initialize(ompt_function_lookup_t lookup, ompt_data_t* tool_data);

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
