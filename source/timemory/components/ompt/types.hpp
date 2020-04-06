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
