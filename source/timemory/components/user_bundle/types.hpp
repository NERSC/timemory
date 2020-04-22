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
 * \file timemory/components/user_bundle/types.hpp
 * \brief Declare the user_bundle component types
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/dll.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"

//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USER_BUNDLE_SOURCE)
#    define TIMEMORY_USER_BUNDLE_DLL tim_dll_export
#elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
#    define TIMEMORY_USER_BUNDLE_DLL tim_dll_import
#else
#    define TIMEMORY_USER_BUNDLE_DLL
#endif

//======================================================================================//
//
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(user_bundle, size_t Idx,
                                    typename Tag = api::native_tag)
//
TIMEMORY_BUNDLE_INDEX(global_bundle_idx, 10000)
//
TIMEMORY_BUNDLE_INDEX(tuple_bundle_idx, 11000)
//
TIMEMORY_BUNDLE_INDEX(list_bundle_idx, 11100)
//
TIMEMORY_BUNDLE_INDEX(ompt_bundle_idx, 11110)
//
TIMEMORY_BUNDLE_INDEX(mpip_bundle_idx, 11111)
//
TIMEMORY_BUNDLE_INDEX(trace_bundle_idx, 20000)
//
TIMEMORY_BUNDLE_INDEX(profiler_bundle_idx, 22000)
//
TIMEMORY_COMPONENT_ALIAS(user_global_bundle,
                         user_bundle<global_bundle_idx, api::native_tag>)
//
TIMEMORY_COMPONENT_ALIAS(user_tuple_bundle,
                         user_bundle<tuple_bundle_idx, api::native_tag>)
//
TIMEMORY_COMPONENT_ALIAS(user_list_bundle, user_bundle<list_bundle_idx, api::native_tag>)
//
TIMEMORY_COMPONENT_ALIAS(user_ompt_bundle, user_bundle<ompt_bundle_idx, api::native_tag>)
//
TIMEMORY_COMPONENT_ALIAS(user_mpip_bundle, user_bundle<mpip_bundle_idx, api::native_tag>)
//
TIMEMORY_COMPONENT_ALIAS(user_trace_bundle,
                         user_bundle<trace_bundle_idx, api::native_tag>)
//
TIMEMORY_COMPONENT_ALIAS(user_profiler_bundle,
                         user_bundle<profiler_bundle_idx, api::native_tag>)

//--------------------------------------------------------------------------------------//
//
//                              IS USER BUNDLE
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//

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
    using type = component_tuple<component::user_ompt_bundle>;
};
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Type>
struct is_user_bundle<component::user_bundle<Idx, Type>> : true_type
{};
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Type>
struct requires_prefix<component::user_bundle<Idx, Type>> : true_type
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trait
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
template <size_t Idx, typename Type>
struct reset<component::user_bundle<Idx, Type>>
{
    using type = component::user_bundle<Idx, Type>;

    TIMEMORY_DELETED_OBJECT(reset)

    template <typename... Args>
    explicit reset(type&, Args&&...)
    {}
};
}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_global_bundle, USER_GLOBAL_BUNDLE,
                                 "user_global_bundle", "global_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_tuple_bundle, USER_TUPLE_BUNDLE,
                                 "user_tuple_bundle", "tuple_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_list_bundle, USER_LIST_BUNDLE, "user_list_bundle",
                                 "list_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_ompt_bundle, USER_OMPT_BUNDLE, "user_ompt_bundle",
                                 "ompt_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_mpip_bundle, USER_MPIP_BUNDLE, "user_mpip_bundle",
                                 "mpip", "mpi_tools", "mpi")
