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
#include "timemory/components/ompt/types.hpp"
#include "timemory/dll.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"

#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
#    define TIMEMORY_USE_USER_BUNDLE_EXTERN
#endif

//======================================================================================//
//
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(user_bundle, size_t Idx, typename Tag)
//
TIMEMORY_BUNDLE_INDEX(global_bundle_idx, 10000)
TIMEMORY_BUNDLE_INDEX(tuple_bundle_idx, 11000)
TIMEMORY_BUNDLE_INDEX(list_bundle_idx, 11100)
// TIMEMORY_BUNDLE_INDEX(ompt_bundle_idx, 11110)
TIMEMORY_BUNDLE_INDEX(mpip_bundle_idx, 11111)
TIMEMORY_BUNDLE_INDEX(ncclp_bundle_idx, 11112)
TIMEMORY_BUNDLE_INDEX(trace_bundle_idx, 20000)
TIMEMORY_BUNDLE_INDEX(profiler_bundle_idx, 22000)
TIMEMORY_BUNDLE_INDEX(kokkosp_bundle_idx, 0)
//
TIMEMORY_COMPONENT_ALIAS(user_global_bundle,
                         user_bundle<global_bundle_idx, project::timemory>)
TIMEMORY_COMPONENT_ALIAS(user_tuple_bundle,
                         user_bundle<tuple_bundle_idx, project::timemory>)
TIMEMORY_COMPONENT_ALIAS(user_list_bundle,
                         user_bundle<list_bundle_idx, project::timemory>)
TIMEMORY_COMPONENT_ALIAS(user_ompt_bundle,
                         user_bundle<ompt_bundle_idx, project::timemory>)
TIMEMORY_COMPONENT_ALIAS(user_mpip_bundle,
                         user_bundle<mpip_bundle_idx, project::timemory>)
TIMEMORY_COMPONENT_ALIAS(user_ncclp_bundle,
                         user_bundle<ncclp_bundle_idx, project::timemory>)
TIMEMORY_COMPONENT_ALIAS(user_trace_bundle,
                         user_bundle<trace_bundle_idx, project::timemory>)
TIMEMORY_COMPONENT_ALIAS(user_profiler_bundle,
                         user_bundle<profiler_bundle_idx, project::timemory>)
TIMEMORY_COMPONENT_ALIAS(user_kokkosp_bundle,
                         user_bundle<kokkosp_bundle_idx, project::kokkosp>)
//
#if !defined(TIMEMORY_USE_OMPT)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_ompt_bundle, false_type)
#endif
//
#if !defined(TIMEMORY_USE_MPI) || !defined(TIMEMORY_USE_GOTCHA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_mpip_bundle, false_type)
#endif
//
#if !defined(TIMEMORY_USE_NCCL) || !defined(TIMEMORY_USE_GOTCHA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_ncclp_bundle, false_type)
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              IS USER BUNDLE
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SET_COMPONENT_API(component::user_global_bundle, project::timemory, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::user_list_bundle, project::timemory, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::user_tuple_bundle, project::timemory, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::user_mpip_bundle, project::timemory,
                           os::supports_linux)
TIMEMORY_SET_COMPONENT_API(component::user_ncclp_bundle, project::timemory,
                           os::supports_linux)
TIMEMORY_SET_COMPONENT_API(component::user_trace_bundle, project::timemory, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::user_profiler_bundle, project::timemory,
                           os::agnostic)
//
namespace tim
{
namespace trait
{
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
namespace concepts
{
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class Tuple, typename... T>
struct has_user_bundle<Tuple<T...>>
{
    using type = typename get_true_types<trait::is_user_bundle, Tuple<T...>>::type;
    static constexpr bool value = (mpl::get_tuple_size<type>::value != 0);
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace concepts
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
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_ncclp_bundle, USER_NCCLP_BUNDLE,
                                 "user_ncclp_bundle", "ncclp", "nccl_tools", "nccl")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_trace_bundle, USER_TRACE_BUNDLE,
                                 "user_trace_bundle", "trace_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_profiler_bundle, USER_PROFILER_BUNDLE,
                                 "user_profiler_bundle", "profiler_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_kokkosp_bundle, USER_KOKKOSP_BUNDLE,
                                 "user_kokkos_bundle", "kokkos_bundle",
                                 "user_kokkosp_bundle", "kokkosp_bundle")
