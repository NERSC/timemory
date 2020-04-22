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
 * \file timemory/components/user_bundle/extern.hpp
 * \brief Include the extern declarations for user_bundle components
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/macros.hpp"
//
#include "timemory/components/user_bundle/components.hpp"
#include "timemory/components/user_bundle/types.hpp"
//
#include "timemory/environment/extern.hpp"
#include "timemory/hash/extern.hpp"
#include "timemory/manager/extern.hpp"
#include "timemory/plotting/extern.hpp"
#include "timemory/settings/extern.hpp"
//
#if defined(TIMEMORY_USER_BUNDLE_SOURCE) ||                                              \
    (!defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_USER_BUNDLE_EXTERN))
// source/header-only requirements
#    include "timemory/environment/declaration.hpp"
#    include "timemory/operations/definition.hpp"
#    include "timemory/plotting/definition.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/storage/definition.hpp"
#else
// extern requirements
#    include "timemory/environment/declaration.hpp"
#    include "timemory/operations/definition.hpp"
#    include "timemory/plotting/declaration.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/storage/declaration.hpp"
#endif
//
#include "timemory/components/extern.hpp"
//
#include "timemory/runtime/enumerate.hpp"
//
//======================================================================================//
//
#if defined(TIMEMORY_SOURCE) || defined(TIMEMORY_USER_BUNDLE_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS)
#        define TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(...)                              \
            TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_STORAGE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(...)                                 \
            TIMEMORY_INSTANTIATE_EXTERN_STORAGE(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(...)                                \
            template TIMEMORY_USER_BUNDLE_DLL __VA_ARGS__;
#    endif
//
//--------------------------------------------------------------------------------------//
//
#elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS)
#        define TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(...)                              \
            TIMEMORY_DECLARE_EXTERN_OPERATIONS(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_STORAGE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(...)                                 \
            TIMEMORY_DECLARE_EXTERN_STORAGE(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(...)                                \
            extern template TIMEMORY_USER_BUNDLE_DLL __VA_ARGS__;
#    endif
//
//--------------------------------------------------------------------------------------//
//
#else
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS)
#        define TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_STORAGE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#endif
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                          Base instantiation
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(
    struct base<user_bundle<global_bundle_idx, api::native_tag>, void>)
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(
    struct base<user_bundle<tuple_bundle_idx, api::native_tag>, void>)
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(
    struct base<user_bundle<list_bundle_idx, api::native_tag>, void>)
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(
    struct base<user_bundle<ompt_bundle_idx, api::native_tag>, void>)
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(
    struct base<user_bundle<mpip_bundle_idx, api::native_tag>, void>)
//
//--------------------------------------------------------------------------------------//
//
//                          Derived instantiation
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(
    struct user_bundle<global_bundle_idx, api::native_tag>)
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(
    struct user_bundle<tuple_bundle_idx, api::native_tag>)
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(struct user_bundle<list_bundle_idx, api::native_tag>)
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(struct user_bundle<ompt_bundle_idx, api::native_tag>)
//
TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(struct user_bundle<mpip_bundle_idx, api::native_tag>)
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
//
TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(component::user_global_bundle, false)
//
TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(component::user_tuple_bundle, false)
//
TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(component::user_list_bundle, false)
//
TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(component::user_ompt_bundle, false)
//
TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(component::user_mpip_bundle, false)
//
//======================================================================================//
//
TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(component::user_global_bundle, user_global_bundle)
//
TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(component::user_tuple_bundle, user_tuple_bundle)
//
TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(component::user_list_bundle, user_list_bundle)
//
TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(component::user_ompt_bundle, user_ompt_bundle)
//
TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(component::user_mpip_bundle, user_mpip_bundle)
//
//======================================================================================//
//
