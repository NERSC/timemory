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

//======================================================================================//

#if defined(TIMEMORY_USE_GOTCHA) && defined(TIMEMORY_USE_MPI)
#    if !defined(TIMEMORY_USE_MPIP)
#        define TIMEMORY_USE_MPIP
#    endif
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
//======================================================================================//

#include "timemory/components/user_bundle/properties.hpp"
#include "timemory/components/user_bundle/traits.hpp"
