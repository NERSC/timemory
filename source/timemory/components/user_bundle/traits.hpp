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
 * \file timemory/components/user_bundle/traits.hpp
 * \brief Configure the type-traits for the user_bundle components
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/components/user_bundle/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_USE_OMPT)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_ompt_bundle, false_type)
#endif

#if !defined(TIMEMORY_USE_MPIP)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_mpip_bundle, false_type)
#endif

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
struct omp_tools
{
    using type = component_tuple<component::user_ompt_bundle>;
};
//
//--------------------------------------------------------------------------------------//
//
template <size_t _Idx, typename _Type>
struct is_user_bundle<component::user_bundle<_Idx, _Type>> : true_type
{};
//
//--------------------------------------------------------------------------------------//
//
template <size_t _Idx, typename _Type>
struct requires_prefix<component::user_bundle<_Idx, _Type>> : true_type
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trait
}  // namespace tim
