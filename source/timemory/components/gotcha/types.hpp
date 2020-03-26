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
 * \file timemory/components/gotcha/types.hpp
 * \brief Declare the gotcha component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

//======================================================================================//
//
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(gotcha, size_t Nt, typename Components,
                                    typename Differentiator = void)
//
TIMEMORY_DECLARE_COMPONENT(malloc_gotcha)
//
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(mpip_handle, typename Toolset, typename Tag)
//
//======================================================================================//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::malloc_gotcha, double)
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_GOTCHA)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::malloc_gotcha, false_type)
//
namespace tim
{
namespace trait
{
//
template <size_t N, typename Comp, typename Diff>
struct is_available<component::gotcha<N, Comp, Diff>> : false_type
{};
//
}  // namespace trait
}  // namespace tim
//
#endif  // TIMEMORY_USE_GOTCHA
//
//--------------------------------------------------------------------------------------//
//
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::malloc_gotcha, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              IS MEMORY CATEGORY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::malloc_gotcha, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              USES MEMORY UNITS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::malloc_gotcha, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              IS GOTCHA
//                              START PRIORITY
//                              STOP PRIORITY
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace trait
{
//
template <size_t N, typename Comp, typename Diff>
struct is_gotcha<component::gotcha<N, Comp, Diff>> : true_type
{};
//
//
template <size_t N, typename Comp, typename Diff>
struct start_priority<component::gotcha<N, Comp, Diff>> : priority_constant<256>
{};
//
//
template <size_t N, typename Comp, typename Diff>
struct stop_priority<component::gotcha<N, Comp, Diff>> : priority_constant<-256>
{};
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
template <size_t Nt, typename Components, typename Differentiator>
struct is_gotcha<component::gotcha<Nt, Components, Differentiator>> : true_type
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace concepts
}  // namespace tim
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(malloc_gotcha, MALLOC_GOTCHA, "malloc_gotcha", "")
//
//======================================================================================//
//
