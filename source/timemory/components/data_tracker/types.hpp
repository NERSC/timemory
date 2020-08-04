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
 * \file timemory/components/data_tracker/types.hpp
 * \brief Declare the data_tracker component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/data/handler.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_TEMPLATE_COMPONENT(data_tracker, typename InpT,
                                    typename Tag     = api::native_tag,
                                    typename Handler = data::handler<InpT, Tag>,
                                    typename StoreT  = InpT)

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace component
{
using data_tracker_integer  = data_tracker<intmax_t, api::native_tag>;
using data_tracker_unsigned = data_tracker<size_t, api::native_tag>;
using data_tracker_floating = data_tracker<double, api::native_tag>;
}  // namespace component
//
namespace trait
{
template <typename InpT, typename Tag, typename Handler, typename StoreT>
struct component_apis<component::data_tracker<InpT, Tag, Handler, StoreT>>
{
    using type = type_list<project::timemory, category::logger, os::agnostic>;
};
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
//                              BASE HAS ACCUM
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace trait
{
template <typename InpT, typename Tag, typename Handler, typename StoreT>
struct base_has_accum<component::data_tracker<InpT, Tag, Handler, StoreT>> : false_type
{};
//
template <typename InpT, typename Tag, typename Handler, typename StoreT>
struct is_component<component::data_tracker<InpT, Tag, Handler, StoreT>> : true_type
{};
//
template <>
struct python_args<TIMEMORY_RECORD, component::data_tracker_integer>
{
    using type = type_list<intmax_t>;
};
//
template <>
struct python_args<TIMEMORY_RECORD, component::data_tracker_unsigned>
{
    using type = type_list<size_t>;
};
//
template <>
struct python_args<TIMEMORY_RECORD, component::data_tracker_floating>
{
    using type = type_list<double>;
};
//
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(data_tracker_integer, DATA_TRACKER_INTEGER,
                                 "data_tracker_integer", "integer_data_tracker")
TIMEMORY_PROPERTY_SPECIALIZATION(data_tracker_unsigned, DATA_TRACKER_UNSIGNED,
                                 "data_tracker_unsigned", "unsigned_data_tracker")
TIMEMORY_PROPERTY_SPECIALIZATION(data_tracker_floating, DATA_TRACKER_FLOATING,
                                 "data_tracker_floating", "floating_data_tracker")
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_METADATA_SPECIALIZATION(
    data_tracker_integer, "data_integer", "Stores signed integer data w.r.t. call-graph",
    "Useful for tracking +/- values in different call-graph contexts")
TIMEMORY_METADATA_SPECIALIZATION(
    data_tracker_unsigned, "data_unsigned",
    "Stores unsigned integer data w.r.t. call-graph",
    "Useful for tracking iterations, etc. in different call-graph contexts")
TIMEMORY_METADATA_SPECIALIZATION(
    data_tracker_floating, "data_floating",
    "Stores double-precision fp data w.r.t. call-graph",
    "Useful for tracking values in different call-graph contexts")
//
//======================================================================================//
