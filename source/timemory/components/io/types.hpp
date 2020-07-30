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
 * \file timemory/components/io/types.hpp
 * \brief Declare the io component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_COMPONENT(read_bytes)
TIMEMORY_DECLARE_COMPONENT(written_bytes)

namespace tim
{
namespace resource_usage
{
namespace alias
{
template <size_t N>
using farray_t  = std::array<double, N>;
using pair_dd_t = std::pair<double, double>;
}  // namespace alias
}  // namespace resource_usage
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_STATISTICS_TYPE(component::read_bytes, resource_usage::alias::pair_dd_t)
TIMEMORY_STATISTICS_TYPE(component::written_bytes, resource_usage::alias::farray_t<2>)

//--------------------------------------------------------------------------------------//
//
//                              ECHO ENABLED TRAIT
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::written_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::read_bytes, false_type)

//--------------------------------------------------------------------------------------//
//
//                              FILE SAMPLER
//
//--------------------------------------------------------------------------------------//

#if defined(_LINUX) || (defined(_UNIX) && !defined(_MACOS))

TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::written_bytes, true_type)

#endif

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::written_bytes, true_type)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LABEL PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::written_bytes, true_type)

//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::written_bytes, true_type)

//
//      WINDOWS (non-UNIX)
//
#if !defined(_UNIX)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::read_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::written_bytes, false_type)

#endif

//--------------------------------------------------------------------------------------//
//
//                              IS MEMORY CATEGORY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::written_bytes, true_type)

//--------------------------------------------------------------------------------------//
//
//                              USES MEMORY UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::written_bytes, true_type)

//--------------------------------------------------------------------------------------//
//
//                              UNITS SPECIALIZATIONS
//
//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
//--------------------------------------------------------------------------------------//

template <>
struct units<component::read_bytes>
{
    using type         = std::pair<double, double>;
    using display_type = std::pair<std::string, std::string>;
};

//--------------------------------------------------------------------------------------//

template <>
struct units<component::written_bytes>
{
    using type         = std::array<double, 2>;
    using display_type = std::array<std::string, 2>;
};

//--------------------------------------------------------------------------------------//

}  // namespace trait
}  // namespace tim

TIMEMORY_PROPERTY_SPECIALIZATION(read_bytes, READ_BYTES, "read_bytes", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(written_bytes, WRITTEN_BYTES, "written_bytes",
                                 "write_bytes")
