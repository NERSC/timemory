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
 * \file timemory/components/gotcha/extern.hpp
 * \brief Include the extern declarations for gotcha components
 */

#pragma once

#include "timemory/components/extern/common.hpp"
#include "timemory/components/gotcha/components.hpp"
#include "timemory/components/gotcha/memory_allocations.hpp"
#include "timemory/components/macros.hpp"

TIMEMORY_EXTERN_COMPONENT(malloc_gotcha, true, double)
TIMEMORY_EXTERN_COMPONENT(memory_allocations, false, void)

namespace tim
{
namespace alias
{
// this component is indirectly referenced in malloc_gotcha
using malloc_gotcha_type = component::gotcha<component::malloc_gotcha::data_size,
                                             component_tuple<component::malloc_gotcha>,
                                             type_list<component::malloc_gotcha>>;
}  // namespace alias
}  // namespace tim

TIMEMORY_EXTERN_STORAGE(tim::alias::malloc_gotcha_type)
