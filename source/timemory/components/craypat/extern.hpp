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
 * \file timemory/components/craypat/extern.hpp
 * \brief Include the extern declarations for craypat components
 */

#pragma once

#include "timemory/components/craypat/components.hpp"
#include "timemory/components/extern/common.hpp"
#include "timemory/components/macros.hpp"

TIMEMORY_EXTERN_COMPONENT(craypat_record, false, void)
TIMEMORY_EXTERN_COMPONENT(craypat_region, false, void)
TIMEMORY_EXTERN_COMPONENT(craypat_counters, true, std::vector<unsigned long>)
TIMEMORY_EXTERN_COMPONENT(craypat_heap_stats, false, void)
TIMEMORY_EXTERN_COMPONENT(craypat_flush_buffer, false, unsigned long)
