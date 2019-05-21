//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file timemory.hpp
 * \headerfile timemory.hpp "timemory/timemory.hpp"
 * All-inclusive timemory header
 *
 */

#pragma once

#include "timemory/apply.hpp"
#include "timemory/auto_macros.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/clocks.hpp"
#include "timemory/component_list.hpp"
#include "timemory/component_tuple.hpp"
#include "timemory/components.hpp"
#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/mpi.hpp"
#include "timemory/rusage.hpp"
#include "timemory/serializer.hpp"
#include "timemory/settings.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/singleton.hpp"
#include "timemory/storage.hpp"
#include "timemory/testing.hpp"
#include "timemory/units.hpp"
#include "timemory/utility.hpp"

// CUPTI does not have a dummy API
#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/cupti.hpp"
#endif
