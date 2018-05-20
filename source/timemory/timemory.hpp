//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file timemory.hpp
 * \headerfile timemory.hpp "timemory/timemory.hpp"
 * All-inclusive timemory header + extern declarations
 *
 */

#ifndef timemory_hpp_
#define timemory_hpp_

#include "timemory/macros.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/environment.hpp"
#include "timemory/formatters.hpp"
#include "timemory/manager.hpp"
#include "timemory/mpi.hpp"
#include "timemory/rss.hpp"
#include "timemory/serializer.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/timer.hpp"
#include "timemory/units.hpp"
#include "timemory/utility.hpp"
#include "timemory/singleton.hpp"

// functions that do the initialization and destruction
void _timemory_initialization();
void _timemory_finalization();

// for backwards-compatibility
namespace tim { typedef manager timing_manager; }

#endif
