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

/** \file ctimemory.cpp
 * This is the C++ proxy for the C interface. Compilation of this file is not
 * required for C++ codes but is compiled into "libtimemory.*" (timemory-cxx-library)
 * so that the "libctimemory.*" can be linked during the TiMemory build and
 * "libctimemory.*" can be stand-alone linked to C code.
 *
 */

#include "timemory/manager.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_tuple.hpp"
#include "timemory/variadic/component_list.hpp"
#include "timemory/variadic/component_tuple.hpp"

extern "C"
{
#include "timemory/ctimemory.h"
}

//======================================================================================//
//
//                      C++ extern template instantiation
//
//======================================================================================//

#if defined(TIMEMORY_BUILD_EXTERN_TEMPLATES)

//--------------------------------------------------------------------------------------//
// individual
//
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::cuda_event)

//--------------------------------------------------------------------------------------//
// auto_timer_t
//
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock,
                                  tim::component::user_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util, tim::component::cuda_event)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock,
                                  tim::component::user_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util, tim::component::current_rss,
                                  tim::component::peak_rss, tim::component::cuda_event)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util, tim::component::current_rss,
                                  tim::component::peak_rss, tim::component::cuda_event)

//--------------------------------------------------------------------------------------//
// miscellaneous
//

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock, tim::component::cpu_clock,
                                  tim::component::cuda_event)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util, tim::component::cuda_event)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock,
                                  tim::component::user_clock, tim::component::cuda_event)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util, tim::component::peak_rss,
                                  tim::component::current_rss, tim::component::cuda_event)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(
    tim::component::real_clock, tim::component::system_clock,
    tim::component::thread_cpu_clock, tim::component::thread_cpu_util,
    tim::component::process_cpu_clock, tim::component::process_cpu_util,
    tim::component::peak_rss, tim::component::current_rss, tim::component::cuda_event)

#endif  // defined(TIMEMORY_BUILD_EXTERN_TEMPLATES)
