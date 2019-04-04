// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file auto_timer.hpp
 * \headerfile auto_timer.hpp "timemory/auto_timer.hpp"
 * Automatic timers
 * Usage with macros (recommended):
 *    \param TIMEMORY_AUTO_TIMER("")
 *    \param TIMEMORY_BASIC_AUTO_TIMER()
 *    \param auto t = TIMEMORY_AUTO_TIMER_OBJ()
 *    \param auto t = TIMEMORY_BASIC_AUTO_TIMER_OBJ()
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/auto_macros.hpp"
#include "timemory/auto_object.hpp"
#include "timemory/macros.hpp"
#include "timemory/string.hpp"
#include "timemory/utility.hpp"

/*
TIM_NAMESPACE_BEGIN

tim_api class auto_timer : public tim::auto_object<auto_timer, tim::timer>
{
public:
    typedef tim::string                              string_t;
    typedef auto_timer                               auto_type;
    typedef tim::timer                               object_type;
    typedef tim::auto_object<auto_type, object_type> auto_object_type;

public:
    // standard constructor
    auto_timer(const string_t&, const int32_t& lineno = 0, const string_t& = "cxx",
               bool report_at_exit = false);

public:
    // public member functions
    object_type&       local_timer() { return local_object(); }
    const object_type& local_timer() const { return local_object(); }
};

//--------------------------------------------------------------------------------------//

inline auto_timer::auto_timer(const string_t& timer_tag, const int32_t& lineno,
                              const string_t& lang_tag, bool report_at_exit)
: auto_object_type(timer_tag, lineno, lang_tag, report_at_exit)
{
}

//--------------------------------------------------------------------------------------//

TIM_NAMESPACE_END

//======================================================================================//
//
//                              macros
//
//======================================================================================//

typedef tim::auto_timer auto_timer_t;

#define TIMEMORY_BLANK_AUTO_TIMER(str) TIMEMORY_BLANK_AUTO_OBJECT(tim::auto_timer, str)

#define TIMEMORY_BASIC_AUTO_TIMER(str) TIMEMORY_BASIC_AUTO_OBJECT(tim::auto_timer, str)

#define TIMEMORY_AUTO_TIMER(str) TIMEMORY_AUTO_OBJECT(tim::auto_timer, str)

#define TIMEMORY_AUTO_TIMER_OBJ(str) TIMEMORY_AUTO_OBJECT_OBJ(tim::auto_timer, str)

#define TIMEMORY_BASIC_AUTO_TIMER_OBJ(str)                                               \
    TIMEMORY_BASIC_AUTO_OBJECT_OBJ(tim::auto_timer, str)

#define TIMEMORY_DEBUG_BASIC_AUTO_TIMER(str)                                             \
    TIMEMORY_DEBUG_BASIC_AUTO_OBJECT(tim::auto_timer, str)

#define TIMEMORY_DEBUG_AUTO_TIMER(str) TIMEMORY_DEBUG_AUTO_OBJECT(tim::auto_timer, str)
*/
//--------------------------------------------------------------------------------------//
