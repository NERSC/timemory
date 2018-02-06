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
//

/** \file auto_timer.hpp
 * Automatic timers
 * Just use \param TIMEMORY_AUTO_TIMER()
 */

#ifndef auto_timer_hpp_
#define auto_timer_hpp_

#include "timemory/namespace.hpp"
#include "timemory/utility.hpp"
#include "timemory/timing_manager.hpp"

#include <string>
#include <cstdint>

namespace NAME_TIM
{

class auto_timer
{
public:
    typedef NAME_TIM::timing_manager::tim_timer_t   tim_timer_t;
    typedef auto_timer                              this_type;

public:
    // Constructor and Destructors
    auto_timer(const std::string&, const int32_t& lineno,
               const std::string& = "cxx", bool report_at_exit = false);
    virtual ~auto_timer();

public:
    // static public functions
    static uint64_t& ncount();
    static uint64_t& nhash();
    static bool alloc_next();

private:
    bool            m_report_at_exit;
    uint64_t        m_hash;
    tim_timer_t*    m_timer;
    tim_timer_t     m_temp_timer;
};

//----------------------------------------------------------------------------//

} // namespace NAME_TIM

//----------------------------------------------------------------------------//

typedef NAME_TIM::auto_timer                     auto_timer_t;

#if !defined(TIMEMORY_AUTO_TIMER)

// stringify some macro
#   define STRINGIZE(X) STRINGIZE2(X)
#   define STRINGIZE2(X) #X
#   define LINE_STRING STRINGIZE(__LINE__)

// helper macros for assembling unique variable name
#   define AUTO_TIMER_NAME_COMBINE(X, Y) X##Y
#   define AUTO_TIMER_NAME(Y) AUTO_TIMER_NAME_COMBINE(macro_auto_timer, Y)

// helper macro for "__FUNC__@'__FILE__':__LINE__" tagging
#   define AUTO_TIMER_STR(A, B) std::string("@'") + \
    std::string( A ).substr(std::string( A ).find_last_of("/")+1) + std::string("':") + B

// simple tagging w/ function name + optional extra string
#   define TIMEMORY_AUTO_TIMER_BASIC(str) \
        auto_timer_t AUTO_TIMER_NAME(__LINE__)(std::string(__FUNCTION__) + \
            std::string(str), __LINE__)

// standard tagging with function name + optional extra string + "@'filename':##"
#   define TIMEMORY_AUTO_TIMER(str) \
    auto_timer_t AUTO_TIMER_NAME(__LINE__)(std::string(__FUNCTION__) + \
            std::string(str) + AUTO_TIMER_STR(__FILE__, LINE_STRING ), __LINE__)

#endif

//----------------------------------------------------------------------------//

#endif

