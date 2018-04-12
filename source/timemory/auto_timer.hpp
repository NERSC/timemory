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

#include <string>
#include <cstdint>

#include "timemory/macros.hpp"
#include "timemory/utility.hpp"
#include "timemory/manager.hpp"

namespace tim
{

class tim_api auto_timer
{
public:
    typedef tim::manager::tim_timer_t   tim_timer_t;
    typedef auto_timer                  this_type;
    typedef std::string                 string_t;
    typedef tim::manager::counter_t     counter_t;

public:
    // standard constructor
    auto_timer(const string_t&, const int32_t& lineno,
               const string_t& = "cxx", bool report_at_exit = false);
    // construct from existing timer
    auto_timer(tim_timer_t&, const int32_t& lineno,
               const string_t& = "cxx", bool report_at_exit = false);
    // destructor
    virtual ~auto_timer();

    tim_timer_t* local_timer() const { return m_temp_timer; }
    tim_timer_t* global_timer() const { return m_timer; }

public:
    // static public functions
    static counter_t& ncount();
    static counter_t& nhash();
    static counter_t& pcount();
    static counter_t& phash();
    static bool alloc_next();

private:
    bool            m_report_at_exit;
    uint64_t        m_hash;
    tim_timer_t*    m_timer;
    tim_timer_t*    m_temp_timer;
};

//----------------------------------------------------------------------------//

} // namespace tim

//----------------------------------------------------------------------------//

typedef     tim::auto_timer     auto_timer_t;

#if !defined(TIMEMORY_AUTO_TIMER)

//----------------------------------------------------------------------------//
// helper macros for assembling unique variable name
#define AUTO_TIMER_NAME_COMBINE(X, Y) X##Y
#define AUTO_TIMER_NAME(Y) AUTO_TIMER_NAME_COMBINE(macro_auto_timer, Y)
// helper macro for "__FUNC__@'__FILE__':__LINE__" tagging
#define AUTO_TIMER_STR(A, B) std::string("@'") + \
    std::string( A ).substr(std::string( A ).find_last_of("/")+1) + \
    std::string("':") + B

//----------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_SIGN(str)
 *
 * helper macro for "__FUNC__" + str tagging
 *
 * Usage:
 *
 *      void func()
 *      {
 *          auto_timer_t timer(TIMEMORY_AUTO_SIGN_BASIC("example"), __LINE__)
 *          ...
 *      }
 */
#   define TIMEMORY_BASIC_AUTO_SIGN(str) \
        std::string(std::string(__FUNCTION__) + std::string(str))

//----------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_SIGN(str)
 *
 * helper macro for "__FUNC__" + str + '@__FILE__':__LINE__" tagging
 *
 * Usage:
 *
 *      void func()
 *      {
 *          auto_timer_t timer(TIMEMORY_AUTO_SIGN("example"), __LINE__)
 *          ...
 *      }
 */
#   define TIMEMORY_AUTO_SIGN(str) \
        std::string(std::string(__FUNCTION__) + std::string(str) + \
                    AUTO_TIMER_STR(__FILE__, TIMEMORY_LINE_STRING ))

//----------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_TIMER(str)
 *
 * simple tagging with <function name> + <string> where the string param
 * \a str is optional
 *
 * Signature:
 *      __FUNC__ + str
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_BASIC_AUTO_TIMER();
 *          ...
 *      }
 *
 * Example where str == "(15)":
 *
 *      > [pyc] some_func(15) :  0.363 wall, ... etc.
 */
#define TIMEMORY_BASIC_AUTO_TIMER(str) \
        auto_timer_t AUTO_TIMER_NAME(__LINE__)(std::string(__FUNCTION__) + \
            std::string(str), __LINE__)


//----------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_TIMER(str)
 *
 * standard tagging with <function name> + <string> + "@'<filename>':<line>"
 * where the string param \a str is optional
 *
 * Signature:
 *
 *      __FUNC__ + str + '@__FILE__':__LINE__
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_AUTO_TIMER();
 *          ...
 *      }
 *
 * Example where str == "(15)":
 *
 *      > [pyc] some_func(15)@'nested_test.py':69 :  0.363 wall, ... etc.
 */
#define TIMEMORY_AUTO_TIMER(str) \
    auto_timer_t AUTO_TIMER_NAME(__LINE__)(std::string(__FUNCTION__) + \
            std::string(str) + \
            AUTO_TIMER_STR(__FILE__, TIMEMORY_LINE_STRING ), __LINE__)


//----------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_TIMER_OBJ(str)
 *
 * Similar to \ref TIMEMORY_BASIC_AUTO_TIMER(str) but assignable.
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          auto_timer_t* timer = new TIMEMORY_BASIC_AUTO_TIMER_OBJ();
 *          ...
 *      }
*/
#define TIMEMORY_BASIC_AUTO_TIMER_OBJ(str) \
    auto_timer_t(std::string(__FUNCTION__) + std::string(str), __LINE__)


//----------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_TIMER_OBJ(str)
 *
 * Similar to \ref TIMEMORY_AUTO_TIMER(str) but assignable.
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          auto_timer_t* timer = new TIMEMORY_AUTO_TIMER_OBJ();
 *          ...
 *      }
 *
 */
#define TIMEMORY_AUTO_TIMER_OBJ(str) \
    auto_timer_t(std::string(__FUNCTION__) + std::string(str) + \
                 AUTO_TIMER_STR(__FILE__, TIMEMORY_LINE_STRING ), __LINE__)


//----------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_TIMER_BASIC(str)
 *
 * backwards compatibility for \ref TIMEMORY_BASIC_AUTO_TIMER(str)
 *
 */
#define TIMEMORY_AUTO_TIMER_BASIC(str) TIMEMORY_BASIC_AUTO_TIMER(str)

#endif

//----------------------------------------------------------------------------//

#endif

