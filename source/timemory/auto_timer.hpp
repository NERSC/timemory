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

#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/string.hpp"
#include "timemory/utility.hpp"

namespace tim
{
tim_api class auto_timer
{
public:
    typedef tim::manager::tim_timer_t tim_timer_t;
    typedef auto_timer                this_type;
    typedef tim::string               string_t;
    typedef tim::manager::counter_t   counter_t;

public:
    // standard constructor
    auto_timer(const string_t&, const int32_t& lineno, const string_t& = "cxx",
               bool report_at_exit = false);
    // destructor
    virtual ~auto_timer();

public:
    // public member functions
    tim_timer_t&       local_timer() { return m_temp_timer; }
    const tim_timer_t& local_timer() const { return m_temp_timer; }

public:
    // public static functions
    static counter_t& ncount();
    static counter_t& nhash();
    static counter_t& pcount();
    static counter_t& phash();
    static bool       alloc_next();

protected:
    inline string_t get_tag(const string_t& timer_tag, const string_t& lang_tag);

private:
    bool        m_enabled;
    bool        m_report_at_exit;
    uint64_t    m_hash;
    tim_timer_t m_temp_timer;
};

//--------------------------------------------------------------------------------------//

inline auto_timer::auto_timer(const string_t& timer_tag, const int32_t& lineno,
                              const string_t& lang_tag, bool report_at_exit)
: m_enabled(auto_timer::alloc_next())
, m_report_at_exit(report_at_exit)
, m_hash((m_enabled) ? (lineno + std::hash<string_t>()(timer_tag)) : 0)
, m_temp_timer(tim_timer_t(
      m_enabled,
      (m_enabled)
          ? &manager::instance()->timer(
                timer_tag, lang_tag,
                (m_enabled) ? ((uint64_t)(auto_timer::pcount() + auto_timer::ncount()++))
                            : ((uint64_t)(0)),
                (m_enabled)
                    ? ((uint64_t)(auto_timer::phash() + (auto_timer::nhash() += m_hash)))
                    : ((uint64_t)(0)))
          : nullptr))
{
}

//--------------------------------------------------------------------------------------//

inline auto_timer::~auto_timer()
{
    if(m_enabled)
    {
        // stop the timer
        m_temp_timer.stop();
        assert(m_temp_timer.summation_timer() != nullptr);
        *m_temp_timer.summation_timer() += m_temp_timer;

        // report timer at exit
        if(m_report_at_exit)
        {
            m_temp_timer.grab_metadata(*(m_temp_timer.summation_timer()));

            // show number of laps in temporary timer
            uint64_t _laps = m_temp_timer.summation_timer()->accum().size();
            m_temp_timer.accum().size() += _laps;

            // threadsafe output w.r.t. other timers
            m_temp_timer.report(std::cout, true, true);
        }

        // decrement hash keys
        auto_timer::ncount()--;
        auto_timer::nhash() -= m_hash;
        manager::instance()->pop_graph();
    }
}

//--------------------------------------------------------------------------------------//

auto_timer::string_t
auto_timer::get_tag(const string_t& timer_tag, const string_t& lang_tag)
{
#if defined(TIMEMORY_USE_MPI)
    std::stringstream ss;
    if(tim::mpi_is_initialized())
        ss << tim::mpi_rank();
    ss << "> [" << lang_tag << "] " << timer_tag;
    return ss.str();
#else
    std::stringstream ss;
    ss << "> [" << lang_tag << "] " << timer_tag;
    return ss.str();
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

typedef tim::auto_timer auto_timer_t;

//======================================================================================//
//
//                      CXX macros
//
//======================================================================================//

#if !defined(TIMEMORY_AUTO_TIMER)

#    if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#        define __TIMEMORY_FUNCTION__ __PRETTY_FUNCTION__
#    else
#        define __TIMEMORY_FUNCTION__ __FUNCTION__
#    endif

//--------------------------------------------------------------------------------------//
// helper macros for assembling unique variable name
#    define AUTO_TIMER_NAME_COMBINE(X, Y) X##Y
#    define AUTO_TIMER_NAME(Y) AUTO_TIMER_NAME_COMBINE(macro_auto_timer, Y)
// helper macro for "__FUNC__@'__FILE__':__LINE__" tagging
#    define AUTO_TIMER_STR(A, B)                                                         \
        std::string("@'") +                                                              \
            std::string(A).substr(std::string(A).find_last_of("/") + 1) +                \
            std::string("':") + B

//--------------------------------------------------------------------------------------//
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
#    define TIMEMORY_BASIC_AUTO_SIGN(str)                                                \
        std::string(std::string(__TIMEMORY_FUNCTION__) + std::string(str))

//--------------------------------------------------------------------------------------//
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
#    define TIMEMORY_AUTO_SIGN(str)                                                      \
        std::string(std::string(__TIMEMORY_FUNCTION__) + std::string(str) +              \
                    AUTO_TIMER_STR(__FILE__, TIMEMORY_LINE_STRING))

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_AUTO_TIMER(str)
 *
 * auto-timer with fully customizable string
 * \a str is optional
 *
 * Signature:
 *      str
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_BLANK_AUTO_TIMER("some-string");
 *          ...
 *      }
 *
 * Example where str == "(15)":
 *
 *      > [pyc] some_func(15) :  0.363 wall, ... etc.
 */
#    define TIMEMORY_BLANK_AUTO_TIMER(str)                                               \
        auto_timer_t AUTO_TIMER_NAME(__LINE__)(std::string(str), __LINE__)

//--------------------------------------------------------------------------------------//
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
#    define TIMEMORY_BASIC_AUTO_TIMER(str)                                               \
        auto_timer_t AUTO_TIMER_NAME(__LINE__)(                                          \
            std::string(__TIMEMORY_FUNCTION__) + std::string(str), __LINE__)

//--------------------------------------------------------------------------------------//
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
 *          TIMEMORY_AUTO_TIMER("");
 *          ...
 *      }
 *
 * Example where str == "(15)":
 *
 *      > [pyc] some_func(15)@'nested_test.py':69 :  0.363 wall, ... etc.
 */
#    define TIMEMORY_AUTO_TIMER(str)                                                     \
        auto_timer_t AUTO_TIMER_NAME(__LINE__)(                                          \
            std::string(__TIMEMORY_FUNCTION__) + std::string(str) +                      \
                AUTO_TIMER_STR(__FILE__, TIMEMORY_LINE_STRING),                          \
            __LINE__)

//--------------------------------------------------------------------------------------//
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
#    define TIMEMORY_BASIC_AUTO_TIMER_OBJ(str)                                           \
        auto_timer_t(std::string(__TIMEMORY_FUNCTION__) + std::string(str), __LINE__)

//--------------------------------------------------------------------------------------//
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
#    define TIMEMORY_AUTO_TIMER_OBJ(str)                                                 \
        auto_timer_t(std::string(__TIMEMORY_FUNCTION__) + std::string(str) +             \
                         AUTO_TIMER_STR(__FILE__, TIMEMORY_LINE_STRING),                 \
                     __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_TIMER_BASIC(str)
 *
 * backwards compatibility for \ref TIMEMORY_BASIC_AUTO_TIMER(str)
 *
 */
#    define TIMEMORY_AUTO_TIMER_BASIC(str) TIMEMORY_BASIC_AUTO_TIMER(str)

#endif

//======================================================================================//
//
//                      PRODUCTION AND DEBUG
//
//======================================================================================//

#if defined(TIMEMORY_DEBUG)
#    define TIMEMORY_DEBUG_BASIC_AUTO_TIMER(str)                                         \
        auto_timer_t AUTO_TIMER_NAME(__LINE__)(                                          \
            std::string(__TIMEMORY_FUNCTION__) + std::string(str), __LINE__)
#    define TIMEMORY_DEBUG_AUTO_TIMER(str)                                               \
        auto_timer_t AUTO_TIMER_NAME(__LINE__)(                                          \
            std::string(__TIMEMORY_FUNCTION__) + std::string(str) +                      \
                AUTO_TIMER_STR(__FILE__, TIMEMORY_LINE_STRING),                          \
            __LINE__)
#else
#    define TIMEMORY_DEBUG_BASIC_AUTO_TIMER(str)                                         \
        {                                                                                \
            ;                                                                            \
        }
#    define TIMEMORY_DEBUG_AUTO_TIMER(str)                                               \
        {                                                                                \
            ;                                                                            \
        }
#endif
//--------------------------------------------------------------------------------------//
