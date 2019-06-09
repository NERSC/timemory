//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

/** \file auto_macros.hpp
 * \headerfile auto_macros.hpp "timemory/auto_macros.hpp"
 * Generic macros that are intended to be building-blocks for other macros, e.g.
 * TIMEMORY_AUTO_TUPLE and TIMEMORY_AUTO_TIMER
 *
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/apply.hpp"
#include "timemory/macros.hpp"
#include "timemory/utility.hpp"

//======================================================================================//
//
//                      CXX variadic macros
//
//======================================================================================//

namespace priv
{
using apply = tim::apply<std::string>;
}

namespace tim
{
// e.g. tim::str::join(...)
using str = tim::apply<std::string>;
}

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_AUTO_TIMER)

#    if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#        define __TIMEMORY_FUNCTION__ __PRETTY_FUNCTION__
#    else
#        define __TIMEMORY_FUNCTION__ __FUNCTION__
#    endif

//--------------------------------------------------------------------------------------//
// helper macros for assembling unique variable name
#    define LINE_STRING priv::apply::join("", __LINE__)
#    define AUTO_NAME_COMBINE(X, Y) X##Y
#    define AUTO_NAME(Y) AUTO_NAME_COMBINE(macro_auto_timer, Y)
#    define AUTO_TYPEDEF(Y) AUTO_NAME_COMBINE(typedef_auto_tuple, Y)
// helper macro for "__FUNCTION__@'__FILE__':__LINE__" tagging
#    if !defined(_WINDOWS)
#        define AUTO_STR(A, B)                                                           \
            priv::apply::join(                                                           \
                "", "@'", std::string(A).substr(std::string(A).find_last_of('/') + 1),   \
                "':", B)
#    else
#        define AUTO_STR(A, B)                                                           \
            priv::apply::join(                                                           \
                "", "@'", std::string(A).substr(std::string(A).find_last_of('\\') + 1),  \
                "':", B)
#    endif

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_SIGN(...)
 *
 * helper macro for "__FUNCTION__" + ... tagging
 *
 * Usage:
 *
 *      void func()
 *      {
 *          auto_timer_t timer(TIMEMORY_BASIC_AUTO_SIGN("example"), __LINE__)
 *          ...
 *      }
 */
#    define TIMEMORY_BASIC_AUTO_SIGN(...)                                                \
        priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_SIGN(...)
 *
 * helper macro for "__FUNCTION__" + ... + '@__FILE__':__LINE__" tagging
 *
 * Usage:
 *
 *      void func()
 *      {
 *          auto_timer_t timer(TIMEMORY_AUTO_SIGN("example"), __LINE__)
 *          ...
 *      }
 */
#    define TIMEMORY_AUTO_SIGN(...)                                                      \
        priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__,                        \
                          AUTO_STR(__FILE__, LINE_STRING))

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_AUTO_OBJECT(type, arg)
 *
 * No tagging is provided. This create the fastest generation of an auto object
 * because it does not variadically combine arguments
 *
 * Signature:
 *      None
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_BLANK_AUTO_OBJECT(type, __FUNCTION__);
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func :  0.363 wall, ... etc.
 */
#    define TIMEMORY_BLANK_AUTO_OBJECT(type, signature)                                  \
        type AUTO_NAME(__LINE__)(signature, __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_OBJECT(type, ...)
 *
 * simple tagging with <function name> + <string> where the string param
 * \a ... is optional
 *
 * Signature:
 *      __FUNCTION__ + ...
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_BASIC_AUTO_OBJECT();
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func(15) :  0.363 wall, ... etc.
 */
#    define TIMEMORY_BASIC_AUTO_OBJECT(type, ...)                                        \
        type AUTO_NAME(__LINE__)(                                                        \
            priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__), __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_OBJECT(type, ...)
 *
 * standard tagging with <function name> + <string> + "@'<filename>':<line>"
 * where the string param \a ... is optional
 *
 * Signature:
 *
 *      __FUNCTION__ + ... + '@__FILE__':__LINE__
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_AUTO_OBJECT();
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func(15)@'nested_test.py':69 :  0.363 wall, ... etc.
 */
#    define TIMEMORY_AUTO_OBJECT(type, ...)                                              \
        type AUTO_NAME(__LINE__)(priv::apply::join("", __TIMEMORY_FUNCTION__,            \
                                                   __VA_ARGS__,                          \
                                                   AUTO_STR(__FILE__, LINE_STRING)),     \
                                 __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_OBJECT_OBJ(type, ...)
 *
 * Similar to \ref TIMEMORY_BASIC_AUTO_OBJECT(...) but assignable.
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          auto_timer_t* timer = new TIMEMORY_BASIC_AUTO_OBJECT_OBJ();
 *          ...
 *      }
 */
#    define TIMEMORY_BASIC_AUTO_OBJECT_OBJ(type, ...)                                    \
        type(priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__)), __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_OBJECT_OBJ(type, ...)
 *
 * Similar to \ref TIMEMORY_AUTO_OBJECT(...) but assignable.
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          auto_timer_t* timer = new TIMEMORY_AUTO_OBJECT_OBJ();
 *          ...
 *      }
 *
 */
#    define TIMEMORY_AUTO_OBJECT_OBJ(type, ...)                                          \
        type(priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__,                   \
                               AUTO_STR(__FILE__, LINE_STRING)),                         \
             __LINE__)

#endif

//======================================================================================//
//
//                      PRODUCTION AND DEBUG
//
//======================================================================================//

#if defined(DEBUG)
#    define TIMEMORY_DEBUG_BASIC_AUTO_OBJECT(type, ...)                                  \
        type AUTO_NAME(__LINE__)(                                                        \
            priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__), __LINE__)
#    define TIMEMORY_DEBUG_AUTO_OBJECT(type, ...)                                        \
        type AUTO_NAME(__LINE__)(priv::apply::join("", __TIMEMORY_FUNCTION__,            \
                                                   __VA_ARGS__,                          \
                                                   AUTO_STR(__FILE__, LINE_STRING)),     \
                                 __LINE__)
#else
#    define TIMEMORY_DEBUG_BASIC_AUTO_OBJECT(type, ...)                                  \
        {                                                                                \
            ;                                                                            \
        }
#    define TIMEMORY_DEBUG_AUTO_OBJECT(type, ...)                                        \
        {                                                                                \
            ;                                                                            \
        }
#endif

//--------------------------------------------------------------------------------------//
