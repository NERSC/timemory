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
 * \headerfile auto_macros.hpp "timemory/variadic/macros.hpp"
 * Generic macros that are intended to be building-blocks for other macros, e.g.
 * TIMEMORY_TUPLE and TIMEMORY_TIMER
 *
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/mpl/apply.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

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

#if !defined(TIMEMORY_MACROS)

#    define TIMEMORY_MACROS

//======================================================================================//
//
//                      HELPER MACROS
//
//======================================================================================//

//--------------------------------------------------------------------------------------//
// allow encoding the parameters in signature
//
#    if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#        define __TIMEMORY_FUNCTION__ __PRETTY_FUNCTION__
#    else
#        define __TIMEMORY_FUNCTION__ __FUNCTION__
#    endif

//--------------------------------------------------------------------------------------//
// helper macros for assembling unique variable name
//
#    define _LINE_STRING priv::apply::join("", __LINE__)
#    define _AUTO_NAME_COMBINE(X, Y) X##Y
#    define _AUTO_NAME(Y) _AUTO_NAME_COMBINE(timemory_variable_, Y)
#    define _AUTO_TYPEDEF(Y) _AUTO_NAME_COMBINE(timemory_variable_type_, Y)
//
// helper macro for "__FUNCTION__@'__FILE__':__LINE__" tagging
//
#    if !defined(_WINDOWS)
#        define _AUTO_STR(A, B)                                                          \
            priv::apply::join(                                                           \
                "", "@'", std::string(A).substr(std::string(A).find_last_of('/') + 1),   \
                "':", B)
#    else
#        define _AUTO_STR(A, B)                                                          \
            priv::apply::join(                                                           \
                "", "@'", std::string(A).substr(std::string(A).find_last_of('\\') + 1),  \
                "':", B)
#    endif

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_JOIN(delim, ...)
 *
 * helper macro for joining a variadic list into a string
 *
 * Usage:
 *
 *      void run_fibonacci(long n)
 *      {
 *          auto label = TIMEMORY_JOIN("_", "fibonacci(", n, ")");
 *          ...
 *      }
 *
 * Produces:
 *
 *      std::string label = "fibonacci(_43_)";
 *
 * when n = 43
 *
 */
#    define TIMEMORY_JOIN(delim, ...) priv::apply::join(delim, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_LABEL(...)
 *
 * helper macro for "__FUNCTION__" + ... tagging
 *
 * Usage:
 *
 *      void func()
 *      {
 *          auto_timer_t timer(TIMEMORY_BASIC_LABEL("example"), __LINE__)
 *          ...
 *      }
 */
#    define TIMEMORY_BASIC_LABEL(...)                                                    \
        priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_LABEL(...)
 *
 * helper macro for "__FUNCTION__" + ... + '@__FILE__':__LINE__" tagging
 *
 * Usage:
 *
 *      void func()
 *      {
 *          auto_timer_t timer(TIMEMORY_LABEL("example"), __LINE__)
 *          ...
 *      }
 */
#    define TIMEMORY_LABEL(...)                                                          \
        priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__,                        \
                          _AUTO_STR(__FILE__, _LINE_STRING))

//======================================================================================//
//
//                      OBJECT MACROS
//
//======================================================================================//

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_OBJECT(type, arg)
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
 *          TIMEMORY_BLANK_OBJECT(type, __FUNCTION__);
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func :  0.363 wall, ... etc.
 */
#    define TIMEMORY_BLANK_OBJECT(type, ...)                                             \
        type _AUTO_NAME(__LINE__)(TIMEMORY_JOIN("", __VA_ARGS__), __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_OBJECT(type, ...)
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
 *          TIMEMORY_BASIC_OBJECT();
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func(15) :  0.363 wall, ... etc.
 */
#    define TIMEMORY_BASIC_OBJECT(type, ...)                                             \
        type _AUTO_NAME(__LINE__)(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__), \
                                  __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_OBJECT(type, ...)
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
 *          TIMEMORY_OBJECT();
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func(15)@'nested_test.py':69 :  0.363 wall, ... etc.
 */
#    define TIMEMORY_OBJECT(type, ...)                                                   \
        type _AUTO_NAME(__LINE__)(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__,  \
                                                _AUTO_STR(__FILE__, _LINE_STRING)),      \
                                  __LINE__)

//======================================================================================//
//
//                      CALIPER MACROS
//
//======================================================================================//

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_CALIPER(id, type, arg)
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
 *          TIMEMORY_BLANK_CALIPER(1, type, __FUNCTION__);
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func :  0.363 wall, ... etc.
 */
#    define TIMEMORY_BLANK_CALIPER(id, type, ...)                                        \
        type _AUTO_NAME(id)(TIMEMORY_JOIN("", __VA_ARGS__), __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_CALIPER(id, type, ...)
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
 *          TIMEMORY_BASIC_CALIPER();
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func(15) :  0.363 wall, ... etc.
 */
#    define TIMEMORY_BASIC_CALIPER(id, type, ...)                                        \
        type _AUTO_NAME(id)(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__),       \
                            __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_CALIPER(id, type, ...)
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
 *          TIMEMORY_CALIPER();
 *          ...
 *      }
 *
 * Example where ... == "(15)":
 *
 *      > [pyc] some_func(15)@'nested_test.py':69 :  0.363 wall, ... etc.
 */
#    define TIMEMORY_CALIPER(id, type, ...)                                              \
        type _AUTO_NAME(id)(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__,        \
                                          _AUTO_STR(__FILE__, _LINE_STRING)),            \
                            __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_CALIPER_APPLY(id, func, ...)
 *
 * apply a function to a caliper, e.g. start or stop
 *
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_CALIPER(1, (tim::auto_tuple<tim::component::real_clock>), "");
 *          TIMEMORY_CALIPER_APPLY(1, start);
 *
 *          TIMEMORY_CALIPER_APPLY(1, stop);
 *          ...
 *      }
 *
 */
#    define TIMEMORY_CALIPER_APPLY(id, func, ...) _AUTO_NAME(id).func(__VA_ARGS__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_CALIPER_TYPE_APPLY(id, type, func, ...)
 *
 * apply a function to a caliper, e.g. start or stop
 *
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_CALIPER(1, (tim::auto_tuple<tim::component::real_clock>), "");
 *          TIMEMORY_CALIPER_APPLY(1, start);
 *
 *          TIMEMORY_CALIPER_APPLY(1, stop);
 *          ...
 *      }
 *
 */
#    define TIMEMORY_CALIPER_TYPE_APPLY(id, type, func, ...)                             \
        _AUTO_NAME(id).type_apply<type>(func, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_CALIPER_REFERENCE(id, func, ...)
 *
 * apply a function to a caliper, e.g. start or stop
 *
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          TIMEMORY_CALIPER(1, (tim::auto_tuple<tim::component::real_clock>), "");
 *          auto& obj = TIMEMORY_CALIPER_REFERENCE(1);
 *

 *          obj.stop();
 *          ...
 *      }
 *
 */
#    define TIMEMORY_CALIPER_REFERENCE(id) std::ref(_AUTO_NAME(id)).get()

//======================================================================================//
//
//                      INSTANCE MACROS
//
//======================================================================================//

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_INSTANCE(type, ...)
 *
 * Similar to \ref TIMEMORY_BLANK_OBJECT(...) but assignable.
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          auto_timer_t* timer = new TIMEMORY_BASIC_INSTANCE();
 *          ...
 *      }
 */
#    define TIMEMORY_BLANK_INSTANCE(type, ...)                                           \
        type(TIMEMORY_JOIN("", __VA_ARGS__), __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_INSTANCE(type, ...)
 *
 * Similar to \ref TIMEMORY_BASIC_OBJECT(...) but assignable.
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          auto_timer_t* timer = new TIMEMORY_BASIC_INSTANCE();
 *          ...
 *      }
 */
#    define TIMEMORY_BASIC_INSTANCE(type, ...)                                           \
        type(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__), __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_INSTANCE(type, ...)
 *
 * Similar to \ref TIMEMORY_OBJECT(...) but assignable.
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          auto_timer_t* timer = new TIMEMORY_INSTANCE();
 *          ...
 *      }
 *
 */
#    define TIMEMORY_INSTANCE(type, ...)                                                 \
        type(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__,                       \
                           _AUTO_STR(__FILE__, _LINE_STRING)),                           \
             __LINE__)

//======================================================================================//
//
//                      PRODUCTION AND DEBUG MACROS
//
//======================================================================================//

#    if defined(DEBUG)
#        define TIMEMORY_DEBUG_BASIC_OBJECT(type, ...)                                   \
            type _AUTO_NAME(__LINE__)(                                                   \
                TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__), __LINE__)
#        define TIMEMORY_DEBUG_OBJECT(type, ...)                                         \
            type _AUTO_NAME(__LINE__)(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__,           \
                                                    __VA_ARGS__,                         \
                                                    _AUTO_STR(__FILE__, _LINE_STRING)),  \
                                      __LINE__)
#    else
#        define TIMEMORY_DEBUG_BASIC_OBJECT(type, ...)                                   \
            {                                                                            \
                ;                                                                        \
            }
#        define TIMEMORY_DEBUG_OBJECT(type, ...)                                         \
            {                                                                            \
                ;                                                                        \
            }
#    endif

//--------------------------------------------------------------------------------------//

#    if defined(TIMEMORY_USE_CUDA) && defined(TIMEMORY_USE_NVTX)
#        define TIMEMORY_CALIPER_MARK_STREAM_BEGIN(id, stream)                           \
            TIMEMORY_CALIPER_TYPE_APPLY(id, cuda_event,                                  \
                                        (void (cuda_event::*)(tim::cuda::stream_t)) &    \
                                            cuda_event::mark_begin,                      \
                                        stream);                                         \
            TIMEMORY_CALIPER_TYPE_APPLY(id, nvtx_marker,                                 \
                                        (void (nvtx_marker::*)(tim::cuda::stream_t)) &   \
                                            nvtx_marker::mark_begin,                     \
                                        stream)

#        define TIMEMORY_CALIPER_MARK_STREAM_END(id, stream)                             \
            TIMEMORY_CALIPER_TYPE_APPLY(id, cuda_event,                                  \
                                        (void (cuda_event::*)(tim::cuda::stream_t)) &    \
                                            cuda_event::mark_end,                        \
                                        stream);                                         \
            TIMEMORY_CALIPER_TYPE_APPLY(id, nvtx_marker,                                 \
                                        (void (nvtx_marker::*)(tim::cuda::stream_t)) &   \
                                            nvtx_marker::mark_end,                       \
                                        stream)

#    elif defined(TIMEMORY_USE_CUDA)

#        define TIMEMORY_CALIPER_MARK_STREAM_BEGIN(id, stream)                           \
            TIMEMORY_CALIPER_TYPE_APPLY(id, cuda_event,                                  \
                                        (void (cuda_event::*)(tim::cuda::stream_t)) &    \
                                            cuda_event::mark_begin,                      \
                                        stream);

#        define TIMEMORY_CALIPER_MARK_STREAM_END(id, stream)                             \
            TIMEMORY_CALIPER_TYPE_APPLY(id, cuda_event,                                  \
                                        (void (cuda_event::*)(tim::cuda::stream_t)) &    \
                                            cuda_event::mark_end,                        \
                                        stream)

#    elif defined(TIMEMORY_USE_NVTX)
#        define TIMEMORY_CALIPER_MARK_STREAM_BEGIN(id, stream)                           \
            TIMEMORY_CALIPER_TYPE_APPLY(id, nvtx_marker,                                 \
                                        (void (nvtx_marker::*)(tim::cuda::stream_t)) &   \
                                            nvtx_marker::mark_begin,                     \
                                        stream)

#        define TIMEMORY_CALIPER_MARK_STREAM_END(id, stream)                             \
            TIMEMORY_CALIPER_TYPE_APPLY(id, nvtx_marker,                                 \
                                        (void (nvtx_marker::*)(tim::cuda::stream_t)) &   \
                                            nvtx_marker::mark_end,                       \
                                        stream)

#    else
#        define TIMEMORY_CALIPER_MARK_STREAM_BEGIN(id, stream)
#        define TIMEMORY_CALIPER_MARK_STREAM_END(id, stream)
#    endif

//--------------------------------------------------------------------------------------//

#endif  // !defined(TIMEMORY_MACROS)

//--------------------------------------------------------------------------------------//
