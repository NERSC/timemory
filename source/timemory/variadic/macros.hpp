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
}  // namespace tim

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

#    define TIMEMORY_JOIN(delim, ...) priv::apply::join(delim, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_LABEL(...)                                                    \
        priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_LABEL(...)                                                          \
        priv::apply::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__,                        \
                          _AUTO_STR(__FILE__, _LINE_STRING))

//======================================================================================//
//
//                      MARKER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_MARKER(type, ...)                                             \
        type _AUTO_NAME(__LINE__)(TIMEMORY_JOIN("", __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_MARKER(type, ...)                                             \
        type _AUTO_NAME(__LINE__)(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_MARKER(type, ...)                                                   \
        type _AUTO_NAME(__LINE__)(TIMEMORY_LABEL(__VA_ARGS__))

//======================================================================================//
//
//                      POINTER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_POINTER(type, ...)                                            \
        std::unique_ptr<type> _AUTO_NAME(__LINE__) = std::unique_ptr<type>(              \
            (::tim::settings::enabled()) ? new type(TIMEMORY_JOIN("", __VA_ARGS__))      \
                                         : nullptr)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_POINTER(type, ...)                                            \
        std::unique_ptr<type> _AUTO_NAME(__LINE__) = std::unique_ptr<type>(              \
            (::tim::settings::enabled())                                                 \
                ? new type(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__))        \
                : nullptr)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_POINTER(type, ...)                                                  \
        std::unique_ptr<type> _AUTO_NAME(__LINE__) = std::unique_ptr<type>(              \
            (::tim::settings::enabled()) ? new type(TIMEMORY_LABEL(__VA_ARGS__))         \
                                         : nullptr)

//======================================================================================//
//
//                      CALIPER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_CALIPER(id, type, ...)                                        \
        type _AUTO_NAME(id)(TIMEMORY_JOIN("", __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_CALIPER(id, type, ...)                                        \
        type _AUTO_NAME(id)(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER(id, type, ...)                                              \
        type _AUTO_NAME(id)(TIMEMORY_LABEL(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_BLANK_CALIPER(id, type, ...)                                 \
        static type _AUTO_NAME(id)(TIMEMORY_JOIN("", __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_BASIC_CALIPER(id, type, ...)                                 \
        static type _AUTO_NAME(id)(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_CALIPER(id, type, ...)                                       \
        static type _AUTO_NAME(id)(TIMEMORY_LABEL(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_REFERENCE(id) std::ref(_AUTO_NAME(id)).get()

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_APPLY(id, func, ...) _AUTO_NAME(id).func(__VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_APPLY(id, type, func, ...)                             \
        _AUTO_NAME(id).type_apply<type>(func, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_APPLY0(id, func) _AUTO_NAME(id).func()

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_APPLY0(id, type, func)                                 \
        _AUTO_NAME(id).type_apply<type>(func)

//======================================================================================//
//
//                      HANDLE MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_HANDLE(type, ...) type(TIMEMORY_JOIN("", __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_HANDLE(type, ...)                                             \
        type(TIMEMORY_JOIN("", __TIMEMORY_FUNCTION__, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_HANDLE(type, ...) type(TIMEMORY_LABEL(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BLANK_POINTER_HANDLE(type, ...)                                     \
        (::tim::settings::enabled()) ? new TIMEMORY_BLANK_HANDLE(type, __VA_ARGS__)      \
                                     : nullptr

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_POINTER_HANDLE(type, ...)                                     \
        (::tim::settings::enabled()) ? new TIMEMORY_BASIC_HANDLE(type, __VA_ARGS__)      \
                                     : nullptr

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_POINTER_HANDLE(type, ...)                                           \
        (::tim::settings::enabled()) ? new TIMEMORY_HANDLE(type, __VA_ARGS__) : nullptr

//======================================================================================//
//
//                      DEBUG MACROS
//
//======================================================================================//

#    if defined(DEBUG)

//--------------------------------------------------------------------------------------//

#        define TIMEMORY_DEBUG_BLANK_MARKER(type, ...)                                   \
            TIMEMORY_BASIC_MARKER(type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#        define TIMEMORY_DEBUG_BASIC_MARKER(type, ...)                                   \
            TIMEMORY_BASIC_MARKER(type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#        define TIMEMORY_DEBUG_MARKER(type, ...) TIMEMORY_MARKER(type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    else
#        define TIMEMORY_DEBUG_BLANK_MARKER(type, ...)
#        define TIMEMORY_DEBUG_BASIC_MARKER(type, ...)
#        define TIMEMORY_DEBUG_MARKER(type, ...)
#    endif

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CONFIGURE(type, ...) type::configure(__VA_ARGS__)

//--------------------------------------------------------------------------------------//

// deprecated macros
#    include "timemory/details/macros.hpp"

//--------------------------------------------------------------------------------------//

#endif  // !defined(TIMEMORY_MACROS)

//--------------------------------------------------------------------------------------//
