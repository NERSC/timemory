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

#include "timemory/bits/types.hpp"
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
// e.g. tim::string::join(...)
using string = tim::apply<std::string>;
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
#    define _LINE_STRING ::tim::string::join("", __LINE__)
#    define _AUTO_NAME_COMBINE(X, Y) X##Y
#    define _AUTO_NAME(Y) _AUTO_NAME_COMBINE(timemory_variable_, Y)
#    define _AUTO_TYPEDEF(Y) _AUTO_NAME_COMBINE(timemory_variable_type_, Y)
//
// helper macro for "__FUNCTION__@'__FILE__':__LINE__" tagging
//
#    if !defined(_WINDOWS)
#        define _AUTO_STR(A, B)                                                          \
            ::tim::string::join(                                                         \
                "", "@'", std::string(A).substr(std::string(A).find_last_of('/') + 1),   \
                "':", B)
#    else
#        define _AUTO_STR(A, B)                                                          \
            ::tim::string::join(                                                         \
                "", "@'", std::string(A).substr(std::string(A).find_last_of('\\') + 1),  \
                "':", B)
#    endif

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_JOIN(delim, ...) ::tim::string::join(delim, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_LABEL(...)                                                    \
        ::tim::string::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_LABEL(...)                                                          \
        ::tim::string::join("", __TIMEMORY_FUNCTION__, __VA_ARGS__,                      \
                            _AUTO_STR(__FILE__, _LINE_STRING))

//======================================================================================//
//
//                      MARKER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_MARKER(type, ...)                                             \
        TIMEMORY_STATIC_SOURCE_LOCATION(blank, __VA_ARGS__);                             \
        type _AUTO_NAME(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_MARKER(type, ...)                                             \
        TIMEMORY_STATIC_SOURCE_LOCATION(basic, __VA_ARGS__);                             \
        type _AUTO_NAME(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_MARKER(type, ...)                                                   \
        TIMEMORY_STATIC_SOURCE_LOCATION(full, __VA_ARGS__);                              \
        type _AUTO_NAME(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//======================================================================================//
//
//                      POINTER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_POINTER(type, ...)                                            \
        TIMEMORY_STATIC_SOURCE_LOCATION(blank, __VA_ARGS__);                             \
        std::unique_ptr<type> _AUTO_NAME(__LINE__) = std::unique_ptr<type>(              \
            (::tim::settings::enabled()) ? new type(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))  \
                                         : nullptr)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_POINTER(type, ...)                                            \
        TIMEMORY_STATIC_SOURCE_LOCATION(basic, __VA_ARGS__);                             \
        std::unique_ptr<type> _AUTO_NAME(__LINE__) = std::unique_ptr<type>(              \
            (::tim::settings::enabled()) ? new type(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))  \
                                         : nullptr)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_POINTER(type, ...)                                                  \
        TIMEMORY_STATIC_SOURCE_LOCATION(full, __VA_ARGS__);                              \
        std::unique_ptr<type> _AUTO_NAME(__LINE__) = std::unique_ptr<type>(              \
            (::tim::settings::enabled()) ? new type(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))  \
                                         : nullptr)

//======================================================================================//
//
//                      CALIPER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_CALIPER(id, type, ...)                                        \
        TIMEMORY_STATIC_SOURCE_LOCATION(blank, __VA_ARGS__);                             \
        type _AUTO_NAME(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_CALIPER(id, type, ...)                                        \
        TIMEMORY_STATIC_SOURCE_LOCATION(basic, __VA_ARGS__);                             \
        type _AUTO_NAME(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER(id, type, ...)                                              \
        TIMEMORY_STATIC_SOURCE_LOCATION(full, __VA_ARGS__);                              \
        type _AUTO_NAME(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_BLANK_CALIPER(id, type, ...)                                 \
        TIMEMORY_STATIC_SOURCE_LOCATION(blank, __VA_ARGS__);                             \
        static type _AUTO_NAME(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_BASIC_CALIPER(id, type, ...)                                 \
        TIMEMORY_STATIC_SOURCE_LOCATION(basic, __VA_ARGS__);                             \
        static type _AUTO_NAME(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_CALIPER(id, type, ...)                                       \
        TIMEMORY_STATIC_SOURCE_LOCATION(full, __VA_ARGS__);                              \
        static type _AUTO_NAME(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

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

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_LAMBDA(id, lambda, ...) lambda(_AUTO_NAME(id), __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_LAMBDA(id, type, lambda, ...)                          \
        lambda(_AUTO_NAME(id).get<type>(), __VA_ARGS__)

//======================================================================================//
//
//                      HANDLE MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_HANDLE(type, ...)                                             \
        type(TIMEMORY_INLINE_SOURCE_LOCATION(blank, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_HANDLE(type, ...)                                             \
        type(TIMEMORY_INLINE_SOURCE_LOCATION(basic, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_HANDLE(type, ...)                                                   \
        type(TIMEMORY_INLINE_SOURCE_LOCATION(full, __VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BLANK_RAW_POINTER(type, ...)                                        \
        (::tim::settings::enabled())                                                     \
            ? new type(TIMEMORY_INLINE_SOURCE_LOCATION(blank, __VA_ARGS__))              \
            : nullptr

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_RAW_POINTER(type, ...)                                        \
        (::tim::settings::enabled())                                                     \
            ? new type(TIMEMORY_INLINE_SOURCE_LOCATION(basic, __VA_ARGS__))              \
            : nullptr

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_RAW_POINTER(type, ...)                                              \
        (::tim::settings::enabled())                                                     \
            ? new type(TIMEMORY_INLINE_SOURCE_LOCATION(full, __VA_ARGS__))               \
            : nullptr

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
#    include "timemory/utility/bits/macros.hpp"

//--------------------------------------------------------------------------------------//

#endif  // !defined(TIMEMORY_MACROS)

//--------------------------------------------------------------------------------------//
