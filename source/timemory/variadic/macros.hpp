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
#include "timemory/compat/macros.h"
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

#    if !defined(_WINDOWS)
#        define _TIM_FILENAME_DELIM '/'
#    else
#        define _TIM_FILENAME_DELIM '\\'
#    endif

//--------------------------------------------------------------------------------------//

#    define _TIM_FILESTR                                                                 \
        std::string(__FILE__).substr(                                                    \
            std::string(__FILE__).find_last_of(_TIM_FILENAME_DELIM) + 1)

//--------------------------------------------------------------------------------------//

#    define _TIM_FILELINE ::tim::string::join(":", _TIM_FILESTR, _TIM_LINESTR)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_JOIN(delim, ...) ::tim::string::join(delim, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BLANK_LABEL(...) ::tim::string::join("", __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_LABEL(...) ::tim::string::join("", _TIM_FUNC, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_FULL_LABEL ::tim::string::join("/", _TIM_FUNC, _TIM_FILELINE)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_LABEL(...)                                                          \
        TIMEMORY_JOIN("_", TIMEMORY_FULL_LABEL, TIMEMORY_JOIN("", __VA_ARGS__))

//======================================================================================//
//
//                      MARKER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_MARKER(type, ...)                                             \
        _TIM_STATIC_SRC_LOCATION(blank, __VA_ARGS__);                                    \
        type _TIM_VARIABLE(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_MARKER(type, ...)                                             \
        _TIM_STATIC_SRC_LOCATION(basic, __VA_ARGS__);                                    \
        type _TIM_VARIABLE(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_MARKER(type, ...)                                                   \
        _TIM_STATIC_SRC_LOCATION(full, __VA_ARGS__);                                     \
        type _TIM_VARIABLE(__LINE__)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//======================================================================================//
//
//                      POINTER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_POINTER(type, ...)                                            \
        _TIM_STATIC_SRC_LOCATION(blank, __VA_ARGS__);                                    \
        std::unique_ptr<type> _TIM_VARIABLE(__LINE__) = std::unique_ptr<type>(           \
            (::tim::settings::enabled()) ? new type(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))  \
                                         : nullptr)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_POINTER(type, ...)                                            \
        _TIM_STATIC_SRC_LOCATION(basic, __VA_ARGS__);                                    \
        std::unique_ptr<type> _TIM_VARIABLE(__LINE__) = std::unique_ptr<type>(           \
            (::tim::settings::enabled()) ? new type(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))  \
                                         : nullptr)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_POINTER(type, ...)                                                  \
        _TIM_STATIC_SRC_LOCATION(full, __VA_ARGS__);                                     \
        std::unique_ptr<type> _TIM_VARIABLE(__LINE__) = std::unique_ptr<type>(           \
            (::tim::settings::enabled()) ? new type(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))  \
                                         : nullptr)

//======================================================================================//
//
//                      CALIPER MACROS
//
//======================================================================================//

#    define TIMEMORY_BLANK_CALIPER(id, type, ...)                                        \
        _TIM_STATIC_SRC_LOCATION(blank, __VA_ARGS__);                                    \
        type _TIM_VARIABLE(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_BASIC_CALIPER(id, type, ...)                                        \
        _TIM_STATIC_SRC_LOCATION(basic, __VA_ARGS__);                                    \
        type _TIM_VARIABLE(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER(id, type, ...)                                              \
        _TIM_STATIC_SRC_LOCATION(full, __VA_ARGS__);                                     \
        type _TIM_VARIABLE(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_BLANK_CALIPER(id, type, ...)                                 \
        _TIM_STATIC_SRC_LOCATION(blank, __VA_ARGS__);                                    \
        static type _TIM_VARIABLE(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_BASIC_CALIPER(id, type, ...)                                 \
        _TIM_STATIC_SRC_LOCATION(basic, __VA_ARGS__);                                    \
        static type _TIM_VARIABLE(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_STATIC_CALIPER(id, type, ...)                                       \
        _TIM_STATIC_SRC_LOCATION(full, __VA_ARGS__);                                     \
        static type _TIM_VARIABLE(id)(TIMEMORY_CAPTURE_ARGS(__VA_ARGS__))

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_REFERENCE(id) std::ref(_TIM_VARIABLE(id)).get()

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_APPLY(id, func, ...) _TIM_VARIABLE(id).func(__VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_APPLY(id, type, func, ...)                             \
        _TIM_VARIABLE(id).type_apply<type>(func, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_APPLY0(id, func) _TIM_VARIABLE(id).func()

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_APPLY0(id, type, func)                                 \
        _TIM_VARIABLE(id).type_apply<type>(func)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_LAMBDA(id, lambda, ...)                                     \
        lambda(_TIM_VARIABLE(id), __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_CALIPER_TYPE_LAMBDA(id, type, lambda, ...)                          \
        lambda(_TIM_VARIABLE(id).get<type>(), __VA_ARGS__)

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
