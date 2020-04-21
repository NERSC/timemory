// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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

#pragma once

#define _TIM_STRINGIZE(X) _TIM_STRINGIZE2(X)
#define _TIM_STRINGIZE2(X) #X
#define _TIM_VAR_NAME_COMBINE(X, Y) X##Y
#define _TIM_VARIABLE(Y) _TIM_VAR_NAME_COMBINE(timemory_variable_, Y)
#define _TIM_TYPEDEF(Y) _TIM_VAR_NAME_COMBINE(timemory_typedef_, Y)

#define _TIM_LINESTR _TIM_STRINGIZE(__LINE__)

#if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#    define _TIM_FUNC __PRETTY_FUNCTION__
#else
#    define _TIM_FUNC __FUNCTION__
#endif

#if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED)

#    define TIMEMORY_SPRINTF(...)

#else
#    if defined(__cplusplus)
#        define TIMEMORY_SPRINTF(VAR, LEN, FMT, ...)                                     \
            std::unique_ptr<char> VAR_PTR = std::unique_ptr<char>(new char[LEN]);        \
            char*                 VAR     = VAR_PTR.get();                               \
            sprintf(VAR, FMT, __VA_ARGS__);
#    else
#        define TIMEMORY_SPRINTF(VAR, LEN, FMT, ...)                                     \
            char VAR[LEN];                                                               \
            sprintf(VAR, FMT, __VA_ARGS__);
#    endif

#endif

//======================================================================================//
//
//      Operating System
//
//======================================================================================//

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS
#    endif
#elif defined(__APPLE__) || defined(__MACH__)
#    if !defined(_MACOS)
#        define _MACOS
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(_LINUX)
#        define _LINUX
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif
#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#    if !defined(_UNIX)
#        define _UNIX
#    endif
#endif

//======================================================================================//
//
//      Windows DLL settings
//
//======================================================================================//

#if !defined(tim_cdll)
#    if defined(_WINDOWS)
#        if defined(TIMEMORY_CDLL_EXPORT)
#            define tim_cdll __declspec(dllexport)
#        elif defined(TIMEMORY_CDLL_IMPORT)
#            define tim_cdll __declspec(dllimport)
#        else
#            define tim_cdll
#        endif
#    else
#        define tim_cdll
#    endif
#endif

#if !defined(tim_dll)
#    if defined(_WINDOWS)
#        if defined(TIMEMORY_DLL_EXPORT)
#            define tim_dll __declspec(dllexport)
#        elif defined(TIMEMORY_DLL_IMPORT)
#            define tim_dll __declspec(dllimport)
#        else
#            define tim_dll
#        endif
#    else
#        define tim_dll
#    endif
#endif

//======================================================================================//
//
//      Visibility
//
//======================================================================================//

#if defined(TIMEMORY_USE_VISIBILITY)
#    define TIMEMORY_VISIBILITY(mode) __attribute__((visibility(mode)))
#else
#    define TIMEMORY_VISIBILITY(mode)
#endif

//======================================================================================//
//
//      General attribute
//
//======================================================================================//

#if !defined(declare_attribute)
#    if defined(__GNUC__) || defined(__clang__)
#        define declare_attribute(attr) __attribute__((attr))
#    elif defined(_WIN32)
#        define declare_attribute(attr) __declspec(attr)
#    endif
#endif

//======================================================================================//
//
//      Symbol override
//
//======================================================================================//

#if !defined(TIMEMORY_WEAK_PREFIX)
#    if !defined(_WINDOWS)
#        if defined(__clang__) && defined(__APPLE__)
#            define TIMEMORY_WEAK_PREFIX
#        else
#            define TIMEMORY_WEAK_PREFIX __attribute__((weak))
#        endif
#    else
#        define TIMEMORY_WEAK_PREFIX
#    endif
#endif

#if !defined(TIMEMORY_WEAK_POSTFIX)
#    if !defined(_WINDOWS)
#        if defined(__clang__) && defined(__APPLE__)
#            define TIMEMORY_WEAK_POSTFIX __attribute__((weak_import))
#        else
#            define TIMEMORY_WEAK_POSTFIX
#        endif
#    else
#        define TIMEMORY_WEAK_POSTFIX
#    endif
#endif

//======================================================================================//
//
//      Library Constructor/Destructor
//
//======================================================================================//

#if !defined(__library_ctor__)
#    if !defined(_WINDOWS)
#        define __library_ctor__ __attribute__((constructor))
#    else
#        define __library_ctor__
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(__library_dtor__)
#    if !defined(_WINDOWS)
#        define __library_dtor__ __attribute__((destructor))
#    else
#        define __library_dtor__
#    endif
#endif
