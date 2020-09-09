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

//======================================================================================//
//
//      WINDOWS
//
//======================================================================================//

#if defined(_WIN32) || defined(_WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS
#    endif
#endif

//======================================================================================//
//
//      GLOBAL LINKING
//
//======================================================================================//

// Define macros for WIN32 for importing/exporting external symbols to DLLs
#if !defined(tim_dll)
#    define tim_dll
/*
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
*/
#endif

#if defined(_WINDOWS) && (defined(TIMEMORY_DLL_EXPORT) || defined(TIMEMORY_DLL_IMPORT))
#    if !defined(tim_dll_export)
//#        define tim_dll_export __declspec(dllexport)
#        define tim_dll_export
#    endif
#    if !defined(tim_dll_import)
//#        define tim_dll_import __declspec(dllimport)
#        define tim_dll_import
#    endif
#else
#    if !defined(tim_dll_export)
#        define tim_dll_export
#    endif
#    if !defined(tim_dll_import)
#        define tim_dll_import
#    endif
#endif

//======================================================================================//
//
//      WINDOWS WARNINGS
//
//======================================================================================//

#if defined(_WINDOWS)

#    pragma warning(disable : 4003)   // not enough actual params
#    pragma warning(disable : 4068)   // unknown pragma
#    pragma warning(disable : 4129)   // unrecognized char escape
#    pragma warning(disable : 4146)   // unsigned
#    pragma warning(disable : 4217)   // locally defined symbol
#    pragma warning(disable : 4244)   // possible loss of data
#    pragma warning(disable : 4251)   // needs to have dll-interface to be used
#    pragma warning(disable : 4267)   // possible loss of data
#    pragma warning(disable : 4305)   // truncation from 'double' to 'float'
#    pragma warning(disable : 4522)   // multiple assignment operators specified
#    pragma warning(disable : 4661)   // no suitable definition for template inst
#    pragma warning(disable : 4700)   // uninitialized local variable used
#    pragma warning(disable : 4786)   // ID truncated to '255' char in debug info
#    pragma warning(disable : 4996)   // function may be unsafe
#    pragma warning(disable : 5030)   // attribute is not recognized
#    pragma warning(disable : 26495)  // Always initialize member variable (cereal issue)

#endif
