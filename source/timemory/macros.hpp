//  MIT License
//  
//  Copyright (c) 2018, The Regents of the University of California, 
// through Lawrence Berkeley National Laboratory (subject to receipt of any 
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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
//

/** \file macros.hpp
 * \headerfile macros.hpp "timemory/macros.hpp"
 * Useful macros for:
 *   - Operating system
 *   - Language
 *   - Compiler
 *   - Windows-specific macros
 */

#ifndef macros_hpp_
#define macros_hpp_

//============================================================================//
//
//      Operating System
//
//============================================================================//

// machine bits
#if defined(__x86_64__)
#   if !defined(_64BIT)
#       define _64BIT
#   endif
#else
#   if !defined(_32BIT)
#       define _32BIT
#   endif
#endif

//----------------------------------------------------------------------------//
// base operating system

#if defined(_WIN32) || defined(_WIN64)
#   if !defined(_WINDOWS)
#       define _WINDOWS
#   endif
//----------------------------------------------------------------------------//

#elif defined(__APPLE__) || defined(__MACH__)
#   if !defined(_MACOS)
#       define _MACOS
#   endif
#   if !defined(_UNIX)
#       define _UNIX
#   endif
//----------------------------------------------------------------------------//

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#   if !defined(_LINUX)
#       define _LINUX
#   endif
#   if !defined(_UNIX)
#       define _UNIX
#   endif
//----------------------------------------------------------------------------//

#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#   if !defined(_UNIX)
#       define _UNIX
#   endif
#endif

//----------------------------------------------------------------------------//

#if defined(_LINUX) || defined(_MACOS)
#   define _C_UNIX  // common unix derivative (i.e. Linux or macOS)
#endif

//============================================================================//
//
//      LANGUAGE
//
//============================================================================//

// Define C++11
#ifndef CXX11
#   if __cplusplus > 199711L   // C++11
#       define CXX11
#   endif
#endif

//----------------------------------------------------------------------------//

// Define C++14
#ifndef CXX14
#   if __cplusplus > 201103L   // C++14
#       define CXX14
#   endif
#endif

//============================================================================//
//
//      Compiler
//
//============================================================================//

//  clang compiler
#if defined(__clang__)
#   define _C_CLANG
#endif

//----------------------------------------------------------------------------//

//  Intel compiler
#if defined(__INTEL_COMPILER)
#   define _C_INTEL
#   if __INTEL_COMPILER < 1500
#       define _C_INTEL_DEPREC // older intel compiler
#   endif
#endif

//----------------------------------------------------------------------------//

// GNU compiler
#if defined(__GNUC__)
#   if (__GNUC__>=4 && __GNUC_MINOR__<9)
#       define _C_GNU_DEPREC // older GNU compiler with questionable C++11 support
#       define _C_GNU
#   elif (__GNUC__>=4 && __GNUC_MINOR__>=9) || __GNUC__>=5
#       define _C_GNU
#   endif
#endif

//============================================================================//
//
//      OS + Compiler
//
//============================================================================//

// macOS using clang
#if defined(_MACOS) && defined(_C_CLANG) && defined(_64BIT)
#   define _MACOS_CLANG
#endif

//----------------------------------------------------------------------------//

//  linux using clang
#if defined(_LINUX) && defined(_C_CLANG)
#   define _LINUX_CLANG
#endif


//============================================================================//
//
//      GLOBAL LINKING
//
//============================================================================//

#ifdef __cplusplus
#   define EXTERN_C       extern "C"
#   define EXTERN_C_BEGIN extern "C" {
#   define EXTERN_C_END   }
#else
#   define EXTERN_C
#   define EXTERN_C_BEGIN
#   define EXTERN_C_END
#endif

//----------------------------------------------------------------------------//

// Define macros for WIN32 for importing/exporting external symbols to DLLs
#if defined(_WINDOWS) && !defined(_TIMEMORY_ARCHIVE)
#   if defined(_TIMEMORY_DLL)
#       define tim_api __declspec(dllexport)
#       define tim_api_static static __declspec(dllexport)
#   else
#       define tim_api __declspec(dllimport)
#       define tim_api_static static __declspec(dllimport)
#   endif
#else
#   define tim_api
#   define tim_api_static static
#endif

//============================================================================//
//
//      WINDOWS WARNINGS
//
//============================================================================//

#if defined(_WINDOWS)
#pragma warning(disable:4786) // ID truncated to '255' char in debug info
#pragma warning(disable:4068) // unknown pragma
#pragma warning(disable:4003) // not enough actual params
#pragma warning(disable:4244) // possible loss of data
#pragma warning(disable:4146) // unsigned
#pragma warning(disable:4129) // unrecognized char escape
#pragma warning(disable:4996) // function may be unsafe
#pragma warning(disable:4267) // possible loss of data
#pragma warning(disable:4700) // uninitialized local variable used
#pragma warning(disable:4217) // locally defined symbol
#pragma warning(disable:4251) // needs to have dll-interface to be used
#endif

//============================================================================//
//
//      THREAD-LOCAL STORAGE
//
//============================================================================//

#if defined(_MACOS_CLANG) || defined(_LINUX_CLANG)
#   define tim_static_thread_local static thread_local
#   define tim_thread_local thread_local
//----------------------------------------------------------------------------//

#elif defined(_C_UNIX) && !defined(_C_INTEL) && defined(_C_GNU)
#   if defined(_C_GNU_DEPREC)
#       define tim_static_thread_local static __thread
#   else
#       define tim_static_thread_local static thread_local
#   endif
#   define tim_thread_local thread_local
//----------------------------------------------------------------------------//

#elif defined(_C_UNIX) && defined(_C_INTEL)
#   if defined(_C_INTEL_DEPREC)
#       define tim_static_thread_local static __thread
#       define tim_thread_local __thread
#   else
#       define tim_static_thread_local static thread_local
#       define tim_thread_local thread_local
#   endif
//----------------------------------------------------------------------------//

#elif defined(_AIX)
#   define tim_static_thread_local static thread_local
#   define tim_thread_local thread_local
//----------------------------------------------------------------------------//

#elif defined(WIN32)
#   define tim_static_thread_local static thread_local
#   define tim_thread_local thread_local
//----------------------------------------------------------------------------//

#else
#   define tim_static_thread_local
#   define tim_thread_local

#endif

//============================================================================//
//
//      FLOATING POINT EXCEPTIONS
//
//============================================================================//
#include <cfenv>
#include <cmath>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma STDC FENV_ACCESS ON
#pragma GCC diagnostic pop

#if !defined(_WINDOWS)
#   define init_priority(N) __attribute__((init_priority(N)))
#   define init_construct(N) __attribute__((constructor(N)))
#   define __c_ctor__ __attribute__((constructor))
#   define __c_dtor__  __attribute__((destructor))
#else
#   define init_priority(N)
#   define init_construct(N)
#   define __c_ctor__
#   define __c_dtor__
#endif

//============================================================================//
//
//      WINDOWS SIGNALS (dummy)
//
//============================================================================//

#if defined(_WINDOWS)
//   dummy definition of SIGHUP
#    ifndef SIGHUP
#        define SIGHUP 1
#    endif
//   dummy definition of SIGINT
#    ifndef SIGINT
#        define SIGINT 2
#    endif
//   dummy definition of SIGQUIT
#    ifndef SIGQUIT
#        define SIGQUIT 3
#    endif
//   dummy definition of SIGILL
#    ifndef SIGILL
#        define SIGILL 4
#    endif
//   dummy definition of SIGTRAP
#    ifndef SIGTRAP
#        define SIGTRAP 5
#    endif
//   dummy definition of SIGABRT
#    ifndef SIGABRT
#        define SIGABRT 6
#    endif
//   dummy definition of SIGEMT
#    ifndef SIGEMT
#        define SIGEMT 7
#    endif
//   dummy definition of SIGFPE
#    ifndef SIGFPE
#        define SIGFPE 8
#    endif
//   dummy definition of SIGKILL
#    ifndef SIGKILL
#        define SIGKILL 9
#    endif
//   dummy definition of SIGBUS
#    ifndef SIGBUS
#        define SIGBUS 10
#    endif
//   dummy definition of SIGSEGV
#    ifndef SIGSEGV
#        define SIGSEGV 11
#    endif
//   dummy definition of SIGSYS
#    ifndef SIGSYS
#        define SIGSYS 12
#    endif
//   dummy definition of SIGPIPE
#    ifndef SIGPIPE
#        define SIGPIPE 13
#    endif
//   dummy definition of SIGALRM
#    ifndef SIGALRM
#        define SIGALRM 14
#    endif
//   dummy definition of SIGTERM
#    ifndef SIGTERM
#        define SIGTERM 15
#    endif
//   dummy definition of SIGURG
#    ifndef SIGURG
#        define SIGURG 16
#    endif
//   dummy definition of SIGSTOP
#    ifndef SIGSTOP
#        define SIGSTOP 17
#    endif
//   dummy definition of SIGTSTP
#    ifndef SIGTSTP
#        define SIGTSTP 18
#    endif
//   dummy definition of SIGCONT
#    ifndef SIGCONT
#        define SIGCONT 19
#    endif
//   dummy definition of SIGCHLD
#    ifndef SIGCHLD
#        define SIGCHLD 20
#    endif
//   dummy definition of SIGTTIN
#    ifndef SIGTTIN
#        define SIGTTIN 21
#    endif
//   dummy definition of SIGTTOU
#    ifndef SIGTTOU
#        define SIGTTOU 22
#    endif
//   dummy definition of SIGIO
#    ifndef SIGIO
#        define SIGIO 23
#    endif
//   dummy definition of SIGXCPU
#    ifndef SIGXCPU
#        define SIGXCPU 24
#    endif
//   dummy definition of SIGXFSZ
#    ifndef SIGXFSZ
#        define SIGXFSZ 25
#    endif
//   dummy definition of SIGVTALRM
#    ifndef SIGVTALRM
#        define SIGVTALRM 26
#    endif
//   dummy definition of SIGPROF
#    ifndef SIGPROF
#        define SIGPROF 27
#    endif
//   dummy definition of SIGWINCH
#    ifndef SIGWINCH
#        define SIGWINCH 28
#    endif
//   dummy definition of SIGINFO
#    ifndef SIGINFO
#        define SIGINFO 29
#    endif
//   dummy definition of SIGUSR1
#    ifndef SIGUSR1
#        define SIGUSR1 30
#    endif
//   dummy definition of SIGUSR2
#    ifndef SIGUSR2
#        define SIGUSR2 31
#    endif
#endif  // defined(_WINDOWS)

#if defined(_WINDOWS)
#   if !defined(NOMINMAX)
#       define NOMINMAX
#   endif
#endif

#endif // macros_hpp_
