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

/** \file namespace.hpp
 * Defines NAME_TIM if not defined
 *
 */

#ifndef namespace_hpp_
#define namespace_hpp_

// The namespace can be configured with CMake:
//      -DTIMEMORY_NAMESPACE=<new top-level namespace>
//
// Below is not really needed if TiMemory is imported via TiMemoryConfig.cmake
//
// Top-level namespace (default = tim)

#if !defined(NAME_TIM)
#   define NAME_TIM tim
#endif

// sort out the operating system
#if defined(_WIN32) || defined(_WIN64)
#   if !defined(_WINDOWS)
#       define _WINDOWS
#   endif
#elif defined(__APPLE__) || defined(__MACH__)
#   if !defined(_MACOS)
#       define _MACOS
#   endif
#   if !defined(_UNIX)
#       define _UNIX
#   endif
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#   if !defined(_LINUX)
#       define _LINUX
#   endif
#   if !defined(_UNIX)
#       define _UNIX
#   endif
#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#   if !defined(_UNIX)
#       define _UNIX
#   endif
#endif

#ifdef __cplusplus
#   define EXTERN_C       extern "C"
#   define EXTERN_C_BEGIN extern "C" {
#   define EXTERN_C_END   }
#else
#   define EXTERN_C       /* Nothing */
#   define EXTERN_C_BEGIN /* Nothing */
#   define EXTERN_C_END   /* Nothing */
#endif


// Define C++11
#ifndef CXX11
#   if __cplusplus > 199711L   // C++11
#       define CXX11
#   endif
#endif

// Define C++14
#ifndef CXX14
#   if __cplusplus > 201103L   // C++14
#       define CXX14
#   endif
#endif

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

#endif
