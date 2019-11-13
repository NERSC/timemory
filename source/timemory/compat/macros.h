// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#if defined(DISABLE_TIMEMORY)

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
