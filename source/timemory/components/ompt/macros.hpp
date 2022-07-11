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

#include "timemory/api.hpp"

#if defined(TIMEMORY_OMPT_SOURCE)
#    define TIMEMORY_OMPT_INLINE
#elif defined(TIMEMORY_USE_OMPT_EXTERN)
#    define TIMEMORY_OMPT_INLINE
#else
#    define TIMEMORY_OMPT_HEADER_MODE 1
#    define TIMEMORY_OMPT_INLINE inline
#endif

#if !defined(TIMEMORY_OMPT_API_TAG)
#    define TIMEMORY_OMPT_API_TAG TIMEMORY_API
#endif

// for callback declarations
#if !defined(TIMEMORY_OMPT_CBCAST)
#    if defined(TIMEMORY_USE_OMPT)
#        define TIMEMORY_OMPT_CBCAST(NAME) reinterpret_cast<ompt_callback_t>(&NAME)
#    else
#        define TIMEMORY_OMPT_CBCAST(...)
#    endif
#endif
