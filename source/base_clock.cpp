//  MIT License
//  
//  Copyright (c) 2018, The Regents of the University of California, 
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

#include "timemory/base_clock.hpp"

#if defined(_WINDOWS)

//============================================================================//

int gettimeofday(struct timeval* t, void* /* timezone */)
{
    struct _timeb timebuffer;
#   if defined(_WIN64)
    _ftime64(&timebuffer);
#   elif defined(_WIN32)
    _ftime(&timebuffer);
#   endif
    t->tv_sec=timebuffer.time;
    t->tv_usec=1000*timebuffer.millitm;
    return 0;
}

//============================================================================//

clock_t times (struct tms* __buffer)
{
    __buffer->tms_utime = clock();
    __buffer->tms_stime = 0;
    __buffer->tms_cstime = 0;
    __buffer->tms_cutime = 0;
    return __buffer->tms_utime;
}

//============================================================================//

#else

int32_t base_clock_dummy::asymbol = 0;

#endif
