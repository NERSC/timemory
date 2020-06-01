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
//

#include "timemory/library.h"
#include "timemory/timemory.hpp"

using namespace tim::component;
//
//--------------------------------------------------------------------------------------//
//
extern "C" void
timemory_register_ex_custom_dynamic_instr()
{
    PRINT_HERE("%s", "");
    // insert monotonic clock component into structure
    // used by timemory-run in --mode=trace
    user_trace_bundle::global_init(nullptr);
    user_trace_bundle::configure<monotonic_clock>();
    // tim::configure<user_trace_bundle>("monotonic_clock");

    // insert monotonic clock component into structure
    // used by timemory-run in --mode=region
    timemory_add_components("monotonic_clock");
}
//
//--------------------------------------------------------------------------------------//
//
extern "C" void
timemory_deregister_ex_custom_dynamic_instr()
{}
//
//--------------------------------------------------------------------------------------//
//
