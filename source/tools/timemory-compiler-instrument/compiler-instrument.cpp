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

extern "C"
{
    extern void timemory_profile_func_enter(void* this_fn, void* call_site)
        __attribute__((visibility("hidden")));
    extern void timemory_profile_func_exit(void* this_fn, void* call_site)
        __attribute__((visibility("hidden")));
    void __cyg_profile_func_enter(void* this_fn, void* call_site)
        __attribute__((visibility("default"))) __attribute__((no_instrument_function));
    void __cyg_profile_func_exit(void* this_fn, void* call_site)
        __attribute__((visibility("default"))) __attribute__((no_instrument_function));
}

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    void __cyg_profile_func_enter(void* this_fn, void* call_site)
    {
        timemory_profile_func_enter(this_fn, call_site);
    }
    //
    void __cyg_profile_func_exit(void* this_fn, void* call_site)
    {
        timemory_profile_func_exit(this_fn, call_site);
    }
}  // extern "C"
