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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "jlcxx/jlcxx.hpp"
#include "timemory/library.h"
#include "timemory/timemory.hpp"

JLCXX_MODULE
define_julia_module(jlcxx::Module& timemory)
{
    timemory.method("init", [](std::string _v) {
        char* _argv = const_cast<char*>(_v.data());
        timemory_init_library(1, &_argv);
    });
    timemory.method("finalize", &timemory_finalize_library);
    timemory.method("push", &timemory_push_region);
    timemory.method("pop", &timemory_pop_region);
    timemory.method("pause", &timemory_pause);
    timemory.method("resume", &timemory_resume);
    timemory.method("set_default", &timemory_set_default);
    timemory.method("add_components", &timemory_add_components);
    timemory.method("remove_components", &timemory_remove_components);
    timemory.method("push_components", &timemory_push_components);
    timemory.method("pop_components", &timemory_pop_components);
    timemory.method("begin", &timemory_get_begin_record);
    timemory.method("end", &timemory_end_record);
}
