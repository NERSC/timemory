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

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"

#if !defined(TIMEMORY_USE_LIKWID)
#    if !defined(LIKWID_MARKER_INIT)
#        define LIKWID_MARKER_INIT
#        define LIKWID_MARKER_THREADINIT
#        define LIKWID_MARKER_SWITCH
#        define LIKWID_MARKER_REGISTER(...)
#        define LIKWID_MARKER_START(...)
#        define LIKWID_MARKER_STOP(...)
#        define LIKWID_MARKER_CLOSE
#        define LIKWID_MARKER_GET(...)
#        define LIKWID_MARKER_RESET(...)
#    endif

#    if !defined(LIKWID_WITH_NVMON)
#        define LIKWID_NVMARKER_INIT
#        define LIKWID_NVMARKER_THREADINIT
#        define LIKWID_NVMARKER_SWITCH
#        define LIKWID_NVMARKER_REGISTER(...)
#        define LIKWID_NVMARKER_START(...)
#        define LIKWID_NVMARKER_STOP(...)
#        define LIKWID_NVMARKER_CLOSE
#        define LIKWID_NVMARKER_GET(...)
#        define LIKWID_NVMARKER_RESET(...)
#    endif
#else
#    include <likwid-marker.h>
#    include <likwid.h>
#endif

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)
/*
extern template struct base<likwid_perfmon, void, policy::global_init,
                            policy::thread_init>;
extern template struct base<likwid_nvmon, void, policy::global_init,
                            policy::thread_init>;
*/
#endif

//======================================================================================//
//
//  Activate perfmon (CPU)
//
//======================================================================================//

struct likwid_perfmon
: public base<likwid_perfmon, void, policy::global_init, policy::thread_init>
{
    // timemory component api
    using value_type = void;
    using this_type  = likwid_perfmon;
    using base_type =
        base<this_type, value_type, policy::global_init, policy::thread_init>;

    static std::string label() { return "likwid_perfmon"; }
    static std::string description() { return "LIKWID perfmon (CPU) marker forwarding"; }
    static value_type  record() {}

    static void invoke_global_init(storage_type*) { LIKWID_MARKER_INIT; }
    static void invoke_thread_init(storage_type*) { LIKWID_MARKER_THREADINIT; }

    likwid_perfmon() = default;

    likwid_perfmon(const std::string& _prefix)
    : is_registered(false)
    , prefix(_prefix)
    {
        register_marker();
    }

    void start() { LIKWID_MARKER_START(prefix); }
    void stop() { LIKWID_MARKER_STOP(prefix); }

    void set_prefix(const std::string& _prefix)
    {
        prefix = _prefix;
        register_marker();
    }

    void register_marker()
    {
        if(is_registered)
            return;
        if(prefix.length() == 0)
            return;
        is_registered = true;
        LIKWID_MARKER_REGISTER(prefix);
    }

private:
    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
    bool        is_registered = false;
    std::string prefix        = "";
};

//======================================================================================//
//
//  Activate nvmon (GPU)
//
//======================================================================================//

struct likwid_nvmon
: public base<likwid_nvmon, void, policy::global_init, policy::thread_init>
{
    // timemory component api
    using value_type = void;
    using this_type  = likwid_nvmon;
    using base_type =
        base<this_type, value_type, policy::global_init, policy::thread_init>;

    static std::string label() { return "likwid_nvmon"; }
    static std::string description() { return "LIKWID nvmon (GPU) marker forwarding"; }
    static value_type  record() {}

    static void invoke_global_init(storage_type*) { LIKWID_NVMARKER_INIT; }
    static void invoke_thread_init(storage_type*) { LIKWID_NVMARKER_THREADINIT; }

    likwid_nvmon() = default;

    likwid_nvmon(const std::string& _prefix)
    : is_registered(false)
    , prefix(_prefix)
    {
        register_marker();
    }

    void start() { LIKWID_NVMARKER_START(prefix); }
    void stop() { LIKWID_NVMARKER_STOP(prefix); }

    void set_prefix(const std::string& _prefix)
    {
        prefix = _prefix;
        register_marker();
    }

    void register_marker()
    {
        if(is_registered)
            return;
        if(prefix.length() == 0)
            return;
        is_registered = true;
        LIKWID_NVMARKER_REGISTER(prefix);
    }

private:
    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
    bool        is_registered = false;
    std::string prefix        = "";
};

}  // namespace component
}  // namespace tim
