//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

/**
 * \file timemory/components/likwid/components.hpp
 * \brief Implementation of the likwid component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/likwid/backends.hpp"
#include "timemory/components/likwid/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//======================================================================================//
//
//  Activate perfmon (CPU)
//
//======================================================================================//
//
struct likwid_marker : public base<likwid_marker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = likwid_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "likwid_marker"; }
    static std::string description() { return "LIKWID perfmon (CPU) marker forwarding"; }
    static value_type  record() {}

    static void global_init(storage_type*) { LIKWID_MARKER_INIT; }
    static void thread_init(storage_type*) { LIKWID_MARKER_THREADINIT; }

    likwid_marker() = default;

    likwid_marker(const std::string& _prefix)
    : prefix(_prefix)
    {
        register_marker();
    }

    void start()
    {
#if defined(TIMEMORY_USE_LIKWID)
        likwid_markerStartRegion(prefix.c_str());
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_LIKWID)
        likwid_markerStopRegion(prefix.c_str());
#endif
    }

    void reset()
    {
#if defined(TIMEMORY_USE_LIKWID)
        likwid_markerResetRegion(prefix.c_str());
#endif
    }

    void register_marker()
    {
#if defined(LIKWID_WITH_NVMON)
        likwid_gpuMarkerRegisterRegion(prefix.c_str());
#endif
    }

    void set_prefix(const std::string& _prefix)
    {
        prefix = _prefix;
        register_marker();
    }

private:
    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
    std::string prefix = "";
};
//
//======================================================================================//
//
//  Activate nvmon (GPU)
//
//======================================================================================//
//
struct likwid_nvmarker : public base<likwid_nvmarker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = likwid_nvmarker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "likwid_nvmarker"; }
    static std::string description() { return "LIKWID nvmon (GPU) marker forwarding"; }
    static value_type  record() {}

    static void global_init(storage_type*) { LIKWID_NVMARKER_INIT; }
    static void thread_init(storage_type*) { LIKWID_NVMARKER_THREADINIT; }

    likwid_nvmarker() = default;

    likwid_nvmarker(const std::string& _prefix)
    : prefix(_prefix)
    {
        register_marker();
    }

    void start()
    {
#if defined(LIKWID_WITH_NVMON)
        likwid_gpuMarkerStartRegion(prefix.c_str());
#endif
    }

    void stop()
    {
#if defined(LIKWID_WITH_NVMON)
        likwid_gpuMarkerStopRegion(prefix.c_str());
#endif
    }

    void reset()
    {
#if defined(LIKWID_WITH_NVMON)
        likwid_gpuMarkerResetRegion(prefix.c_str());
#endif
    }

    void register_marker()
    {
#if defined(LIKWID_WITH_NVMON)
        likwid_gpuMarkerRegisterRegion(prefix.c_str());
#endif
    }

    void set_prefix(const std::string& _prefix)
    {
        prefix = _prefix;
        register_marker();
    }

private:
    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
    std::string prefix = "";
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
