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
 * \file timemory/components/tau_marker/components.hpp
 * \brief Implementation of the tau_marker component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/tau_marker/backends.hpp"
#include "timemory/components/tau_marker/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
struct tau_marker : public base<tau_marker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = tau_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "tau"; }
    static std::string description()
    {
        return "Forwards markers to TAU instrumentation (via Tau_start and Tau_stop)";
    }

#if defined(TIMEMORY_USE_TAU)
    static void global_init() { Tau_set_node(dmp::rank()); }
    static void thread_init() { TAU_REGISTER_THREAD(); }
    static void start(const char* _prefix) { Tau_start(_prefix); }

    static void stop(const char* _prefix)
    {
        Tau_stop(_prefix);
        consume_parameters(_prefix);
    }

    TIMEMORY_DEFAULT_OBJECT(tau_marker)

    tau_marker(const char* _prefix)
    : m_prefix(_prefix)
    {}

    void start() { Tau_start(m_prefix); }
    void stop() { Tau_stop(m_prefix); }
    void set_prefix(const char* _prefix) { m_prefix = _prefix; }

private:
    const char m_prefix = nullptr;
#endif
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
