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

/** \file timemory/components/tau.hpp
 * \headerfile timemory/components/tau.hpp "timemory/components/tau.hpp"
 * Provides marker forwarding to TAU
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/storage.hpp"
#include "timemory/variadic/types.hpp"

#if defined(TIMEMORY_USE_TAU)
#    define TAU_ENABLED
#    define TAU_DOT_H_LESS_HEADERS
#    include "TAU.h"
#endif

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<tau_marker, void>;

#endif

struct tau_marker : public base<tau_marker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = tau_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "tau"; }
    static std::string description() { return "TAU_start and TAU_stop instrumentation"; }
    static value_type  record() {}

    static void global_init(storage_type*)
    {
#if defined(TIMEMORY_USE_TAU)
        Tau_set_node(dmp::rank());
#endif
    }

    static void thread_init(storage_type*)
    {
#if defined(TIMEMORY_USE_TAU)
        TAU_REGISTER_THREAD();
#endif
    }

    tau_marker(const std::string& _prefix = "")
    : m_prefix(_prefix)
    {}

    void start()
    {
#if defined(TIMEMORY_USE_TAU)
        Tau_start(m_prefix.c_str());
#endif
    }
    void stop()
    {
#if defined(TIMEMORY_USE_TAU)
        Tau_stop(m_prefix.c_str());
#endif
    }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }

    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
private:
    std::string m_prefix = "";
};

}  // namespace component
}  // namespace tim
