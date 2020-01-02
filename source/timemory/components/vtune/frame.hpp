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

#include "timemory/backends/ittnotify.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<vtune_frame, void>;

#endif

//--------------------------------------------------------------------------------------//
// create VTune frames
//
struct vtune_frame : public base<vtune_frame, void>
{
    using value_type = void;
    using this_type  = vtune_frame;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "vtune_frame"; }
    static std::string description() { return "Create VTune frames"; }
    static value_type  record() {}

    static void global_init(storage_type*) { ittnotify::pause(); }
    static void global_finalize(storage_type*) { ittnotify::pause(); }

    void start()
    {
        get_index()++;
        set_started();
        if(m_index == 0)
            ittnotify::resume();
        if(m_domain == nullptr)
            m_domain = ittnotify::create_domain(m_prefix);
        ittnotify::start_frame(m_domain);
    }

    void stop()
    {
        auto _index = --get_index();
        if(m_domain)
            ittnotify::end_frame(m_domain);
        if(_index == 0)
            ittnotify::pause();
        set_stopped();
    }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }

protected:
    std::string          m_prefix;
    int64_t              m_index  = -1;
    ittnotify::domain_t* m_domain = nullptr;

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance(0);
        return _instance;
    }
};
}  // namespace component
}  // namespace tim
