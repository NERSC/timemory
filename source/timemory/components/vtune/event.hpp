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

#include "timemory/backends/ittnotify.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
// create VTune events
//
struct vtune_event
: public base<vtune_event, void, policy::global_init, policy::global_finalize>
{
    using value_type = void;
    using this_type  = vtune_event;
    using base_type =
        base<this_type, value_type, policy::global_init, policy::global_finalize>;

    static std::string label() { return "vtune_event"; }
    static std::string description() { return "Create VTune events"; }
    static value_type  record() {}

    static void invoke_global_init(storage_type*) { ittnotify::pause(); }
    static void invoke_global_finalize(storage_type*) { ittnotify::pause(); }

    void start()
    {
        get_index()++;
        set_started();
        if(m_index == 0)
            ittnotify::resume();
        if(!m_created)
        {
            m_event   = ittnotify::create_event(m_prefix);
            m_created = true;
        }
        ittnotify::start_event(m_event);
    }

    void stop()
    {
        auto _index = --get_index();
        ittnotify::end_event(m_event);
        if(_index == 0)
            ittnotify::pause();
        set_stopped();
    }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }

protected:
    bool               m_created = false;
    std::string        m_prefix;
    int64_t            m_index = -1;
    ittnotify::event_t m_event;

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance;
        return _instance;
    }
};
}  // namespace component
}  // namespace tim
