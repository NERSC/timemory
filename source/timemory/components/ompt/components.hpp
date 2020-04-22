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
 * \file timemory/components/ompt/components.hpp
 * \brief Implementation of the ompt component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/macros.hpp"
//
#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/types.hpp"

//======================================================================================//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Api>
struct ompt_handle
: public base<ompt_handle<Api>, void>
, private policy::instance_tracker<ompt_handle<Api>>
{
    using api_type     = Api;
    using this_type    = ompt_handle<api_type>;
    using value_type   = void;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using toolset_type = typename trait::ompt_handle<api_type>::type;
    using tracker_type = policy::instance_tracker<this_type>;

    static std::string label() { return "ompt_handle"; }
    static std::string description()
    {
        return std::string("OpenMP toolset ") + demangle<api_type>();
    }

    static auto& get_initializer()
    {
        static std::function<void()> _instance = []() {};
        return _instance;
    }

    static void configure()
    {
        static int32_t _once = 0;
        if(_once++ == 0)
            this_type::get_initializer()();
    }

    static void global_init(storage_type*)
    {
        // if handle gets initialized (i.e. used), it indicates we want to disable
        trait::runtime_enabled<toolset_type>::set(false);
        configure();
    }

    static void global_finalize(storage_type*)
    {
        trait::runtime_enabled<toolset_type>::set(false);
    }

    void start()
    {
#if defined(TIMEMORY_USE_OMPT)
        tracker_type::start();
        if(m_tot == 0)
            trait::runtime_enabled<toolset_type>::set(true);
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_OMPT)
        tracker_type::stop();
        if(m_tot == 0)
            trait::runtime_enabled<toolset_type>::set(false);
#endif
    }

private:
    using tracker_type::m_tot;

public:
    void set_prefix(const std::string& _prefix)
    {
        if(_prefix.empty())
            return;
        tim::auto_lock_t lk(get_persistent_data().m_mutex);
        get_persistent_data().m_prefix = _prefix + "/";
    }

    static std::string get_prefix()
    {
        tim::auto_lock_t lk(get_persistent_data().m_mutex);
        return get_persistent_data().m_prefix;
    }

private:
    struct persistent_data
    {
        std::string m_prefix;
        mutex_t     m_mutex;
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
