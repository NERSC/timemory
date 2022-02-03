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

#include "timemory/components/papi/backends.hpp"
#include "timemory/components/papi/macros.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                          Common PAPI configuration
//
//--------------------------------------------------------------------------------------//
//
struct TIMEMORY_VISIBLE papi_common
{
public:
    template <typename Tp>
    using vector_t = std::vector<Tp>;

    template <typename Tp, size_t N>
    using array_t = std::array<Tp, N>;

    using size_type         = size_t;
    using event_list        = vector_t<int>;
    using value_type        = vector_t<long long>;
    using entry_type        = typename value_type::value_type;
    using get_initializer_t = std::function<event_list()>;

    //----------------------------------------------------------------------------------//

    struct state_data
    {
        TIMEMORY_DEFAULT_OBJECT(state_data)

        bool          is_initialized = false;
        bool          is_finalized   = false;
        bool          is_working     = false;
        volatile bool is_running     = false;
    };

    //----------------------------------------------------------------------------------//

    struct common_data
    {
        TIMEMORY_DEFAULT_OBJECT(common_data)

        bool          is_configured = false;
        bool          is_fixed      = false;
        int           event_set     = PAPI_NULL;
        vector_t<int> events        = {};
    };

    //----------------------------------------------------------------------------------//

    static state_data& state();

    template <typename Tp>
    static common_data& data();

    template <typename Tp>
    static auto& event_set();

    template <typename Tp>
    static auto& is_configured();

    template <typename Tp>
    static auto& is_running();

    template <typename Tp>
    static auto& is_fixed();

    template <typename Tp>
    static auto& get_events();

    //----------------------------------------------------------------------------------//

    static void overflow_handler(int evt_set, void* address, long long overflow_vector,
                                 void* context);

    static void add_event(int evt);

    static bool initialize_papi();
    static bool finalize_papi();

    template <typename Tp>
    static get_initializer_t& get_initializer();

    template <typename Tp>
    static void initialize();

    template <typename Tp>
    static void finalize();

public:
    TIMEMORY_DEFAULT_OBJECT(papi_common)

protected:
    event_list events{};

protected:
    static vector_t<int>& private_events()
    {
        static auto _instance = vector_t<int>{};
        return _instance;
    }
};

template <typename Tp>
inline papi_common::common_data&
papi_common::data()
{
    static thread_local common_data _instance{};
    return _instance;
}

template <typename Tp>
inline auto&
papi_common::event_set()
{
    return data<Tp>().event_set;
}

template <typename Tp>
inline auto&
papi_common::is_configured()
{
    return data<Tp>().is_configured;
}

template <typename Tp>
inline auto&
papi_common::is_running()
{
    return state().is_running;
}

template <typename Tp>
inline auto&
papi_common::is_fixed()
{
    return data<Tp>().is_fixed;
}

template <typename Tp>
inline auto&
papi_common::get_events()
{
    auto& _ret = data<Tp>().events;
    if(!is_fixed<Tp>() && _ret.empty())
        _ret = get_initializer<Tp>()();
    return _ret;
}

template <typename Tp>
inline papi_common::get_initializer_t&
papi_common::get_initializer()
{
    static get_initializer_t _instance = []() {
        if(!papi_common::initialize_papi())
        {
            if(!settings::papi_quiet())
                fprintf(stderr,
                        "[papi_common]> PAPI could not be initialized (initialized: "
                        "%s, working: %s)\n",
                        state().is_initialized ? "y" : "n",
                        state().is_working ? "y" : "n");
            return vector_t<int>{};
        }

        auto events_str = settings::papi_events();

        if(settings::verbose() > 1 || settings::debug())
        {
            printf("[papi_common]> TIMEMORY_PAPI_EVENTS: '%s'...\n", events_str.c_str());
        }

        // don't delimit colons!
        vector_t<string_t> events_str_list = delimit(events_str, "\"',; ");
        vector_t<int>      events_list;

        auto& pevents = private_events();
        for(int& pevent : pevents)
        {
            auto fitr = std::find(events_list.begin(), events_list.end(), pevent);
            if(fitr == events_list.end())
                events_list.push_back(pevent);
        }

        for(const auto& itr : events_str_list)
        {
            if(itr.length() == 0)
                continue;

            if(settings::debug())
            {
                printf("[papi_common]> Getting event code from '%s'...\n", itr.c_str());
            }

            int evt_code = papi::get_event_code(itr);
            if(evt_code == PAPI_NOT_INITED)  // defined as zero
            {
                std::stringstream ss;
                ss << "[papi_common] Error creating event with ID: " << itr;
                if(settings::papi_fail_on_error())
                {
                    TIMEMORY_EXCEPTION(ss.str());
                }
                else
                {
                    fprintf(stderr, "%s\n", ss.str().c_str());
                }
            }
            else
            {
                auto fitr = std::find(events_list.begin(), events_list.end(), evt_code);
                if(fitr == events_list.end())
                {
                    if(settings::debug() || settings::verbose() > 1)
                    {
                        printf("[papi_common] Successfully created event '%s' with "
                               "code '%i'...\n",
                               itr.c_str(), evt_code);
                    }
                    events_list.push_back(evt_code);
                }
                else
                {
                    if(settings::debug() || settings::verbose() > 1)
                    {
                        printf("[papi_common] Event '%s' with code '%i' already "
                               "exists...\n",
                               itr.c_str(), evt_code);
                    }
                }
            }
        }

        return events_list;
    };
    return _instance;
}

template <typename Tp>
inline void
papi_common::initialize()
{
    if(!is_configured<Tp>() && initialize_papi())
    {
        auto _quiet   = settings::papi_quiet();
        auto _debug   = (_quiet) ? false : settings::debug();
        auto _verbose = (_quiet) ? -1 : settings::verbose();
        if(is_running<Tp>())
        {
            if(_debug || _verbose > 0)
                PRINT_HERE("papi event set %i is already running", event_set<Tp>());
            return;
        }
        auto& _event_set = event_set<Tp>();
        auto& _events    = get_events<Tp>();
        if(!_events.empty())
        {
            if(_debug || _verbose > 1)
                PRINT_HERE("configuring %i papi events", (int) _events.size());
            papi::create_event_set(&_event_set, settings::papi_multiplexing());
            papi::add_events(_event_set, _events.data(), _events.size());
            if(settings::papi_overflow() > 0)
            {
                for(auto itr : _events)
                {
                    papi::overflow(_event_set, itr, settings::papi_overflow(), 0,
                                   &overflow_handler);
                }
            }
            if(settings::papi_attach())
                papi::attach(_event_set, process::get_target_id());
            papi::start(_event_set);
            is_running<Tp>()    = true;
            is_configured<Tp>() = papi::working();
        }
        if(!_events.empty() && !is_configured<Tp>())
        {
            CONDITIONAL_PRINT_HERE(!_quiet, "Warning! Configuring %i papi events failed",
                                   (int) _events.size());
        }
    }
}

template <typename Tp>
inline void
papi_common::finalize()
{
    if(!initialize_papi())
        return;
    if(!is_running<Tp>())
        return;
    auto& _event_set = event_set<Tp>();
    auto& _events    = get_events<Tp>();
    if(!_events.empty() && _event_set != PAPI_NULL && _event_set >= 0)
    {
        value_type values(_events.size(), 0);
        papi::stop(_event_set, values.data());
        papi::remove_events(_event_set, _events.data(), _events.size());
        papi::destroy_event_set(_event_set);
        _event_set = PAPI_NULL;
        _events.clear();
        is_running<Tp>() = false;
    }
}
}  // namespace component
}  // namespace tim

#if defined(TIMEMORY_PAPI_HEADER_ONLY_MODE) && TIMEMORY_PAPI_HEADER_ONLY_MODE > 0
#    include "timemory/components/papi/papi_common.cpp"
#endif
