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

#include "timemory/library.h"
#include "timemory/timemory.hpp"
//
#include "timemory/components/ompt.hpp"

#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;

using api_t          = tim::api::native_tag;
using ompt_handle_t  = ompt_handle<api_t>;
using ompt_toolset_t = typename ompt_handle_t::toolset_type;
using ompt_bundle_t  = tim::component_tuple<ompt_handle_t>;
using handle_map_t   = std::unordered_map<uint64_t, ompt_bundle_t>;

static bool
setup_ompt()
{
    tim::trait::runtime_enabled<ompt_toolset_t>::set(false);
    return tim::trait::runtime_enabled<ompt_toolset_t>::get();
}

static handle_map_t          f_handle_map;
static std::atomic<uint64_t> f_handle_count;
static const bool            initial_config = setup_ompt();
static auto                  settings       = tim::settings::shared_instance<api_t>();

extern "C"
{
    void timemory_ompt_library_ctor() {}

    uint64_t init_timemory_ompt_tools()
    {
        // provide environment variable for enabling/disabling
        if(tim::get_env<bool>("ENABLE_TIMEMORY_OMPT", true))
        {
            tim::auto_lock_t lk(tim::type_mutex<ompt_handle_t>());
            auto             idx = ++f_handle_count;
            f_handle_map.insert(
                { idx, ompt_bundle_t("openmp", true, tim::scope::get_default()) });
            f_handle_map[idx].start();
            return idx;
        }
        else
        {
            return 0;
        }
    }

    uint64_t stop_timemory_ompt_tools(uint64_t id)
    {
        if(id == 0)
        {
            for(auto& itr : f_handle_map)
                itr.second.stop();
            f_handle_map.clear();
        }
        else
        {
            auto itr = f_handle_map.find(id);
            if(itr != f_handle_map.end())
            {
                itr->second.stop();
                f_handle_map.erase(itr);
            }
        }
        return f_handle_map.size();
    }

    void register_timemory_ompt()
    {
        DEBUG_PRINT_HERE("%s", "");
        tim::auto_lock_t lk(tim::type_mutex<ompt_handle_t>());
        if(f_handle_map.count(0) == 0)
        {
            f_handle_map.insert(
                { 0, ompt_bundle_t("openmp", true, tim::scope::get_default()) });
            f_handle_map[0].start();
        }
    }

    void deregister_timemory_ompt()
    {
        DEBUG_PRINT_HERE("%s", "");
        stop_timemory_ompt_tools(0);
    }

    // Below are for FORTRAN codes
    void     timemory_ompt_library_ctor_() {}
    uint64_t init_timemory_ompt_tools_() { return init_timemory_ompt_tools(); }
    uint64_t stop_timemory_ompt_tools_(uint64_t id)
    {
        return stop_timemory_ompt_tools(id);
    }
    void register_timemory_ompt_() { register_timemory_ompt(); }
    void deregister_timemory_ompt_() { deregister_timemory_ompt(); }

}  // extern "C"
