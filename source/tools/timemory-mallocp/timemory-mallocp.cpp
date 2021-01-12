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

#include "timemory/components/gotcha/memory_allocations.hpp"
#include "timemory/library.h"
#include "timemory/timemory.hpp"

#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;

using malloc_toolset_t     = tim::component_tuple<memory_allocations>;
using malloc_region_t      = tim::lightweight_tuple<malloc_gotcha>;
using malloc_region_pair_t = std::pair<const char*, malloc_region_t>;
using malloc_region_vec_t  = std::vector<malloc_region_pair_t>;
//
namespace
{
uint64_t                                global_cnt = 0;
uint64_t                                global_id  = std::numeric_limits<uint64_t>::max();
static thread_local malloc_region_vec_t regions    = {};
static thread_local int64_t             region_idx = 0;
}  // namespace
//
//--------------------------------------------------------------------------------------//
//
namespace
{
auto&
get_toolset()
{
    static auto _instance = std::shared_ptr<malloc_toolset_t>{};
    return _instance;
}
}  // namespace
//
//--------------------------------------------------------------------------------------//
//
extern "C"
{
    void timemory_mallocp_library_ctor() {}

    uint64_t timemory_start_mallocp()
    {
        // provide environment variable for enabling/disabling
        if(tim::get_env<bool>("TIMEMORY_ENABLE_MALLOCP", true))
        {
            auto& _handle = get_toolset();
            if(!_handle)
            {
                _handle       = std::make_shared<malloc_toolset_t>("timemory-mallocp");
                auto _cleanup = []() {
                    auto& _handle = get_toolset();
                    if(_handle)
                    {
                        _handle->stop();
                        _handle.reset();
                    }
                };

                tim::manager::instance()->add_cleanup("timemory-mallocp", _cleanup);
                global_id = global_cnt;
            }
            _handle->start();
            return global_cnt++;
        }
        else
        {
            return 0;
        }
    }

    uint64_t timemory_stop_mallocp(uint64_t id)
    {
        if(id == global_id)
            global_cnt = 1;
        if(global_cnt == 1)
        {
            tim::manager::instance()->cleanup("timemory-mallocp");
        }
        return --global_cnt;
    }

    void timemory_register_mallocp() { global_id = timemory_start_mallocp(); }
    void timemory_deregister_mallocp()
    {
        timemory_stop_mallocp(global_id);
        global_id = std::numeric_limits<uint64_t>::max();
    }

    uint64_t timemory_reserve_mallocp(uint64_t _sz)
    {
        auto& _val = memory_allocations::gotcha_type::get_suppression();
        gotcha_suppression::auto_toggle _lk{ _val };
        if(_sz >= regions.size())
        {
            regions.resize(_sz, malloc_region_pair_t{ nullptr, malloc_region_t{} });
        }
        return regions.size();
    }

    void timemory_push_mallocp(const char* _key)
    {
        if(static_cast<int64_t>(regions.size()) <= region_idx)
            timemory_reserve_mallocp(regions.size() + 10);

        auto& _suppress = memory_allocations::gotcha_type::get_suppression();
        gotcha_suppression::auto_toggle _lk{ _suppress };

        regions.at(region_idx).first = _key;
        malloc_region_t& _obj        = regions.at(region_idx).second;
        ++region_idx;
        _obj.rekey(_key);
        _obj.push();
        //_obj.start();
    }

    void timemory_pop_mallocp(const char*)
    {
        if(region_idx == 0)
            return;

        --region_idx;
        if(region_idx >= static_cast<int64_t>(regions.size()))
            return;

        malloc_region_t& _obj = regions.at(region_idx).second;

        auto& _suppress = memory_allocations::gotcha_type::get_suppression();
        gotcha_suppression::auto_toggle _lk{ _suppress };
        //_obj.stop();
        _obj.pop();
    }

    // Below are for FORTRAN codes
    void     timemory_mallocp_library_ctor_() {}
    uint64_t timemory_start_mallocp_() { return timemory_start_mallocp(); }
    uint64_t timemory_stop_mallocp_(uint64_t id) { return timemory_stop_mallocp(id); }
    void     timemory_register_mallocp_() { timemory_register_mallocp(); }
    void     timemory_deregister_mallocp_() { timemory_deregister_mallocp(); }
    void     timemory_reserve_mallocp_(uint64_t _sz) { timemory_reserve_mallocp(_sz); }
    void     timemory_push_mallocp_(const char* _key) { timemory_push_mallocp(_key); }
    void     timemory_pop_mallocp_(const char* _key) { timemory_pop_mallocp(_key); }

}  // extern "C"
