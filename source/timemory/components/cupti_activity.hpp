//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
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

/** \file cupti.hpp
 * \headerfile cupti_activity.hpp "timemory/cupti_activity.hpp"
 * Provides implementation of CUPTI routines.
 *
 */

#pragma once

#include "timemory/backends/bits/cupti.hpp"
#include "timemory/backends/cuda.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/units.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

#endif

//--------------------------------------------------------------------------------------//
//
//          CUPTI component
//
//--------------------------------------------------------------------------------------//

//#if defined(TIMEMORY_USE_CUPTI)

struct cupti_activity
: public base<cupti_activity, uint64_t, policy::global_init, policy::global_finalize>
{
    // required aliases
    using value_type = uint64_t;
    using this_type  = cupti_activity;
    using base_type =
        base<cupti_activity, value_type, policy::global_init, policy::global_finalize>;

    // component-specific aliases
    using ratio_t           = std::nano;
    using size_type         = std::size_t;
    using string_t          = std::string;
    using receiver_type     = cupti::activity::receiver;
    using kind_vector_type  = std::vector<cupti::activity_kind_t>;
    using get_initializer_t = std::function<kind_vector_type()>;
    using kernel_elapsed_t  = typename cupti::activity::receiver::named_elapsed_t;
    using kernel_names_t    = std::unordered_set<std::string>;

    static std::string label() { return "cupti_activity"; }
    static std::string description() { return "CUpti Activity API"; }

    //----------------------------------------------------------------------------------//

    static get_initializer_t& get_initializer()
    {
        static auto _lambda_instance = []() -> kind_vector_type {
            std::vector<cupti::activity_kind_t> _kinds;
            auto                                lvl = settings::cupti_activity_level();

            /// look up integer codes in <timemory/details/cupti.hpp>
            auto vec = delimit(settings::cupti_activity_kinds());
            for(const auto& itr : vec)
            {
                int iactivity = atoi(itr.c_str());
                if(iactivity > static_cast<int>(CUPTI_ACTIVITY_KIND_INVALID) &&
                   iactivity < static_cast<int>(CUPTI_ACTIVITY_KIND_COUNT))
                {
                    _kinds.push_back(static_cast<cupti::activity_kind_t>(iactivity));
                }
            }

            // if found settings in environment, use those
            if(!_kinds.empty())
            {
                return _kinds;
            }
            else if(lvl == 0)
            {
                // general settings for kernels, runtime, overhead
                _kinds = { { CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                             CUPTI_ACTIVITY_KIND_RUNTIME,
                             CUPTI_ACTIVITY_KIND_OVERHEAD } };
            }
            else if(lvl == 1)
            {
                // general settings for kernels, runtime, memory, overhead
                _kinds = { { CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                             CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
                             CUPTI_ACTIVITY_KIND_RUNTIME,
                             CUPTI_ACTIVITY_KIND_OVERHEAD } };
            }
            else if(lvl == 2)
            {
                // general settings for kernels, runtime, memory, overhead, and device
                _kinds = { { CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                             CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
                             CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DEVICE,
                             CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_OVERHEAD } };
            }
            else if(lvl > 2)
            {
                // general settings for kernels, runtime, memory, overhead, device,
                // stream, CDP kernels
                _kinds = { { CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                             CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
                             CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DEVICE,
                             CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_OVERHEAD,
                             CUPTI_ACTIVITY_KIND_MARKER, CUPTI_ACTIVITY_KIND_STREAM,
                             CUPTI_ACTIVITY_KIND_CDP_KERNEL } };
            }
            return _kinds;
        };
        static get_initializer_t _instance = _lambda_instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static kind_vector_type get_kind_types()
    {
        static kind_vector_type _instance = get_initializer()();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void invoke_global_init(storage_type*)
    {
        static std::atomic<short> _once;
        if(_once++ > 0)
            return;
        cupti::activity::initialize_trace(get_kind_types());
        cupti::init_driver();
    }

    //----------------------------------------------------------------------------------//

    static void invoke_global_finalize(storage_type*)
    {
        cupti::activity::finalize_trace(get_kind_types());
    }

    //----------------------------------------------------------------------------------//

    static value_type record() { return cupti::activity::get_receiver().get(); }

    //----------------------------------------------------------------------------------//

public:
    cupti_activity() = default;

    // make sure it is removed
    ~cupti_activity() { cupti::activity::get_receiver().remove(this); }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        set_started();
        cupti::activity::start_trace(this, depth_change);
        value           = cupti::activity::get_receiver().get();
        m_kernels_index = cupti::activity::get_receiver().get_named_index();
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        cupti::activity::stop_trace(this);
        auto tmp     = cupti::activity::get_receiver().get();
        auto kernels = cupti::activity::get_receiver().get_named(m_kernels_index, true);

        accum += (tmp - value);
        value = std::move(tmp);
        for(const auto& itr : kernels)
            m_kernels_accum[itr.first] += itr.second;
        m_kernels_value = std::move(kernels);

        set_stopped();
    }

    //----------------------------------------------------------------------------------//

    double get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }

    //----------------------------------------------------------------------------------//

    double get() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }

    //----------------------------------------------------------------------------------//

    kernel_elapsed_t get_secondary() const
    {
        return (is_transient) ? m_kernels_accum : m_kernels_value;
    }

private:
    uint64_t         m_kernels_index = 0;
    kernel_elapsed_t m_kernels_value;
    kernel_elapsed_t m_kernels_accum;
};

}  // namespace component
}  // namespace tim
