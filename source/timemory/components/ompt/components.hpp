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
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/types.hpp"

#if defined(TIMEMORY_USE_OMPT)
//
//--------------------------------------------------------------------------------------//
//
static ompt_set_callback_t             ompt_set_callback;
static ompt_get_task_info_t            ompt_get_task_info;
static ompt_get_thread_data_t          ompt_get_thread_data;
static ompt_get_parallel_info_t        ompt_get_parallel_info;
static ompt_get_unique_id_t            ompt_get_unique_id;
static ompt_get_num_places_t           ompt_get_num_places;
static ompt_get_place_proc_ids_t       ompt_get_place_proc_ids;
static ompt_get_place_num_t            ompt_get_place_num;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
static ompt_get_proc_id_t              ompt_get_proc_id;
static ompt_enumerate_states_t         ompt_enumerate_states;
static ompt_enumerate_mutex_impls_t    ompt_enumerate_mutex_impls;
//
//--------------------------------------------------------------------------------------//
//
#endif

//======================================================================================//
//
namespace tim
{
namespace policy
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Api, typename Toolset>
struct ompt_handle
{
    using type               = Toolset;
    using api_type           = Api;
    using function_type      = std::function<void()>;
    using user_ompt_bundle_t = component::user_ompt_bundle;

    //  the default initalizer for OpenMP tools when user_ompt_bundle is included
    static function_type& get_initializer()
    {
        static function_type _instance = []() {};
        return _instance;
    }

    //  this functin calls the initializer for the
    static void configure() { get_initializer()(); }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace policy
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
    using toolset_type = typename trait::ompt_handle<Api>::type;
    using policy_type  = policy::ompt_handle<Api, toolset_type>;
    using tracker_type = policy::instance_tracker<ompt_handle<Api>>;

    using tracker_type::m_tot;

    static std::string label() { return "ompt_handle"; }
    static std::string description()
    {
        return std::string("OpenMP toolset ") + demangle<Api>();
    }

    void start()
    {
        tracker_type::start();
        if(m_tot == 0)
            trait::runtime_enabled<ompt_handle<Api>>::set(true);
    }

    void stop()
    {
        tracker_type::stop();
        if(m_tot == 0)
            trait::runtime_enabled<ompt_handle<Api>>::set(false);
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
