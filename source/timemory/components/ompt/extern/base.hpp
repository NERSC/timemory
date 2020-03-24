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

/**
 * \file timemory/components/ompt/extern/base.hpp
 * \brief Declare the extern base instances for ompt components
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/ompt/traits.hpp"
#include "timemory/components/ompt/types.hpp"
//
#include "timemory/storage/definition.hpp"

#if defined(TIMEMORY_USE_OMPT)
#    include "timemory/components/ompt/ompt.hpp"
#endif

//
//--------------------------------------------------------------------------------------//
//
#if(!defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_OMPT_EXTERN)) ||              \
    defined(TIMEMORY_OMPT_SOURCE)
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
TIMEMORY_OMPT_LINKAGE(int)
ompt_initialize(ompt_function_lookup_t lookup, ompt_data_t* tool_data)
{
    using namespace tim::component;
    using api_type       = TIMEMORY_OMPT_API_TAG;
    using component_type = typename tim::trait::omp_tools<api_type>::type;
    using policy_type    = tim::policy::omp_tools<api_type, component_type>;
    using connector_type = openmp::callback_connector<component_type, api_type>;

    if(!tim::trait::is_available<omp_tools<api_type>>::value)
        return 1;

    tim::consume_parameters(tool_data);

    auto register_callback = [](ompt_callbacks_t name, ompt_callback_t cb) {
        int ret = ompt_set_callback(name, cb);

        if(tim::settings::verbose() < 1 && !tim::settings::debug())
            return ret;

        switch(ret)
        {
            case ompt_set_never:
                printf("[timemory]> WARNING: OMPT Callback for event %d could not be "
                       "registered\n",
                       name);
                break;
            case ompt_set_sometimes:
                printf("[timemory]> OMPT Callback for event %d registered with return "
                       "value %s\n",
                       name, "ompt_set_sometimes");
                break;
            case ompt_set_sometimes_paired:
                printf("[timemory]> OMPT Callback for event %d registered with return "
                       "value %s\n",
                       name, "ompt_set_sometimes_paired");
                break;
            case ompt_set_always:
                printf("[timemory]> OMPT Callback for event %d registered with return "
                       "value %s\n",
                       name, "ompt_set_always");
                break;
        }

        return ret;
    };

    auto timemory_ompt_register_callback = [&](ompt_callbacks_t name,
                                               ompt_callback_t  cb) {
        int ret = register_callback(name, cb);
        tim::consume_parameters(ret);
    };

    ompt_set_callback      = (ompt_set_callback_t) lookup("ompt_set_callback");
    ompt_get_task_info     = (ompt_get_task_info_t) lookup("ompt_get_task_info");
    ompt_get_unique_id     = (ompt_get_unique_id_t) lookup("ompt_get_unique_id");
    ompt_get_thread_data   = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
    ompt_get_parallel_info = (ompt_get_parallel_info_t) lookup("ompt_get_parallel_info");

    ompt_get_num_places = (ompt_get_num_places_t) lookup("ompt_get_num_places");
    ompt_get_place_proc_ids =
        (ompt_get_place_proc_ids_t) lookup("ompt_get_place_proc_ids");
    ompt_get_place_num = (ompt_get_place_num_t) lookup("ompt_get_place_num");
    ompt_get_partition_place_nums =
        (ompt_get_partition_place_nums_t) lookup("ompt_get_partition_place_nums");
    ompt_get_proc_id      = (ompt_get_proc_id_t) lookup("ompt_get_proc_id");
    ompt_enumerate_states = (ompt_enumerate_states_t) lookup("ompt_enumerate_states");
    ompt_enumerate_mutex_impls =
        (ompt_enumerate_mutex_impls_t) lookup("ompt_enumerate_mutex_impls");

    policy_type::configure();

    using parallel_begin_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, const omp_frame_t*, ompt_data_t*, uint32_t,
                             const void*>;

    using parallel_end_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*, ompt_data_t*, const void*>;

    using task_create_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, const omp_frame_t*, ompt_data_t*, int, int,
                             const void*>;

    using task_schedule_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, ompt_task_status_t, ompt_data_t*>;

    using master_cb_t =
        openmp::ompt_wrapper<component_type, connector_type,
                             openmp::mode::measure_callback, ompt_scope_endpoint_t,
                             ompt_data_t*, ompt_data_t*, const void*>;

    using work_cb_t = openmp::ompt_wrapper<
        component_type, connector_type, openmp::mode::measure_callback, ompt_work_type_t,
        ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, uint64_t, const void*>;

    using thread_begin_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_thread_type_t, ompt_data_t*>;

    using thread_end_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*>;

    using implicit_task_cb_t =
        openmp::ompt_wrapper<component_type, connector_type,
                             openmp::mode::measure_callback, ompt_scope_endpoint_t,
                             ompt_data_t*, ompt_data_t*, unsigned int, unsigned int>;

    using sync_region_cb_t =
        openmp::ompt_wrapper<component_type, connector_type,
                             openmp::mode::measure_callback, ompt_sync_region_kind_t,
                             ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*,
                             const void*>;

    using mutex_acquire_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_mutex_kind_t, unsigned int, unsigned int, omp_wait_id_t,
                             const void*>;

    using mutex_acquired_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::end_callback,
                             ompt_mutex_kind_t, omp_wait_id_t, const void*>;

    using mutex_released_cb_t = mutex_acquired_cb_t;

    using target_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_target_type_t, ompt_scope_endpoint_t, int, ompt_data_t*,
                             ompt_id_t, const void*>;

    using target_data_op_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_id_t, ompt_id_t, ompt_target_data_op_t, void*, int,
                             void*, int, size_t, const void*>;

    using target_submit_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_id_t, ompt_id_t, unsigned int>;

    timemory_ompt_register_callback(ompt_callback_parallel_begin,
                                    TIMEMORY_OMPT_CBDECL(parallel_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_parallel_end,
                                    TIMEMORY_OMPT_CBDECL(parallel_end_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_task_create,
                                    TIMEMORY_OMPT_CBDECL(task_create_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_task_schedule,
                                    TIMEMORY_OMPT_CBDECL(task_schedule_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_implicit_task,
                                    TIMEMORY_OMPT_CBDECL(implicit_task_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_thread_begin,
                                    TIMEMORY_OMPT_CBDECL(thread_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_thread_end,
                                    TIMEMORY_OMPT_CBDECL(thread_end_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target,
                                    TIMEMORY_OMPT_CBDECL(target_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target_data_op,
                                    TIMEMORY_OMPT_CBDECL(target_data_op_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target_submit,
                                    TIMEMORY_OMPT_CBDECL(target_submit_cb_t::callback));

    timemory_ompt_register_callback(ompt_callback_master,
                                    TIMEMORY_OMPT_CBDECL(master_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_work,
                                    TIMEMORY_OMPT_CBDECL(work_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_sync_region,
                                    TIMEMORY_OMPT_CBDECL(sync_region_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_mutex_acquire,
                                    TIMEMORY_OMPT_CBDECL(mutex_acquire_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_mutex_acquired,
                                    TIMEMORY_OMPT_CBDECL(mutex_acquired_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_mutex_released,
                                    TIMEMORY_OMPT_CBDECL(mutex_released_cb_t::callback));

    if(tim::settings::verbose() > 0 || tim::settings::debug())
        printf("\n");

    return 1;  // success
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OMPT_LINKAGE(void)
ompt_finalize(ompt_data_t* tool_data)
{
    printf("\n");
    tim::consume_parameters(tool_data);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OMPT_LINKAGE(ompt_start_tool_result_t*)
ompt_start_tool(unsigned int omp_version, const char* runtime_version)
{
    printf("\n[timemory]> OpenMP version: %u, runtime version: %s\n\n", omp_version,
           runtime_version);
    static auto data =
        ompt_start_tool_result_t{ &ompt_initialize, &ompt_finalize, { 0 } };
    return (ompt_start_tool_result_t*) &data;
}
//
//--------------------------------------------------------------------------------------//
//
#else
//
#endif
//
//--------------------------------------------------------------------------------------//
