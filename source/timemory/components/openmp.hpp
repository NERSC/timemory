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

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/variadic/types.hpp"

#include "timemory/runtime/configure.hpp"

#include <omp.h>
#include <ompt.h>

#if !defined(TIMEMORY_OMPT_API_TAG)
#    define TIMEMORY_OMPT_API_TAG ::tim::api::native_tag
#endif

// for callback declarations
#if !defined(CBDECL)
#    define CBDECL(NAME) (ompt_callback_t) & NAME
#endif

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

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//

template <typename Api = api::native_tag>
struct omp_tools
{};

//--------------------------------------------------------------------------------------//

namespace openmp
{
//--------------------------------------------------------------------------------------//

template <typename Enumeration>
static std::string
get_unknown_identifier(Enumeration eid)
{
    using type = Enumeration;
    auto&& ret =
        apply<std::string>::join("-", "unspecialized-enumeration",
                                 demangle<type>().c_str(), static_cast<int>(eid));
    return std::move(ret);
}

//--------------------------------------------------------------------------------------//

template <typename Enumeration>
struct identifier
{
    using type = Enumeration;
    static std::string get(type eid) { return get_unknown_identifier(eid); }
};

//--------------------------------------------------------------------------------------//

template <>
struct identifier<ompt_callbacks_t>
{
    using type      = ompt_callbacks_t;
    using key_map_t = std::unordered_map<int, std::string>;

    static std::string get(type eid)
    {
        static key_map_t _instance = {
            { ompt_callback_thread_begin, "thread_begin" },
            { ompt_callback_thread_end, "thread_end" },
            { ompt_callback_parallel_begin, "parallel_begin" },
            { ompt_callback_parallel_end, "parallel_end" },
            { ompt_callback_task_create, "task_create" },
            { ompt_callback_task_schedule, "task_schedule" },
            { ompt_callback_implicit_task, "implicit_task" },
            { ompt_callback_target, "target" },
            { ompt_callback_target_data_op, "target_data_op" },
            { ompt_callback_target_submit, "target_submit" },
            { ompt_callback_control_tool, "control_tool" },
            { ompt_callback_device_initialize, "device_initialize" },
            { ompt_callback_device_finalize, "device_finalize" },
            { ompt_callback_device_load, "device_load" },
            { ompt_callback_device_unload, "device_unload" },
            { ompt_callback_sync_region_wait, "sync_region_wait" },
            { ompt_callback_mutex_released, "mutex_released" },
            { ompt_callback_dependences, "dependences" },
            { ompt_callback_task_dependence, "task_dependence" },
            { ompt_callback_work, "work" },
            { ompt_callback_master, "master" },
            { ompt_callback_target_map, "target_map" },
            { ompt_callback_sync_region, "sync_region" },
            { ompt_callback_lock_init, "lock_init" },
            { ompt_callback_lock_destroy, "lock_destroy" },
            { ompt_callback_mutex_acquire, "mutex_acquire" },
            { ompt_callback_mutex_acquired, "mutex_acquired" },
            { ompt_callback_nest_lock, "nest_lock" },
            { ompt_callback_flush, "flush" },
            { ompt_callback_cancel, "cancel" },
            { ompt_callback_reduction, "reduction" },
            { ompt_callback_dispatch, "dispatch" },
        };

        auto itr = _instance.find(eid);
        return (itr == _instance.end()) ? get_unknown_identifier(eid) : itr->second;
    }
};

//--------------------------------------------------------------------------------------//

namespace mode
{
/// \class openmp::mode::begin_callback
/// \brief This is the beginning of a dual callback
struct begin_callback
{};
/// \class openmp::mode::end_callback
/// \brief This is the beginning of a dual callback
struct end_callback
{};
/// \class openmp::mode::standalone_callback
/// \brief This is the beginning of a dual callback
struct measure_callback
{};
}  // namespace mode

//--------------------------------------------------------------------------------------//

template <typename Components, typename Connector, typename Mode, typename... Args>
struct ompt_wrapper
{
    // using result_type    = ReturnType;
    using args_type      = std::tuple<Args...>;
    using component_type = Components;

    static void callback(Args... args) { Connector(Mode{}, args...); }

    /*
    static std::string key()
    {
        static std::string _instance = identifier<Enumeration>(Eid);
        return _instance;
    }
    */
};

//--------------------------------------------------------------------------------------//

static const char* ompt_thread_type_labels[] = { NULL, "ompt_thread_initial",
                                                 "ompt_thread_worker",
                                                 "ompt_thread_other" };

static const char* ompt_task_status_labels[] = { NULL, "ompt_task_complete",
                                                 "ompt_task_yield", "ompt_task_cancel",
                                                 "ompt_task_others" };
static const char* ompt_cancel_flag_labels[] = {
    "ompt_cancel_parallel",      "ompt_cancel_sections",  "ompt_cancel_do",
    "ompt_cancel_taskgroup",     "ompt_cancel_activated", "ompt_cancel_detected",
    "ompt_cancel_discarded_task"
};

//--------------------------------------------------------------------------------------//

template <typename Api = api::native_tag>
struct context_handler
{
public:
    //----------------------------------------------------------------------------------//
    // parallel begin
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parent_task_data, const ompt_frame_t* parent_task_frame,
                    ompt_data_t* parallel_data, uint32_t requested_team_size,
                    const void* codeptr)
    {
        consume_parameters(parent_task_frame, requested_team_size, codeptr);
        m_key += "ompt_parallel";
        m_id = std::hash<void*>()(parent_task_data) + std::hash<void*>()(parallel_data);
    }
    //----------------------------------------------------------------------------------//
    // parallel end
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parallel_data, ompt_data_t* parent_task_data,
                    const void* codeptr)
    {
        consume_parameters(parallel_data, codeptr);
        m_key = "ompt_parallel";
        m_id  = std::hash<void*>()(parent_task_data) + std::hash<void*>()(parallel_data);
    }
    //----------------------------------------------------------------------------------//
    // task create
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parent_task_data, const ompt_frame_t* parent_frame,
                    ompt_data_t* new_task_data, int type, int has_dependences,
                    const void* codeptr)
    {
        consume_parameters(parent_task_data, parent_frame, new_task_data, type,
                           has_dependences, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // task scheduler
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* prior_task_data, ompt_task_status_t prior_task_status,
                    ompt_data_t* next_task_data)
    {
        consume_parameters(prior_task_data, prior_task_status, next_task_data);
    }

    //----------------------------------------------------------------------------------//
    // callback master
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, const void* codeptr)
    {
        consume_parameters(endpoint, parallel_data, task_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback work
    //----------------------------------------------------------------------------------//
    context_handler(ompt_work_t wstype, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data, uint64_t count,
                    const void* codeptr)
    {
        consume_parameters(wstype, endpoint, parallel_data, task_data, count, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback thread begin
    //----------------------------------------------------------------------------------//
    context_handler(ompt_thread_t thread_type, ompt_data_t* thread_data)
    {
        consume_parameters(thread_type, thread_data);
    }

    //----------------------------------------------------------------------------------//
    // callback thread end
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* thread_data) { consume_parameters(thread_data); }

    //----------------------------------------------------------------------------------//
    // callback implicit task
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, unsigned int team_size,
                    unsigned int thread_num)
    {
        consume_parameters(endpoint, parallel_data, task_data, team_size, thread_num);
    }

    //----------------------------------------------------------------------------------//
    // callback sync region
    //----------------------------------------------------------------------------------//
    context_handler(ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data,
                    const void* codeptr)
    {
        consume_parameters(kind, endpoint, parallel_data, task_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback idle
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint) { consume_parameters(endpoint); }

    //----------------------------------------------------------------------------------//
    // callback mutex acquire
    //----------------------------------------------------------------------------------//
    context_handler(ompt_mutex_t kind, unsigned int hint, unsigned int impl,
                    ompt_wait_id_t wait_id, const void* codeptr)
    {
        m_key = "ompt_mutex";
        switch(kind)
        {
            case ompt_mutex_lock: m_key += "_lock"; break;
            case ompt_mutex_nest_lock: m_key += "_nest_lock"; break;
            case ompt_mutex_test_lock: m_key += "_test_lock"; break;
            case ompt_mutex_test_nest_lock: m_key += "_test_nest_lock"; break;
            case ompt_mutex_critical: m_key += "_critical"; break;
            case ompt_mutex_atomic: m_key += "_atomic"; break;
            case ompt_mutex_ordered: m_key += "_ordered"; break;
            // case ompt_mutex: m_key += "_generic"; break;
            default: m_key += "_generic"; break;
        }
        m_id = std::hash<size_t>()(static_cast<int>(kind) + static_cast<int>(wait_id)) +
               std::hash<std::string>()(m_key);
        consume_parameters(hint, impl, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback mutex acquired
    // callback mutex released
    //----------------------------------------------------------------------------------//
    context_handler(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void* codeptr)
    {
        consume_parameters(codeptr);
        m_key = "ompt_mutex";
        switch(kind)
        {
            case ompt_mutex_lock: m_key += "_lock"; break;
            case ompt_mutex_nest_lock: m_key += "_nest_lock"; break;
            case ompt_mutex_test_lock: m_key += "_test_lock"; break;
            case ompt_mutex_test_nest_lock: m_key += "_test_nest_lock"; break;
            case ompt_mutex_critical: m_key += "_critical"; break;
            case ompt_mutex_atomic: m_key += "_atomic"; break;
            case ompt_mutex_ordered: m_key += "_ordered"; break;
            // case ompt_mutex: m_key += "_generic"; break;
            default: m_key += "_generic"; break;
        }
        m_id = std::hash<size_t>()(static_cast<int>(kind) + static_cast<int>(wait_id)) +
               std::hash<std::string>()(m_key);
    }

    //----------------------------------------------------------------------------------//
    // callback target
    //----------------------------------------------------------------------------------//
    context_handler(ompt_target_t kind, ompt_scope_endpoint_t endpoint, int device_num,
                    ompt_data_t* task_data, ompt_id_t target_id, const void* codeptr)
    {
        m_key = apply<std::string>::join("_", "ompt_target_device", device_num);
        m_id =
            std::hash<size_t>()(static_cast<int>(kind) + static_cast<int>(device_num)) +
            std::hash<std::string>()(m_key);
        consume_parameters(kind, endpoint, device_num, task_data, target_id, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback target data op
    //----------------------------------------------------------------------------------//
    context_handler(ompt_id_t target_id, ompt_id_t host_op_id,
                    ompt_target_data_op_t optype, void* src_addr, int src_device_num,
                    void* dest_addr, int dest_device_num, size_t bytes,
                    const void* codeptr)
    {
        m_key = apply<std::string>::join("", "ompt_target_data_op_src:", src_device_num,
                                         "_dest:", dest_device_num);
        m_id  = std::hash<size_t>()(static_cast<int>(src_device_num) +
                                   static_cast<int>(dest_device_num)) +
               std::hash<std::string>()(m_key) + std::hash<void*>()(src_addr) +
               std::hash<void*>()(dest_addr);
        consume_parameters(target_id, host_op_id, optype, bytes, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback target submit
    //----------------------------------------------------------------------------------//
    context_handler(ompt_id_t target_id, ompt_id_t host_op_id,
                    unsigned int requested_num_teams)
    {
        m_key = "ompt_target_submit";
        consume_parameters(target_id, host_op_id, requested_num_teams);
    }

public:
    size_t             id() const { return m_id; }
    const std::string& key() const { return m_key; }

protected:
    // the first hash is the string hash, the second is the data hash
    size_t      m_id;
    std::string m_key;
};

//--------------------------------------------------------------------------------------//

template <typename Components, typename Api = api::native_tag>
struct callback_connector
{
    using api_type    = Api;
    using type        = Components;
    using result_type = std::shared_ptr<type>;
    using array_type  = std::deque<result_type>;
    using map_type    = std::unordered_map<size_t, array_type>;

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::begin_callback>::value), int> = 0>
    callback_connector(T, Args... args)
    {
        if(!trait::runtime_enabled<omp_tools<api_type>>::get())
            return;

        context_handler<api_type> ctx(args...);

        // don't provide empty entries
        if(ctx.key().empty())
            return;

        auto c = std::make_shared<type>(ctx.key());

        // persistence handling
        get_key_map()[ctx.id()].emplace_back(c);

        c->construct(args...);
        c->start();
        c->audit(ctx.key(), args...);
    }

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::end_callback>::value), int> = 0>
    callback_connector(T, Args... args)
    {
        if(!trait::runtime_enabled<omp_tools<api_type>>::get())
            return;

        context_handler<api_type> ctx(args...);

        // don't provide empty entries
        if(ctx.key().empty())
            return;

        // persistence handling
        auto itr = get_key_map().find(ctx.id());
        if(itr == get_key_map().end())
            return;
        if(itr->second.empty())
            return;
        auto c = itr->second.back();
        itr->second.pop_back();

        c->audit(ctx.key(), args...);
        c->stop();
    }

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::measure_callback>::value), int> = 0>
    callback_connector(T, Args... args)
    {
        if(!trait::runtime_enabled<omp_tools<api_type>>::get())
            return;

        context_handler<api_type> ctx(args...);

        // don't provide empty entries
        if(ctx.key().empty())
            return;

        auto c = std::make_shared<type>(ctx.key());

        c->construct(args...);
        c->audit(ctx.key(), args...);
        c->measure();
    }

private:
    static map_type& get_key_map()
    {
        static thread_local map_type _instance;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace openmp
}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//

extern "C" int
ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                ompt_data_t* tool_data)
{
    using namespace tim::component;
    using api_type       = TIMEMORY_OMPT_API_TAG;
    using component_type = typename tim::trait::omp_tools<api_type>::type;
    using policy_type    = tim::policy::omp_tools<api_type, component_type>;
    using connector_type = openmp::callback_connector<component_type, api_type>;

    if(!tim::trait::is_available<omp_tools<api_type>>::value)
        return 1;

    tim::consume_parameters(initial_device_num, tool_data);

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
                             ompt_data_t*, const ompt_frame_t*, ompt_data_t*, uint32_t,
                             const void*>;

    using parallel_end_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*, ompt_data_t*, const void*>;

    using task_create_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, const ompt_frame_t*, ompt_data_t*, int, int,
                             const void*>;

    using task_schedule_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, ompt_task_status_t, ompt_data_t*>;

    using master_cb_t =
        openmp::ompt_wrapper<component_type, connector_type,
                             openmp::mode::measure_callback, ompt_scope_endpoint_t,
                             ompt_data_t*, ompt_data_t*, const void*>;

    using work_cb_t = openmp::ompt_wrapper<
        component_type, connector_type, openmp::mode::measure_callback, ompt_work_t,
        ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, uint64_t, const void*>;

    using thread_begin_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_thread_t, ompt_data_t*>;

    using thread_end_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*>;

    using implicit_task_cb_t =
        openmp::ompt_wrapper<component_type, connector_type,
                             openmp::mode::measure_callback, ompt_scope_endpoint_t,
                             ompt_data_t*, ompt_data_t*, unsigned int, unsigned int>;

    using sync_region_cb_t =
        openmp::ompt_wrapper<component_type, connector_type,
                             openmp::mode::measure_callback, ompt_sync_region_t,
                             ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*,
                             const void*>;

    using mutex_acquire_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_mutex_t, unsigned int, unsigned int, ompt_wait_id_t,
                             const void*>;

    using mutex_acquired_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::end_callback,
                             ompt_mutex_t, ompt_wait_id_t, const void*>;

    using mutex_released_cb_t = mutex_acquired_cb_t;

    using target_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_target_t, ompt_scope_endpoint_t, int, ompt_data_t*,
                             ompt_id_t, const void*>;

    using target_data_op_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_id_t, ompt_id_t, ompt_target_data_op_t, void*, int,
                             void*, int, size_t, const void*>;

    using target_submit_cb_t =
        openmp::ompt_wrapper<component_type, connector_type, openmp::mode::begin_callback,
                             ompt_id_t, ompt_id_t, unsigned int>;

    timemory_ompt_register_callback(ompt_callback_parallel_begin,
                                    CBDECL(parallel_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_parallel_end,
                                    CBDECL(parallel_end_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_task_create,
                                    CBDECL(task_create_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_task_schedule,
                                    CBDECL(task_schedule_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_implicit_task,
                                    CBDECL(implicit_task_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_thread_begin,
                                    CBDECL(thread_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_thread_end,
                                    CBDECL(thread_end_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target, CBDECL(target_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target_data_op,
                                    CBDECL(target_data_op_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target_submit,
                                    CBDECL(target_submit_cb_t::callback));

    timemory_ompt_register_callback(ompt_callback_master, CBDECL(master_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_work, CBDECL(work_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_sync_region,
                                    CBDECL(sync_region_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_mutex_acquire,
                                    CBDECL(mutex_acquire_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_mutex_acquired,
                                    CBDECL(mutex_acquired_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_mutex_released,
                                    CBDECL(mutex_released_cb_t::callback));

    if(tim::settings::verbose() > 0 || tim::settings::debug())
        printf("\n");

    return 1;  // success
}

//--------------------------------------------------------------------------------------//

extern "C" void
ompt_finalize(ompt_data_t* tool_data)
{
    printf("\n");
    tim::consume_parameters(tool_data);
}

//--------------------------------------------------------------------------------------//

extern "C" ompt_start_tool_result_t*
ompt_start_tool(unsigned int omp_version, const char* runtime_version)
{
    printf("\n[timemory]> OpenMP version: %u, runtime version: %s\n\n", omp_version,
           runtime_version);
    uint64_t     tool_data = 0;
    static auto* data =
        new ompt_start_tool_result_t{ &ompt_initialize, &ompt_finalize, { tool_data } };
    return (ompt_start_tool_result_t*) data;
}

//--------------------------------------------------------------------------------------//
