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
#include "timemory/macros.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/variadic/types.hpp"
//
#include "timemory/components/ompt/components.hpp"
#include "timemory/components/user_bundle/components.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/variadic/component_tuple.hpp"

#include <omp.h>
#include <ompt.h>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
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
            { ompt_callback_task_dependences, "task_dependences" },
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
        };

        auto itr = _instance.find(eid);
        return (itr == _instance.end()) ? get_unknown_identifier(eid) : itr->second;
    }
};

//--------------------------------------------------------------------------------------//

namespace mode
{
/// \class openmp::mode::begin_callback
/// \brief This is the beginning of a paired callback
struct begin_callback
{};
/// \class openmp::mode::end_callback
/// \brief This is the end of a paired callback
struct end_callback
{};
/// \class openmp::mode::measure_callback
/// \brief This is a sampling callback
struct measure_callback
{};
/// \class openmp::mode::endpoint_callback
/// \brief This is a callback whose first argument designates an endpoint
struct endpoint_callback
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
};

//--------------------------------------------------------------------------------------//

static const char* ompt_thread_type_labels[] = { nullptr, "ompt_thread_initial",
                                                 "ompt_thread_worker",
                                                 "ompt_thread_other" };
/*
static const char* ompt_task_status_labels[] = { nullptr, "ompt_task_complete",
                                                 "ompt_task_yield", "ompt_task_cancel",
                                                 "ompt_task_others" };
static const char* ompt_cancel_flag_labels[] = {
    "ompt_cancel_parallel",      "ompt_cancel_sections",  "ompt_cancel_do",
    "ompt_cancel_taskgroup",     "ompt_cancel_activated", "ompt_cancel_detected",
    "ompt_cancel_discarded_task"
};
*/

static const char* ompt_work_labels[] = { nullptr,
                                          "ompt_work_loop",
                                          "ompt_work_sections",
                                          "ompt_work_single_executor",
                                          "ompt_work_single_other",
                                          "ompt_work_workshare",
                                          "ompt_work_distribute",
                                          "ompt_work_taskloop" };

static const char* ompt_sync_region_labels[] = { nullptr, "ompt_sync_region_barrier",
                                                 "ompt_sync_region_taskwait",
                                                 "ompt_sync_region_taskgroup" };

//--------------------------------------------------------------------------------------//

template <typename Api = api::native_tag>
struct context_handler
{
public:
    //----------------------------------------------------------------------------------//
    // parallel begin
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parent_task_data, const omp_frame_t* parent_task_frame,
                    ompt_data_t* parallel_data, uint32_t requested_team_size,
                    const void* codeptr)
    {
        consume_parameters(parent_task_frame, requested_team_size, codeptr);
        m_key = "ompt_parallel";
        m_id  = std::hash<void*>()(parent_task_data) + std::hash<void*>()(parallel_data);
    }

    //----------------------------------------------------------------------------------//
    // parallel end
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parallel_data, ompt_data_t* parent_task_data,
                    const void* codeptr)
    {
        m_key = "ompt_parallel";
        m_id  = std::hash<void*>()(parent_task_data) + std::hash<void*>()(parallel_data);
        consume_parameters(codeptr);
    }

    //----------------------------------------------------------------------------------//
    // task create
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parent_task_data, const omp_frame_t* parent_frame,
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
        m_key = "ompt_master";
        m_id  = std::hash<std::string>()(m_key) + std::hash<void*>()(parallel_data) +
               std::hash<void*>()(task_data);
        consume_parameters(endpoint, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback work
    //----------------------------------------------------------------------------------//
    context_handler(ompt_work_type_t wstype, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data, uint64_t count,
                    const void* codeptr)
    {
        m_key = ompt_work_labels[wstype];
        m_id  = std::hash<std::string>()(m_key) + std::hash<void*>()(parallel_data) +
               std::hash<void*>()(task_data);
        consume_parameters(endpoint, count, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback thread begin
    //----------------------------------------------------------------------------------//
    context_handler(ompt_thread_type_t thread_type, ompt_data_t* thread_data)
    {
        m_key = ompt_thread_type_labels[thread_type];
        m_id  = std::hash<void*>()(static_cast<void*>(thread_data));
    }

    //----------------------------------------------------------------------------------//
    // callback thread end
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* thread_data)
    {
        m_id = std::hash<void*>()(static_cast<void*>(thread_data));
    }

    //----------------------------------------------------------------------------------//
    // callback implicit task
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, unsigned int team_size,
                    unsigned int thread_num)
    {
        m_key = apply<std::string>::join("_", "ompt_implicit_task", thread_num, "of",
                                         team_size);
        m_id  = std::hash<void*>()(static_cast<void*>(parallel_data)) +
               std::hash<std::string>()(m_key);
        consume_parameters(endpoint, task_data);
    }

    //----------------------------------------------------------------------------------//
    // callback sync region
    //----------------------------------------------------------------------------------//
    context_handler(ompt_sync_region_kind_t kind, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data,
                    const void* codeptr)
    {
        m_key = ompt_sync_region_labels[kind];
        m_id  = std::hash<size_t>()(static_cast<int>(kind)) +
               std::hash<void*>()(static_cast<void*>(parallel_data)) +
               std::hash<std::string>()(m_key);
        consume_parameters(endpoint, task_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback idle
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint)
    {
        m_key = "ompt_idle";
        m_id  = std::hash<std::string>()(m_key);
        consume_parameters(endpoint);
    }

    //----------------------------------------------------------------------------------//
    // callback mutex acquire
    //----------------------------------------------------------------------------------//
    context_handler(ompt_mutex_kind_t kind, unsigned int hint, unsigned int impl,
                    omp_wait_id_t wait_id, const void* codeptr)
    {
        m_key = "ompt_mutex";
        switch(kind)
        {
            case ompt_mutex: break;
            case ompt_mutex_lock: m_key += "_lock"; break;
            case ompt_mutex_nest_lock: m_key += "_nest_lock"; break;
            case ompt_mutex_critical: m_key += "_critical"; break;
            case ompt_mutex_atomic: m_key += "_atomic"; break;
            case ompt_mutex_ordered: m_key += "_ordered"; break;
            default: m_key += "_unknown"; break;
        }
        m_id = std::hash<size_t>()(static_cast<int>(kind) + static_cast<int>(wait_id)) +
               std::hash<std::string>()(m_key);
        consume_parameters(hint, impl, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback mutex acquired
    // callback mutex released
    //----------------------------------------------------------------------------------//
    context_handler(ompt_mutex_kind_t kind, omp_wait_id_t wait_id, const void* codeptr)
    {
        consume_parameters(codeptr);
        m_key = "ompt_mutex";
        switch(kind)
        {
            case ompt_mutex: break;
            case ompt_mutex_lock: m_key += "_lock"; break;
            case ompt_mutex_nest_lock: m_key += "_nest_lock"; break;
            case ompt_mutex_critical: m_key += "_critical"; break;
            case ompt_mutex_atomic: m_key += "_atomic"; break;
            case ompt_mutex_ordered: m_key += "_ordered"; break;
            default: m_key += "_unknown"; break;
        }
        m_id = std::hash<size_t>()(static_cast<int>(kind) + static_cast<int>(wait_id)) +
               std::hash<std::string>()(m_key);
    }

    //----------------------------------------------------------------------------------//
    // callback target
    //----------------------------------------------------------------------------------//
    context_handler(ompt_target_type_t kind, ompt_scope_endpoint_t endpoint,
                    int device_num, ompt_data_t* task_data, ompt_id_t target_id,
                    const void* codeptr)
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
    using api_type            = Api;
    using type                = Components;
    using result_type         = std::shared_ptr<type>;
    using array_type          = std::deque<result_type>;
    using map_type            = std::unordered_map<size_t, array_type>;
    using handle_type         = component::ompt_handle<api_type>;
    using handle_storage_type = typename handle_type::storage_type;
    using bundle_storage_type = typename component::user_ompt_bundle::storage_type;

    static bool is_enabled()
    {
        if(handle_storage_type::is_finalizing() || !manager::instance() ||
           (manager::instance() && manager::instance()->is_finalizing()))
            trait::runtime_enabled<handle_type>::set(false);
        return trait::runtime_enabled<handle_type>::get();
    }

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::begin_callback>::value), int> = 0>
    callback_connector(T, Args... args)
    {
        if(!is_enabled())
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
        if(!is_enabled())
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
        if(!is_enabled())
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

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::endpoint_callback>::value), int> = 0>
    callback_connector(T, ompt_scope_endpoint_t endp, Args... args)
    {
        if(!is_enabled())
            return;

        context_handler<api_type> ctx(endp, args...);

        // don't provide empty entries
        if(ctx.key().empty())
            return;

        if(endp == ompt_scope_begin)
        {
            auto c = std::make_shared<type>(ctx.key());

            // persistence handling
            get_key_map()[ctx.id()].emplace_back(c);

            c->construct(endp, args...);
            c->start();
            c->audit(ctx.key(), endp, args...);
        }
        else if(endp == ompt_scope_end)
        {
            // persistence handling
            auto itr = get_key_map().find(ctx.id());
            if(itr == get_key_map().end())
                return;
            if(itr->second.empty())
                return;
            auto c = itr->second.back();
            itr->second.pop_back();

            c->audit(ctx.key(), endp, args...);
            c->stop();
        }
    }

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::endpoint_callback>::value), int> = 0>
    callback_connector(T, ompt_work_type_t workv, ompt_scope_endpoint_t endp,
                       Args... args)
    {
        if(!is_enabled())
            return;
        generic_endpoint_connector(T{}, workv, endp, args...);
    }

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::endpoint_callback>::value), int> = 0>
    callback_connector(T, ompt_sync_region_kind_t syncv, ompt_scope_endpoint_t endp,
                       Args... args)
    {
        if(!is_enabled())
            return;
        generic_endpoint_connector(T{}, syncv, endp, args...);
    }

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::endpoint_callback>::value), int> = 0>
    callback_connector(T, ompt_target_type_t targv, ompt_scope_endpoint_t endp,
                       Args... args)
    {
        if(!is_enabled())
            return;
        generic_endpoint_connector(T{}, targv, endp, args...);
    }

protected:
    template <typename T, typename Arg, typename... Args,
              enable_if_t<(std::is_same<T, mode::endpoint_callback>::value), int> = 0>
    void generic_endpoint_connector(T, Arg arg, ompt_scope_endpoint_t endp, Args... args)
    {
        context_handler<api_type> ctx(arg, endp, args...);

        // don't provide empty entries
        if(ctx.key().empty())
            return;

        if(endp == ompt_scope_begin)
        {
            auto c = std::make_shared<type>(ctx.key());

            // persistence handling
            get_key_map()[ctx.id()].emplace_back(c);

            c->construct(arg, endp, args...);
            c->start();
            c->audit(ctx.key(), arg, endp, args...);
        }
        else if(endp == ompt_scope_end)
        {
            // persistence handling
            auto itr = get_key_map().find(ctx.id());
            if(itr == get_key_map().end())
                return;
            if(itr->second.empty())
                return;
            auto c = itr->second.back();
            itr->second.pop_back();

            c->audit(ctx.key(), arg, endp, args...);
            c->stop();
        }
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
}  // namespace tim
