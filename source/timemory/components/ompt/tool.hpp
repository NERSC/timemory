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

#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
//
#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/components.hpp"
//
#include <deque>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace openmp
{
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_thread_type_labels[] = { nullptr, "ompt_thread_initial",
                                                 "ompt_thread_worker",
                                                 "ompt_thread_other" };
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_dispatch_type_labels[] = { nullptr, "ompt_dispatch_iteration",
                                                   "ompt_dispatch_section" };
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_sync_region_type_labels[] = {
    nullptr,
    "ompt_sync_region_barrier",
    "ompt_sync_region_barrier_implicit",
    "ompt_sync_region_barrier_explicit",
    "ompt_sync_region_barrier_implementation",
    "ompt_sync_region_taskwait",
    "ompt_sync_region_taskgroup",
    "ompt_sync_region_reduction"
};
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_target_type_labels[] = { nullptr, "ompt_target",
                                                 "ompt_target_enter_data",
                                                 "ompt_target_exit_data",
                                                 "ompt_target_update" };
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_work_labels[] = { nullptr,
                                          "ompt_work_loop",
                                          "ompt_work_sections",
                                          "ompt_work_single_executor",
                                          "ompt_work_single_other",
                                          "ompt_work_workshare",
                                          "ompt_work_distribute",
                                          "ompt_work_taskloop" };
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_target_data_op_labels[] = { nullptr, "ompt_target_data_alloc",
                                                    "ompt_target_data_transfer_to_dev",
                                                    "ompt_target_data_transfer_from_dev",
                                                    "ompt_target_data_delete" };
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_task_status_labels[] = { nullptr,
                                                 "ompt_task_complete",
                                                 "ompt_task_yield",
                                                 "ompt_task_cancel",
                                                 "ompt_task_detach",
                                                 "ompt_task_early_fulfill",
                                                 "ompt_task_late_fulfill",
                                                 "ompt_task_switch" };
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_mutex_t, const char*> ompt_mutex_type_labels = {
    { ompt_mutex_lock, "ompt_mutex_lock" },
    { ompt_mutex_test_lock, "ompt_mutex_test_lock" },
    { ompt_mutex_nest_lock, "ompt_mutex_nest_lock" },
    { ompt_mutex_test_nest_lock, "ompt_mutex_test_nest_lock" },
    { ompt_mutex_critical, "ompt_mutex_critical" },
    { ompt_mutex_atomic, "ompt_mutex_atomic" },
    { ompt_mutex_ordered, "ompt_mutex_ordered" }
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_task_flag_t, const char*> ompt_task_type_labels = {
    { ompt_task_initial, "ompt_task_initial" },
    { ompt_task_implicit, "ompt_task_implicit" },
    { ompt_task_explicit, "ompt_task_explicit" },
    { ompt_task_target, "ompt_task_target" },
    { ompt_task_undeferred, "ompt_task_undeferred" },
    { ompt_task_untied, "ompt_task_untied" },
    { ompt_task_final, "ompt_task_final" },
    { ompt_task_mergeable, "ompt_task_mergeable" },
    { ompt_task_merged, "ompt_task_merged" }
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_target_map_flag_t, const char*> ompt_target_map_labels = {
    { ompt_target_map_flag_to, "ompt_target_map_flag_to" },
    { ompt_target_map_flag_from, "ompt_target_map_flag_from" },
    { ompt_target_map_flag_alloc, "ompt_target_map_flag_alloc" },
    { ompt_target_map_flag_release, "ompt_target_map_flag_release" },
    { ompt_target_map_flag_delete, "ompt_target_map_flag_delete" },
    { ompt_target_map_flag_implicit, "ompt_target_map_flag_implicit" }
};
//
//--------------------------------------------------------------------------------------//
//
#define TIMEMORY_OMPT_ENUM_LABEL(TYPE)                                                   \
    {                                                                                    \
        TYPE, #TYPE                                                                      \
    }
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_dependence_type_t, const char*> ompt_dependence_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_in),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_out),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_inout),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_mutexinoutset),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_source),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_sink)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_cancel_flag_t, const char*> ompt_cancel_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_parallel),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_sections),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_loop),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_taskgroup),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_activated),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_detected),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_discarded_task)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_callbacks_t, const char*> ompt_callback_labels = {
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_thread_begin),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_thread_end),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_parallel_begin),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_parallel_end),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_task_create),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_task_schedule),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_implicit_task),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_data_op),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_submit),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_control_tool),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_device_initialize),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_device_finalize),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_device_load),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_device_unload),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_sync_region_wait),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_mutex_released),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_dependences),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_task_dependence),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_work),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_master),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_map),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_sync_region),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_lock_init),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_lock_destroy),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_mutex_acquire),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_mutex_acquired),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_nest_lock),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_flush),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_cancel),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_reduction),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_dispatch)
};
//
//--------------------------------------------------------------------------------------//
//
inline void
ompt_suppress_unused_variable_warnings()
{
    consume_parameters(
        ompt_task_status_labels, ompt_target_data_op_labels, ompt_work_labels,
        ompt_target_type_labels, ompt_sync_region_type_labels, ompt_dispatch_type_labels,
        ompt_thread_type_labels, ompt_cancel_type_labels, ompt_dependence_type_labels,
        ompt_target_map_labels, ompt_task_type_labels, ompt_mutex_type_labels);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Api>
struct context_handler
{
    using api_type = Api;

public:
    template <typename KeyT, typename MappedT, typename HashT = std::hash<KeyT>>
    using uomap_t = std::unordered_map<KeyT, MappedT, HashT>;

    template <typename Tag, typename KeyT = uint64_t, typename MappedT = ompt_data_t*,
              typename MapT = uomap_t<KeyT, MappedT>>
    static auto& get_data()
    {
        static thread_local MapT _instance;
        return _instance;
    }

    // tags for above
    struct device_state_tag
    {};
    struct device_load_tag
    {};
    struct task_tag
    {};
    struct mutex_tag
    {};
    struct nest_lock_tag
    {};

    using data_map_t = uomap_t<uint64_t, ompt_data_t*>;

public:
    //----------------------------------------------------------------------------------//
    // callback thread begin
    //----------------------------------------------------------------------------------//
    context_handler(ompt_thread_t thread_type, ompt_data_t* thread_data)
    : m_key(ompt_thread_type_labels[thread_type])
    , m_data({ { thread_data, nullptr } })
    {}

    //----------------------------------------------------------------------------------//
    // callback thread end
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* thread_data)
    : m_data({ { thread_data, nullptr } })
    {}

    //----------------------------------------------------------------------------------//
    // parallel begin
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                    ompt_data_t* parallel_data, unsigned int requested_parallelism,
                    int flags, const void* codeptr)
    : m_key("ompt_parallel")
    , m_data({ { nullptr, parallel_data } })
    {
        consume_parameters(task_data, task_frame, requested_parallelism, flags, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // parallel end
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parallel_data, ompt_data_t* task_data, int flags,
                    const void* codeptr)
    : m_key("ompt_parallel")
    , m_data({ { nullptr, parallel_data } })
    {
        consume_parameters(task_data, flags, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback master
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, const void* codeptr)
    : m_key("ompt_master")
    , m_data(
          { { (endpoint == ompt_scope_begin) ? construct_data() : task_data, nullptr } })
    {
        consume_parameters(endpoint, parallel_data, task_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback implicit task
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, unsigned int team_size,
                    unsigned int thread_num)
    : m_key("ompt_implicit_task")
    , m_data(
          { { (endpoint == ompt_scope_begin) ? construct_data() : task_data, nullptr } })
    {
        consume_parameters(endpoint, parallel_data, task_data, team_size, thread_num);
    }

    //----------------------------------------------------------------------------------//
    // callback sync region
    //----------------------------------------------------------------------------------//
    context_handler(ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data,
                    const void* codeptr)
    : m_key(ompt_sync_region_type_labels[kind])
    , m_data(
          { { (endpoint == ompt_scope_begin) ? construct_data() : task_data, nullptr } })
    {
        consume_parameters(endpoint, parallel_data, task_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback mutex acquire
    //----------------------------------------------------------------------------------//
    context_handler(ompt_mutex_t kind, unsigned int hint, unsigned int impl,
                    ompt_wait_id_t wait_id, const void* codeptr)
    : m_key(ompt_mutex_type_labels[kind])
    , m_data({ { construct_data(), nullptr } })
    {
        get_data<mutex_tag>().insert({ wait_id, m_data[0] });
        consume_parameters(hint, impl, wait_id, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback mutex acquired
    // callback mutex released
    //----------------------------------------------------------------------------------//
    context_handler(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void* codeptr)
    : m_key(ompt_mutex_type_labels[kind])
    , m_data({ { nullptr, nullptr } })
    {
        if(get_data<mutex_tag>().find(wait_id) != get_data<mutex_tag>().end())
        {
            m_data[0] = get_data<mutex_tag>()[wait_id];
            m_cleanup = [=]() {
                auto& itr = get_data<mutex_tag>()[wait_id];
                delete itr;
                itr = nullptr;
                get_data<mutex_tag>().erase(wait_id);
            };
        }
        consume_parameters(codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback nest lock
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint, ompt_wait_id_t wait_id,
                    const void* codeptr)
    : m_key("ompt_nested_lock")
    , m_data({ { nullptr, nullptr } })
    {
        if(endpoint == ompt_scope_end &&
           get_data<nest_lock_tag>().find(wait_id) != get_data<nest_lock_tag>().end())
        {
            m_data[0] = get_data<nest_lock_tag>()[wait_id];
            m_cleanup = [=]() {
                auto& itr = get_data<nest_lock_tag>()[wait_id];
                delete itr;
                itr = nullptr;
                get_data<nest_lock_tag>().erase(wait_id);
            };
        }
        else if(endpoint == ompt_scope_begin)
        {
            m_data[0]                          = construct_data();
            get_data<nest_lock_tag>()[wait_id] = m_data[0];
        }

        consume_parameters(endpoint, wait_id, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback task create
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                    ompt_data_t* new_task_data, int flags, int has_dependences,
                    const void* codeptr)
    : m_key("ompt_task_create")
    , m_data({ { task_data, nullptr } })
    {
        consume_parameters(task_frame, new_task_data, flags, has_dependences, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback task scheduler
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* prior_task_data, ompt_task_status_t prior_task_status,
                    ompt_data_t* next_task_data)
    : m_key("ompt_task_schedule")
    , m_data({ { nullptr, next_task_data } })
    {
        consume_parameters(prior_task_data, prior_task_status, next_task_data);
    }

    //----------------------------------------------------------------------------------//
    // callback dispatch
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parallel_data, ompt_data_t* task_data,
                    ompt_dispatch_t kind, ompt_data_t instance)
    : m_key(ompt_dispatch_type_labels[kind])
    , m_data({ { task_data, nullptr } })
    {
        consume_parameters(parallel_data, task_data, kind, instance);
    }

    //----------------------------------------------------------------------------------//
    // callback work
    //----------------------------------------------------------------------------------//
    context_handler(ompt_work_t wstype, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data, uint64_t count,
                    const void* codeptr)
    : m_key(ompt_work_labels[wstype])
    , m_data(
          { { (endpoint == ompt_scope_begin) ? construct_data() : task_data, nullptr } })
    {
        consume_parameters(endpoint, parallel_data, task_data, count, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback flush
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* thread_data, const void* codeptr)
    : m_key("ompt_flush")
    , m_data({ { thread_data, nullptr } })
    {
        consume_parameters(thread_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback cancel
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* thread_data, int flags, const void* codeptr)
    : m_key("ompt_cancel")
    , m_data({ { thread_data, nullptr } })
    {
        consume_parameters(thread_data, flags, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback target
    //----------------------------------------------------------------------------------//
    context_handler(ompt_target_t kind, ompt_scope_endpoint_t endpoint, int device_num,
                    ompt_data_t* task_data, ompt_id_t target_id, const void* codeptr)
    : m_key(mpl::apply<std::string>::join('_', ompt_target_type_labels[kind], "dev",
                                          device_num))
    , m_data(
          { { (endpoint == ompt_scope_begin) ? construct_data() : task_data, nullptr } })
    {
        consume_parameters(kind, endpoint, target_id, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback target data op
    //----------------------------------------------------------------------------------//
    context_handler(ompt_id_t target_id, ompt_id_t host_op_id,
                    ompt_target_data_op_t optype, void* src_addr, int src_device_num,
                    void* dest_addr, int dest_device_num, size_t bytes,
                    const void* codeptr)
    : m_key(mpl::apply<std::string>::join('_', ompt_target_data_op_labels[optype], "src",
                                          src_device_num, "dest", dest_device_num))
    , m_data({ { construct_data(true), nullptr } })
    {
        consume_parameters(target_id, host_op_id, src_addr, dest_addr, bytes, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback target submit
    //----------------------------------------------------------------------------------//
    context_handler(ompt_id_t target_id, ompt_id_t host_op_id,
                    unsigned int requested_num_teams)
    : m_key("ompt_target_submit")
    , m_data({ { nullptr, nullptr } })
    {
        consume_parameters(target_id, host_op_id, requested_num_teams);
    }

    //----------------------------------------------------------------------------------//
    // callback target mapping
    //----------------------------------------------------------------------------------//
    context_handler(ompt_id_t target_id, unsigned int nitems, void** host_addr,
                    void** device_addr, size_t* bytes, unsigned int* mapping_flags)
    : m_key(mpl::apply<std::string>::join('_', "ompt_target_mapping", target_id))
    , m_data({ { nullptr, nullptr } })
    {
        consume_parameters(nitems, host_addr, device_addr, bytes, mapping_flags);
    }

    //----------------------------------------------------------------------------------//
    // callback target device initialize
    //----------------------------------------------------------------------------------//
    context_handler(uint64_t device_num, const char* type, ompt_device_t* device,
                    ompt_function_lookup_t lookup, const char* documentation)
    : m_key(mpl::apply<std::string>::join('_', "ompt_device", device_num, type))
    , m_data({ { construct_data(), nullptr } })
    {
        get_data<device_state_tag>().insert({ device_num, m_data[0] });
        consume_parameters(device, lookup, documentation);
    }

    //----------------------------------------------------------------------------------//
    // callback target device finalize
    //----------------------------------------------------------------------------------//
    context_handler(uint64_t device_num)
    : m_data({ { get_data<device_state_tag>()[device_num], nullptr } })
    , m_cleanup([=]() {
        auto& itr = get_data<device_state_tag>()[device_num];
        delete itr;
        itr = nullptr;
        get_data<device_state_tag>().erase(device_num);
    })
    {}

    //----------------------------------------------------------------------------------//
    // callback target device load
    //----------------------------------------------------------------------------------//
    context_handler(uint64_t device_num, const char* filename, int64_t offset_in_file,
                    void* vma_in_file, size_t bytes, void* host_addr, void* device_addr,
                    uint64_t module_id)
    : m_key(mpl::apply<std::string>::join('_', "ompt_target_load", device_num, filename))
    , m_data({ { construct_data(), nullptr } })
    {
        get_data<device_load_tag, uint64_t, data_map_t>()[device_num].insert(
            { module_id, m_data[0] });
        consume_parameters(offset_in_file, vma_in_file, bytes, host_addr, device_addr);
    }

    //----------------------------------------------------------------------------------//
    // callback target device unload
    //----------------------------------------------------------------------------------//
    context_handler(uint64_t device_num, uint64_t module_id)
    : m_data({ { get_data<device_load_tag, uint64_t, data_map_t>()[device_num][module_id],
                 nullptr } })
    , m_cleanup([=]() {
        auto& itr =
            get_data<device_load_tag, uint64_t, data_map_t>()[device_num][module_id];
        delete itr;
        itr = nullptr;
        get_data<device_load_tag, uint64_t, data_map_t>()[device_num].erase(module_id);
    })
    {}

    ~context_handler() { m_cleanup(); }

public:
    static constexpr size_t size = 2;

    TIMEMORY_NODISCARD bool empty() const
    {
        return (m_key.empty() || (m_data[0] == nullptr && m_data[1] == nullptr));
    }

    TIMEMORY_NODISCARD const std::string& key() const { return m_key; }

    TIMEMORY_NODISCARD ompt_data_t* data(size_t idx = 0) const
    {
        return m_data[idx % size];
    }

    template <size_t Idx, typename Tp, typename Func = std::function<void(Tp*)>>
    auto construct(Func&& f = [](Tp*) {})
        -> decltype(new Tp(std::declval<std::string>()), void())
    {
        auto& itr = std::get<Idx>(m_data);
        if(itr && itr->ptr == nullptr)
        {
            auto obj = new Tp(m_key);
            std::forward<Func>(f)(obj);
            itr->ptr = (void*) obj;
        }
    }

    template <typename Tp, typename Func = std::function<void(Tp*)>>
    auto construct(Func&& f = [](Tp*) {})
    {
        construct<0, Tp>(std::forward<Func>(f));
        construct<1, Tp>(std::forward<Func>(f));
    }

    template <size_t Idx, typename Tp, typename Func = std::function<void(Tp*)>>
    auto destroy(Func&& f = [](Tp*) {})
    {
        auto& itr = std::get<Idx>(m_data);
        if(itr && itr->ptr != nullptr)
        {
            auto obj = static_cast<Tp*>(itr->ptr);
            std::forward<Func>(f)(obj);
            delete obj;
            itr->ptr = nullptr;
        }
    }

    template <typename Tp, typename Func = std::function<void(Tp*)>>
    auto destroy(Func&& f = [](Tp*) {})
    {
        destroy<0, Tp>(std::forward<Func>(f));
        destroy<1, Tp>(std::forward<Func>(f));
    }

    auto construct_data(bool _cleanup = false)
    {
        auto _obj = new ompt_data_t{};
        if(_cleanup)
            m_cleanup = [=]() { delete _obj; };
        return _obj;
    }

protected:
    std::string                    m_key = "";
    std::array<ompt_data_t*, size> m_data;
    std::function<void()>          m_cleanup = [] {};

    template <typename Ct, typename At>
    friend struct callback_connector;

    static uint64_t& get_counter()
    {
        static thread_local uint64_t _instance;
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
struct callback_connector
{
    using api_type    = Api;
    using type        = Components;
    using result_type = std::shared_ptr<type>;
    using array_type  = std::deque<result_type>;
    using map_type    = std::unordered_map<size_t, array_type>;
    using handle_type = component::ompt_handle<api_type>;

    static bool is_enabled()
    {
        if(!manager::instance() ||
           (manager::instance() && manager::instance()->is_finalizing()))
        {
            trait::runtime_enabled<type>::set(false);
            trait::runtime_enabled<handle_type>::set(false);
            return false;
        }

        DEBUG_PRINT_HERE("[timemory-ompt]> %s :: handle enabled = %s",
                         demangle<type>().c_str(),
                         (trait::runtime_enabled<handle_type>::get()) ? "y" : "n");

        return (trait::runtime_enabled<handle_type>::get());
    }

    template <typename T, typename... Args,
              enable_if_t<std::is_same<T, mode::begin_callback>::value, int> = 0>
    callback_connector(T, Args... args);

    template <typename T, typename... Args,
              enable_if_t<std::is_same<T, mode::end_callback>::value, int> = 0>
    callback_connector(T, Args... args);

    template <typename T, typename... Args,
              enable_if_t<std::is_same<T, mode::store_callback>::value, int> = 0>
    callback_connector(T, Args... args);

    template <typename T, typename... Args,
              enable_if_t<std::is_same<T, mode::endpoint_callback>::value, int> = 0>
    callback_connector(T, ompt_scope_endpoint_t endp, Args... args);

    template <typename T, typename... Args,
              enable_if_t<std::is_same<T, mode::endpoint_callback>::value, int> = 0>
    callback_connector(T, ompt_work_t workv, ompt_scope_endpoint_t endp, Args... args)
    {
        if(!is_enabled())
            return;
        generic_endpoint_connector(T{}, workv, endp, args...);
    }

    template <typename T, typename... Args,
              enable_if_t<std::is_same<T, mode::endpoint_callback>::value, int> = 0>
    callback_connector(T, ompt_sync_region_t syncv, ompt_scope_endpoint_t endp,
                       Args... args)
    {
        if(!is_enabled())
            return;
        generic_endpoint_connector(T{}, syncv, endp, args...);
    }

    template <typename T, typename... Args,
              enable_if_t<std::is_same<T, mode::endpoint_callback>::value, int> = 0>
    callback_connector(T, ompt_target_t targv, ompt_scope_endpoint_t endp, Args... args)
    {
        if(!is_enabled())
            return;
        generic_endpoint_connector(T{}, targv, endp, args...);
    }

protected:
    template <typename T, typename Arg, typename... Args,
              enable_if_t<std::is_same<T, mode::endpoint_callback>::value, int> = 0>
    void generic_endpoint_connector(T, Arg arg, ompt_scope_endpoint_t endp, Args... args);

private:
    static map_type& get_key_map()
    {
        static thread_local map_type _instance;
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename... Args,
          enable_if_t<std::is_same<T, mode::begin_callback>::value, int>>
callback_connector<Components, Api>::callback_connector(T, Args... args)
{
    if(!is_enabled())
        return;

    context_handler<api_type> ctx(args...);
    user_context_callback(ctx, ctx.m_key, args...);

    // don't provide empty entries
    if(ctx.empty())
        return;

    user_context_callback<type>(ctx, T{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename... Args,
          enable_if_t<std::is_same<T, mode::end_callback>::value, int>>
callback_connector<Components, Api>::callback_connector(T, Args... args)
{
    if(!is_enabled())
        return;

    context_handler<api_type> ctx(args...);
    user_context_callback(ctx, ctx.m_key, args...);

    // don't provide empty entries
    if(ctx.empty())
        return;

    user_context_callback<type>(ctx, T{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename... Args,
          enable_if_t<std::is_same<T, mode::store_callback>::value, int>>
callback_connector<Components, Api>::callback_connector(T, Args... args)
{
    if(!is_enabled())
        return;

    context_handler<api_type> ctx(args...);
    user_context_callback(ctx, ctx.m_key, args...);

    // don't provide empty entries
    if(ctx.empty())
        return;

    user_context_callback<type>(ctx, T{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename... Args,
          enable_if_t<std::is_same<T, mode::endpoint_callback>::value, int>>
callback_connector<Components, Api>::callback_connector(T, ompt_scope_endpoint_t endp,
                                                        Args... args)
{
    if(!is_enabled())
        return;

    context_handler<api_type> ctx(endp, args...);
    user_context_callback(ctx, ctx.m_key, endp, args...);

    // don't provide empty entries
    if(ctx.empty())
        return;

    user_context_callback<type>(ctx, T{}, endp, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename Arg, typename... Args,
          enable_if_t<std::is_same<T, mode::endpoint_callback>::value, int>>
void
callback_connector<Components, Api>::generic_endpoint_connector(
    T, Arg arg, ompt_scope_endpoint_t endp, Args... args)
{
    context_handler<api_type> ctx(arg, endp, args...);
    user_context_callback(ctx, ctx.m_key, arg, endp, args...);

    // don't provide empty entries
    if(ctx.empty())
        return;

    user_context_callback<type>(ctx, T{}, std::forward<Arg>(arg), endp,
                                std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace openmp
//
//--------------------------------------------------------------------------------------//
//
namespace ompt
{
template <typename ApiT>
static void
configure(ompt_function_lookup_t lookup, int, ompt_data_t*)
{
#if defined(TIMEMORY_USE_OMPT)
    //
    //----------------------------------------------------------------------------------//
    //
    using api_type       = ApiT;
    using handle_type    = component::ompt_handle<ApiT>;
    using toolset_type   = typename trait::ompt_handle<api_type>::type;
    using connector_type = openmp::callback_connector<toolset_type, api_type>;
    //
    //----------------------------------------------------------------------------------//
    //
#    define TIMEMORY_OMPT_LOOKUP(TYPE, NAME)                                             \
        static TYPE OMPT_##NAME = (TYPE) lookup(#NAME);                                  \
        consume_parameters(OMPT_##NAME)
    //
    //----------------------------------------------------------------------------------//
    //
    static auto ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
    //
    TIMEMORY_OMPT_LOOKUP(ompt_get_proc_id_t, ompt_get_proc_id);
    TIMEMORY_OMPT_LOOKUP(ompt_get_num_places_t, ompt_get_num_places);
    TIMEMORY_OMPT_LOOKUP(ompt_get_num_devices_t, ompt_get_num_devices);
    TIMEMORY_OMPT_LOOKUP(ompt_get_unique_id_t, ompt_get_unique_id);
    TIMEMORY_OMPT_LOOKUP(ompt_get_place_num_t, ompt_get_place_num);
    TIMEMORY_OMPT_LOOKUP(ompt_get_place_proc_ids_t, ompt_get_place_proc_ids);
    TIMEMORY_OMPT_LOOKUP(ompt_get_target_info_t, ompt_get_target_info);
    TIMEMORY_OMPT_LOOKUP(ompt_get_thread_data_t, ompt_get_thread_data);
    TIMEMORY_OMPT_LOOKUP(ompt_get_record_type_t, ompt_get_record_type);
    TIMEMORY_OMPT_LOOKUP(ompt_get_record_ompt_t, ompt_get_record_ompt);
    TIMEMORY_OMPT_LOOKUP(ompt_get_parallel_info_t, ompt_get_parallel_info);
    TIMEMORY_OMPT_LOOKUP(ompt_get_device_num_procs_t, ompt_get_device_num_procs);
    TIMEMORY_OMPT_LOOKUP(ompt_get_partition_place_nums_t, ompt_get_partition_place_nums);
    //
    // TIMEMORY_OMPT_LOOKUP(ompt_get_device_time_t, ompt_get_device_time);
    // TIMEMORY_OMPT_LOOKUP(ompt_translate_time_t, ompt_translate_time);
    //
    TIMEMORY_OMPT_LOOKUP(ompt_get_task_info_t, ompt_get_task_info);
    TIMEMORY_OMPT_LOOKUP(ompt_get_task_memory_t, ompt_get_task_memory);
    //
    // TIMEMORY_OMPT_LOOKUP(ompt_set_trace_ompt_t, ompt_set_trace_ompt);
    // TIMEMORY_OMPT_LOOKUP(ompt_start_trace_t, ompt_start_trace);
    // TIMEMORY_OMPT_LOOKUP(ompt_pause_trace_t, ompt_pause_trace);
    //
    TIMEMORY_OMPT_LOOKUP(ompt_enumerate_states_t, ompt_enumerate_states);
    TIMEMORY_OMPT_LOOKUP(ompt_enumerate_mutex_impls_t, ompt_enumerate_mutex_impls);
    //
    TIMEMORY_OMPT_LOOKUP(ompt_callback_mutex_t, ompt_callback_mutex);
    TIMEMORY_OMPT_LOOKUP(ompt_callback_nest_lock_t, ompt_callback_nest_lock);
    TIMEMORY_OMPT_LOOKUP(ompt_callback_flush_t, ompt_callback_flush);
    TIMEMORY_OMPT_LOOKUP(ompt_callback_cancel_t, ompt_callback_cancel);
    TIMEMORY_OMPT_LOOKUP(ompt_callback_dispatch_t, ompt_callback_dispatch);
    TIMEMORY_OMPT_LOOKUP(ompt_callback_buffer_request_t, ompt_callback_buffer_request);
    TIMEMORY_OMPT_LOOKUP(ompt_callback_buffer_complete_t, ompt_callback_buffer_complete);
    TIMEMORY_OMPT_LOOKUP(ompt_callback_dependences_t, ompt_callback_dependences);
    TIMEMORY_OMPT_LOOKUP(ompt_callback_task_dependence_t, ompt_callback_task_dependence);
    //
    TIMEMORY_OMPT_LOOKUP(ompt_finalize_tool_t, ompt_finalize_tool);
    //
    //------------------------------------------------------------------------------//
    //
    if(!trait::is_available<handle_type>::value)
        return;

    handle_type::configure();
    auto manager = tim::manager::instance();
    if(manager)
    {
        auto cleanup_label = demangle<handle_type>();
        auto cleanup_func  = []() { trait::runtime_enabled<toolset_type>::set(false); };
        manager->add_cleanup(cleanup_label, cleanup_func);
    }

    auto register_callback = [](ompt_callbacks_t cbidx, ompt_callback_t cb) {
        int ret = ompt_set_callback(cbidx, cb);
        if(settings::verbose() < 1 && !settings::debug())
            return ret;
        auto name = openmp::ompt_callback_labels[cbidx];
        switch(ret)
        {
            case ompt_set_error:
                fprintf(stderr,
                        "[timemory]> WARNING: OMPT Callback for event '%s' count not "
                        "be registered: '%s'\n",
                        name, "ompt_set_error");
                break;
            case ompt_set_never:
                fprintf(stderr,
                        "[timemory]> WARNING: OMPT Callback for event '%s' could not "
                        "be registered: '%s'\n",
                        name, "ompt_set_never");
                break;
            case ompt_set_impossible:
                fprintf(stderr,
                        "[timemory]> WARNING: OMPT Callback for event '%s' could not "
                        "be registered: '%s'\n",
                        name, "ompt_set_impossible");
                break;
            case ompt_set_sometimes:
                fprintf(stderr,
                        "[timemory]> OMPT Callback for event '%s' registered with "
                        "return value: '%s'\n",
                        name, "ompt_set_sometimes");
                break;
            case ompt_set_sometimes_paired:
                fprintf(stderr,
                        "[timemory]> OMPT Callback for event '%s' registered with "
                        "return value: '%s'\n",
                        name, "ompt_set_sometimes_paired");
                break;
            case ompt_set_always:
                fprintf(stderr,
                        "[timemory]> OMPT Callback for event '%s' registered with "
                        "return value: '%s'\n",
                        name, "ompt_set_always");
                break;
        }
        return ret;
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto timemory_ompt_register_callback = [&](ompt_callbacks_t name,
                                               ompt_callback_t  cb) {
        int ret = register_callback(name, cb);
        consume_parameters(ret);
    };
    //
    //----------------------------------------------------------------------------------//
    //
    //      General thread
    //
    //----------------------------------------------------------------------------------//

    using thread_begin_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_thread_t, ompt_data_t*>;

    using thread_end_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*>;

    timemory_ompt_register_callback(ompt_callback_thread_begin,
                                    TIMEMORY_OMPT_CBDECL(thread_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_thread_end,
                                    TIMEMORY_OMPT_CBDECL(thread_end_cb_t::callback));

    using parallel_begin_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, const ompt_frame_t*, ompt_data_t*,
                             unsigned int, int, const void*>;

    using parallel_end_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*, ompt_data_t*, int, const void*>;

    timemory_ompt_register_callback(ompt_callback_parallel_begin,
                                    TIMEMORY_OMPT_CBDECL(parallel_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_parallel_end,
                                    TIMEMORY_OMPT_CBDECL(parallel_end_cb_t::callback));

    using master_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type,
                             openmp::mode::endpoint_callback, ompt_scope_endpoint_t,
                             ompt_data_t*, ompt_data_t*, const void*>;

    timemory_ompt_register_callback(ompt_callback_master,
                                    TIMEMORY_OMPT_CBDECL(master_cb_t::callback));

    //----------------------------------------------------------------------------------//
    //
    //      Tasking section
    //
    //----------------------------------------------------------------------------------//

    using task_create_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_data_t*, const ompt_frame_t*, ompt_data_t*, int, int,
                             const void*>;

    using task_schedule_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_data_t*, ompt_task_status_t, ompt_data_t*>;

    using work_cb_t = openmp::ompt_wrapper<
        toolset_type, connector_type, openmp::mode::endpoint_callback, ompt_work_t,
        ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, uint64_t, const void*>;

    using implicit_task_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type,
                             openmp::mode::endpoint_callback, ompt_scope_endpoint_t,
                             ompt_data_t*, ompt_data_t*, unsigned int, unsigned int>;

    using dispatch_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*, ompt_data_t*, ompt_dispatch_t, ompt_data_t>;

    timemory_ompt_register_callback(ompt_callback_task_create,
                                    TIMEMORY_OMPT_CBDECL(task_create_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_task_schedule,
                                    TIMEMORY_OMPT_CBDECL(task_schedule_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_work,
                                    TIMEMORY_OMPT_CBDECL(work_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_implicit_task,
                                    TIMEMORY_OMPT_CBDECL(implicit_task_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_dispatch,
                                    TIMEMORY_OMPT_CBDECL(dispatch_cb_t::callback));

    /*using task_dependences_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_data_t*, const ompt_dependence_t*, int>;

    using task_dependence_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_data_t*, ompt_data_t*>;

    timemory_ompt_register_callback(
        ompt_callback_dependences, TIMEMORY_OMPT_CBDECL(task_dependences_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_task_dependence,
                                    TIMEMORY_OMPT_CBDECL(task_dependence_cb_t::callback));
    */
    //----------------------------------------------------------------------------------//
    //
    //      Target section
    //
    //----------------------------------------------------------------------------------//

    using target_cb_t = openmp::ompt_wrapper<
        toolset_type, connector_type, openmp::mode::endpoint_callback, ompt_target_t,
        ompt_scope_endpoint_t, int, ompt_data_t*, ompt_id_t, const void*>;

    timemory_ompt_register_callback(ompt_callback_target,
                                    TIMEMORY_OMPT_CBDECL(target_cb_t::callback));

    using target_init_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             uint64_t, const char*, ompt_device_t*,
                             ompt_function_lookup_t, const char*>;

    using target_finalize_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             uint64_t>;

    timemory_ompt_register_callback(ompt_callback_device_initialize,
                                    TIMEMORY_OMPT_CBDECL(target_init_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_device_finalize,
                                    TIMEMORY_OMPT_CBDECL(target_finalize_cb_t::callback));

    using target_load_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             uint64_t, const char*, int64_t, void*, size_t, void*, void*,
                             uint64_t>;

    using target_unload_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             uint64_t, uint64_t>;

    timemory_ompt_register_callback(ompt_callback_device_load,
                                    TIMEMORY_OMPT_CBDECL(target_load_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_device_unload,
                                    TIMEMORY_OMPT_CBDECL(target_unload_cb_t::callback));

    using target_data_op_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_id_t, ompt_id_t, ompt_target_data_op_t, void*, int,
                             void*, int, size_t, const void*>;

    using target_submit_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_id_t, ompt_id_t, unsigned int>;

    using target_mapping_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_id_t, unsigned int, void**, void**, size_t*,
                             unsigned int*>;

    timemory_ompt_register_callback(ompt_callback_target_data_op,
                                    TIMEMORY_OMPT_CBDECL(target_data_op_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target_submit,
                                    TIMEMORY_OMPT_CBDECL(target_submit_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target_map,
                                    TIMEMORY_OMPT_CBDECL(target_mapping_cb_t::callback));

    //----------------------------------------------------------------------------------//
    //
    //      Sync/work section
    //
    //----------------------------------------------------------------------------------//

    using sync_region_cb_t = openmp::ompt_wrapper<
        toolset_type, connector_type, openmp::mode::endpoint_callback, ompt_sync_region_t,
        ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, const void*>;

    timemory_ompt_register_callback(ompt_callback_sync_region,
                                    TIMEMORY_OMPT_CBDECL(sync_region_cb_t::callback));

    using mutex_nest_lock_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type,
                             openmp::mode::endpoint_callback, ompt_scope_endpoint_t,
                             ompt_wait_id_t, const void*>;

    timemory_ompt_register_callback(ompt_callback_nest_lock,
                                    TIMEMORY_OMPT_CBDECL(mutex_nest_lock_cb_t::callback));

    using mutex_acquire_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_mutex_t, unsigned int, unsigned int, ompt_wait_id_t,
                             const void*>;

    timemory_ompt_register_callback(ompt_callback_mutex_acquire,
                                    TIMEMORY_OMPT_CBDECL(mutex_acquire_cb_t::callback));
    // timemory_ompt_register_callback(ompt_callback_reduction,
    //                                TIMEMORY_OMPT_CBDECL(sync_region_cb_t::callback));

    using mutex_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             ompt_mutex_t, ompt_wait_id_t, const void*>;

    timemory_ompt_register_callback(ompt_callback_mutex_acquired,
                                    TIMEMORY_OMPT_CBDECL(mutex_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_mutex_released,
                                    TIMEMORY_OMPT_CBDECL(mutex_cb_t::callback));

    // timemory_ompt_register_callback(ompt_callback_lock_init,
    //                                TIMEMORY_OMPT_CBDECL(mutex_acquire_cb_t::callback));
    // timemory_ompt_register_callback(ompt_callback_lock_destroy,
    //                                TIMEMORY_OMPT_CBDECL(mutex_cb_t::callback));

    //----------------------------------------------------------------------------------//
    //
    //      Miscellaneous section
    //
    //----------------------------------------------------------------------------------//
    /*
    using flush_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_data_t*, const void*>;

    using cancel_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::store_callback,
                             ompt_data_t*, int, const void*>;

    timemory_ompt_register_callback(ompt_callback_flush,
                                    TIMEMORY_OMPT_CBDECL(flush_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_cancel,
                                    TIMEMORY_OMPT_CBDECL(cancel_cb_t::callback));
    */
    if(settings::verbose() > 0 || settings::debug())
        printf("\n");
#else
    consume_parameters(lookup);
#endif
}
}  // namespace ompt
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
