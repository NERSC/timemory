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

#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
//
#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/components.hpp"
//
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
// static const char* ompt_task_status_labels[] = { nullptr, "ompt_task_complete",
//                                                 "ompt_task_yield", "ompt_task_cancel",
//                                                 "ompt_task_others" };
//
//--------------------------------------------------------------------------------------//
//
// static const char* ompt_cancel_flag_labels[] = {
//    "ompt_cancel_parallel",      "ompt_cancel_sections",  "ompt_cancel_do",
//    "ompt_cancel_taskgroup",     "ompt_cancel_activated", "ompt_cancel_detected",
//    "ompt_cancel_discarded_task"
// };
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
static std::map<ompt_mutex_kind_t, const char*> ompt_mutex_labels = {
    { ompt_mutex, "ompt_mutex" },
    { ompt_mutex_lock, "ompt_mutex_lock" },
    { ompt_mutex_nest_lock, "ompt_mutex_nest_lock" },
    { ompt_mutex_critical, "ompt_mutex_critical" },
    { ompt_mutex_atomic, "ompt_mutex_atomic" },
    { ompt_mutex_ordered, "ompt_mutex_ordered" }
};
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_sync_region_labels[] = { nullptr, "ompt_sync_region_barrier",
                                                 "ompt_sync_region_taskwait",
                                                 "ompt_sync_region_taskgroup" };
//
//--------------------------------------------------------------------------------------//
//
template <typename Api>
struct context_handler
{
    using api_type = Api;

    template <typename T, enable_if_t<(std::is_integral<T>::value), int> = 0>
    static auto get_hash(T val)
    {
        return std::hash<size_t>()(val);
    }

    template <typename T, enable_if_t<(std::is_pointer<T>::value), long> = 0>
    static auto get_hash(const T ptr)
    {
        return std::hash<const void*>()(static_cast<const void*>(ptr));
    }

    template <
        typename T,
        enable_if_t<!(std::is_pointer<T>::value || std::is_integral<T>::value), int> = 0>
    static auto get_hash(T val)
    {
        return std::hash<T>()(val);
    }

    static auto get_hash(const std::string& key) { return std::hash<std::string>()(key); }

    static auto get_hash(const char* key) { return std::hash<std::string>()(key); }

    static auto get_hash(char* key) { return std::hash<std::string>()(key); }

#if defined(OMPT_DEBUG)
    static void insert(const void* codeptr, const std::string& key)
    {
        if(!tim::settings::debug() && settings::verbose() < 1)
            return;

        using codeptr_map_t = std::map<const void*, std::vector<std::string>>;
        static codeptr_map_t codeptr_map;

        auto_lock_t lk(type_mutex<context_handler<Api>>());
        codeptr_map[codeptr].push_back(key);
        std::stringstream ss;
        ss << "\n\n";
        for(const auto& itr : codeptr_map)
        {
            ss << std::setw(16) << itr.first << "  ::   ";
            for(const auto& eitr : itr.second)
                ss << std::setw(20) << eitr << "  ";
            ss << '\n';
        }
        std::cout << ss.str();
    }
#endif

public:
    //----------------------------------------------------------------------------------//
    // parallel begin
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parent_task_data, const omp_frame_t* parent_task_frame,
                    ompt_data_t* parallel_data, uint32_t requested_team_size,
                    const void* codeptr)
    {
        m_key = "ompt_parallel";
        m_id  = get_hash(m_key) + get_hash(parent_task_data);

        consume_parameters(parent_task_data, parent_task_frame, parallel_data,
                           requested_team_size, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // parallel end
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parallel_data, ompt_data_t* parent_task_data,
                    const void* codeptr)
    {
        m_key = "ompt_parallel";
        m_id  = get_hash(m_key) + get_hash(parent_task_data);

        consume_parameters(parallel_data, parent_task_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // task create
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* parent_task_data, const omp_frame_t* parent_frame,
                    ompt_data_t* new_task_data, int type, int has_dependences,
                    const void* codeptr)
    {
        // insert(codeptr, "task_create" + std::to_string(type));
        // insert(parent_task_data, "task_create_task_data" + std::to_string(type));
        // insert(parent_frame, "task_create_parent_frame" + std::to_string(type));
        // insert(new_task_data, "task_create_new_data" + std::to_string(type));

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
        m_id  = get_hash(m_key) + get_hash(task_data);

        consume_parameters(endpoint, parallel_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback work
    //----------------------------------------------------------------------------------//
    context_handler(ompt_work_type_t wstype, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data, uint64_t count,
                    const void* codeptr)
    {
        m_key = ompt_work_labels[wstype];
        m_id  = get_hash(m_key) + get_hash(task_data);

        consume_parameters(endpoint, parallel_data, count, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback thread begin
    //----------------------------------------------------------------------------------//
    context_handler(ompt_thread_type_t thread_type, ompt_data_t* thread_data)
    {
        m_key = ompt_thread_type_labels[thread_type];
        m_id  = get_hash(thread_data);
    }

    //----------------------------------------------------------------------------------//
    // callback thread end
    //----------------------------------------------------------------------------------//
    context_handler(ompt_data_t* thread_data) { m_id = get_hash(thread_data); }

    //----------------------------------------------------------------------------------//
    // callback implicit task
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, unsigned int team_size,
                    unsigned int thread_num)
    {
        // m_key = apply<std::string>::join("_", "ompt_implicit_task", thread_num, "of",
        //                                 team_size);
        m_key = "ompt_implicit_task";
        m_id  = get_hash(m_key);

        consume_parameters(endpoint, parallel_data, task_data, team_size, thread_num);
    }

    //----------------------------------------------------------------------------------//
    // callback sync region
    //----------------------------------------------------------------------------------//
    context_handler(ompt_sync_region_kind_t kind, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data,
                    const void* codeptr)
    {
        m_key = ompt_sync_region_labels[kind];
        m_id  = get_hash(m_key);

        consume_parameters(endpoint, parallel_data, task_data, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback idle
    //----------------------------------------------------------------------------------//
    context_handler(ompt_scope_endpoint_t endpoint)
    {
        m_key = "ompt_idle";
        m_id  = get_hash(m_key);

        consume_parameters(endpoint);
    }

    //----------------------------------------------------------------------------------//
    // callback mutex acquire
    //----------------------------------------------------------------------------------//
    context_handler(ompt_mutex_kind_t kind, unsigned int hint, unsigned int impl,
                    omp_wait_id_t wait_id, const void* codeptr)
    {
        m_key = ompt_mutex_labels[kind];
        m_id  = get_hash(m_key) ^ get_hash(wait_id);

        consume_parameters(hint, impl, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback mutex acquired
    // callback mutex released
    //----------------------------------------------------------------------------------//
    context_handler(ompt_mutex_kind_t kind, omp_wait_id_t wait_id, const void* codeptr)
    {
        m_key = ompt_mutex_labels[kind];
        m_id  = get_hash(m_key) ^ get_hash(wait_id);

        consume_parameters(codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback target
    //----------------------------------------------------------------------------------//
    context_handler(ompt_target_type_t kind, ompt_scope_endpoint_t endpoint,
                    int device_num, ompt_data_t* task_data, ompt_id_t target_id,
                    const void* codeptr)
    {
        m_key = apply<std::string>::join("_", ompt_target_type_labels[kind], "dev",
                                         device_num);
        m_id  = get_hash(m_key) + get_hash(task_data);

        consume_parameters(kind, endpoint, task_data, target_id, codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback target data op
    //----------------------------------------------------------------------------------//
    context_handler(ompt_id_t target_id, ompt_id_t host_op_id,
                    ompt_target_data_op_t optype, void* src_addr, int src_device_num,
                    void* dest_addr, int dest_device_num, size_t bytes,
                    const void* codeptr)
    {
        m_key = apply<std::string>::join("_", ompt_target_data_op_labels[optype], "src",
                                         src_device_num, "dest", dest_device_num);
        m_id  = get_hash(m_key);

        consume_parameters(target_id, host_op_id, optype, src_addr, dest_addr, bytes,
                           codeptr);
    }

    //----------------------------------------------------------------------------------//
    // callback target submit
    //----------------------------------------------------------------------------------//
    context_handler(ompt_id_t target_id, ompt_id_t host_op_id,
                    unsigned int requested_num_teams)
    {
        m_key = "ompt_target_submit";
        m_id  = get_hash(target_id) + get_hash(host_op_id);

        consume_parameters(target_id, host_op_id, requested_num_teams);
    }

    //----------------------------------------------------------------------------------//
    // callback target mapping
    //----------------------------------------------------------------------------------//
    context_handler(ompt_id_t target_id, unsigned int nitems, void** host_addr,
                    void** device_addr, size_t* bytes, unsigned int* mapping_flags)
    {
        consume_parameters(target_id, nitems, host_addr, device_addr, bytes,
                           mapping_flags);
    }

    //----------------------------------------------------------------------------------//
    // callback target device initialize
    //----------------------------------------------------------------------------------//
    context_handler(uint64_t device_num, const char* type, ompt_device_t* device,
                    ompt_function_lookup_t lookup, const char* documentation)
    {
        consume_parameters(device_num, type, device, lookup, documentation);
    }

    //----------------------------------------------------------------------------------//
    // callback target device finalize
    //----------------------------------------------------------------------------------//
    context_handler(uint64_t device_num) { consume_parameters(device_num); }

    //----------------------------------------------------------------------------------//
    // callback target device load
    //----------------------------------------------------------------------------------//
    context_handler(uint64_t device_num, const char* filename, int64_t offset_in_file,
                    void* vma_in_file, size_t bytes, void* host_addr, void* device_addr,
                    uint64_t module_id)
    {
        consume_parameters(device_num, filename, offset_in_file, vma_in_file, bytes,
                           host_addr, device_addr, module_id);
    }

    //----------------------------------------------------------------------------------//
    // callback target device unload
    //----------------------------------------------------------------------------------//
    context_handler(uint64_t device_num, uint64_t module_id)
    {
        consume_parameters(device_num, module_id);
    }

public:
    size_t             id() const { return m_id; }
    const std::string& key() const { return m_key; }

protected:
    // the first hash is the string hash, the second is the data hash
    size_t      m_id;
    std::string m_key;

    template <typename Ct, typename At>
    friend struct callback_connector;
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
            trait::runtime_enabled<type>::set(false);

        return (trait::runtime_enabled<type>::get() &&
                trait::runtime_enabled<handle_type>::get());
    }

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::begin_callback>::value), int> = 0>
    callback_connector(T, Args... args);

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::end_callback>::value), int> = 0>
    callback_connector(T, Args... args);

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::measure_callback>::value), int> = 0>
    callback_connector(T, Args... args);

    template <typename T, typename... Args,
              enable_if_t<(std::is_same<T, mode::endpoint_callback>::value), int> = 0>
    callback_connector(T, ompt_scope_endpoint_t endp, Args... args);

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
          enable_if_t<(std::is_same<T, mode::begin_callback>::value), int>>
callback_connector<Components, Api>::callback_connector(T, Args... args)
{
    if(!is_enabled())
        return;

    context_handler<api_type> ctx(args...);
    user_context_callback(ctx, ctx.m_id, ctx.m_key, args...);

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
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename... Args,
          enable_if_t<(std::is_same<T, mode::end_callback>::value), int>>
callback_connector<Components, Api>::callback_connector(T, Args... args)
{
    if(!is_enabled())
        return;

    context_handler<api_type> ctx(args...);
    user_context_callback(ctx, ctx.m_id, ctx.m_key, args...);

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
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename... Args,
          enable_if_t<(std::is_same<T, mode::measure_callback>::value), int>>
callback_connector<Components, Api>::callback_connector(T, Args... args)
{
    if(!is_enabled())
        return;

    context_handler<api_type> ctx(args...);
    user_context_callback(ctx, ctx.m_id, ctx.m_key, args...);

    // don't provide empty entries
    if(ctx.key().empty())
        return;

    auto c = std::make_shared<type>(ctx.key());

    c->construct(args...);
    c->audit(ctx.key(), args...);
    c->measure();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename... Args,
          enable_if_t<(std::is_same<T, mode::endpoint_callback>::value), int>>
callback_connector<Components, Api>::callback_connector(T, ompt_scope_endpoint_t endp,
                                                        Args... args)
{
    if(!is_enabled())
        return;

    context_handler<api_type> ctx(endp, args...);
    user_context_callback(ctx, ctx.m_id, ctx.m_key, endp, args...);

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
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename Api>
template <typename T, typename Arg, typename... Args,
          enable_if_t<(std::is_same<T, mode::endpoint_callback>::value), int>>
void
callback_connector<Components, Api>::generic_endpoint_connector(
    T, Arg arg, ompt_scope_endpoint_t endp, Args... args)
{
    context_handler<api_type> ctx(arg, endp, args...);
    user_context_callback(ctx, ctx.m_id, ctx.m_key, arg, endp, args...);

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
configure(ompt_function_lookup_t lookup, ompt_data_t*)
{
#if defined(TIMEMORY_USE_OMPT)
    using api_type       = ApiT;
    using handle_type    = component::ompt_handle<ApiT>;
    using toolset_type   = typename trait::ompt_handle<api_type>::type;
    using connector_type = openmp::callback_connector<toolset_type, api_type>;
    //
    //------------------------------------------------------------------------------//
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

    auto register_callback = [](ompt_callbacks_t name, ompt_callback_t cb) {
        int ret = ompt_set_callback(name, cb);
        if(settings::verbose() < 1 && !settings::debug())
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
        consume_parameters(ret);
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

    using parallel_begin_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, const omp_frame_t*, ompt_data_t*, uint32_t,
                             const void*>;

    using parallel_end_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*, ompt_data_t*, const void*>;

    using task_create_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, const omp_frame_t*, ompt_data_t*, int, int,
                             const void*>;

    using task_schedule_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_data_t*, ompt_task_status_t, ompt_data_t*>;

    using master_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type,
                             openmp::mode::endpoint_callback, ompt_scope_endpoint_t,
                             ompt_data_t*, ompt_data_t*, const void*>;

    using work_cb_t = openmp::ompt_wrapper<
        toolset_type, connector_type, openmp::mode::endpoint_callback, ompt_work_type_t,
        ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, uint64_t, const void*>;

    using thread_begin_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_thread_type_t, ompt_data_t*>;

    using thread_end_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             ompt_data_t*>;

    using implicit_task_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type,
                             openmp::mode::endpoint_callback, ompt_scope_endpoint_t,
                             ompt_data_t*, ompt_data_t*, unsigned int, unsigned int>;

    using sync_region_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type,
                             openmp::mode::endpoint_callback, ompt_sync_region_kind_t,
                             ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*,
                             const void*>;

    using mutex_acquire_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_mutex_kind_t, unsigned int, unsigned int, omp_wait_id_t,
                             const void*>;

    using mutex_acquired_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::end_callback,
                             ompt_mutex_kind_t, omp_wait_id_t, const void*>;

    using mutex_released_cb_t = mutex_acquired_cb_t;

    using target_cb_t = openmp::ompt_wrapper<
        toolset_type, connector_type, openmp::mode::endpoint_callback, ompt_target_type_t,
        ompt_scope_endpoint_t, int, ompt_data_t*, ompt_id_t, const void*>;

    using target_data_op_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_id_t, ompt_id_t, ompt_target_data_op_t, void*, int,
                             void*, int, size_t, const void*>;

    using target_submit_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type, openmp::mode::begin_callback,
                             ompt_id_t, ompt_id_t, unsigned int>;

    using target_idle_cb_t =
        openmp::ompt_wrapper<toolset_type, connector_type,
                             openmp::mode::endpoint_callback, ompt_scope_endpoint_t>;

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
    timemory_ompt_register_callback(ompt_callback_idle,
                                    TIMEMORY_OMPT_CBDECL(target_idle_cb_t::callback));

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

    consume_parameters(ompt_set_callback, ompt_get_task_info, ompt_get_unique_id,
                       ompt_get_thread_data, ompt_get_parallel_info, ompt_get_num_places,
                       ompt_get_place_proc_ids, ompt_get_place_num,
                       ompt_get_partition_place_nums, ompt_get_proc_id,
                       ompt_enumerate_states, ompt_enumerate_mutex_impls);

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
