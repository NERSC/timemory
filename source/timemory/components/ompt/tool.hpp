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

#include "timemory/backends/threading.hpp"
#include "timemory/components/ompt/macros.hpp"
#include "timemory/components/ompt/types.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/demangle.hpp"
//
#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/components.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/types.hpp"
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
#define TIMEMORY_OMPT_ENUM_LABEL(TYPE)                                                   \
    {                                                                                    \
        TYPE, #TYPE                                                                      \
    }
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_thread_type_labels[] = { nullptr,
                                                 "ompt_thread_initial",
                                                 "ompt_thread_worker",
                                                 "ompt_thread_other",
                                                 "unsupported_ompt_thread_type",
                                                 "unsupported_ompt_thread_type",
                                                 "unsupported_ompt_thread_type",
                                                 "unsupported_ompt_thread_type",
                                                 "unsupported_ompt_thread_type",
                                                 "unsupported_ompt_thread_type" };
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_dispatch_type_labels[] = { nullptr,
                                                   "ompt_dispatch_iteration",
                                                   "ompt_dispatch_section",
                                                   "ompt_dispatch_ws_loop_chunk",
                                                   "ompt_dispatch_taskloop_chunk",
                                                   "ompt_dispatch_distribute_chunk",
                                                   "unsupported_ompt_dispatch_type",
                                                   "unsupported_ompt_dispatch_type",
                                                   "unsupported_ompt_dispatch_type",
                                                   "unsupported_ompt_dispatch_type",
                                                   "unsupported_ompt_dispatch_type",
                                                   "unsupported_ompt_dispatch_type" };
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
    "ompt_sync_region_reduction",
    "ompt_sync_region_barrier_implicit_workshare",
    "ompt_sync_region_barrier_implicit_parallel",
    "ompt_sync_region_barrier_teams",
    "unsupported_ompt_sync_region_type",
    "unsupported_ompt_sync_region_type",
    "unsupported_ompt_sync_region_type",
    "unsupported_ompt_sync_region_type",
    "unsupported_ompt_sync_region_type",
    "unsupported_ompt_sync_region_type"
};
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_target_type_labels[] = { nullptr,
                                                 "ompt_target",
                                                 "ompt_target_enter_data",
                                                 "ompt_target_exit_data",
                                                 "ompt_target_update",
                                                 "ompt_target_nowait",
                                                 "ompt_target_enter_data_nowait",
                                                 "ompt_target_exit_data_nowait",
                                                 "ompt_target_update_nowait",
                                                 "unsupported_ompt_target_type",
                                                 "unsupported_ompt_target_type",
                                                 "unsupported_ompt_target_type",
                                                 "unsupported_ompt_target_type",
                                                 "unsupported_ompt_target_type",
                                                 "unsupported_ompt_target_type" };
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
                                          "ompt_work_taskloop",
                                          "ompt_work_scope",
                                          "ompt_work_loop_static",
                                          "ompt_work_loop_dynamic",
                                          "ompt_work_loop_guided",
                                          "ompt_work_loop_other",
                                          "unsupported_ompt_work_type",
                                          "unsupported_ompt_work_type",
                                          "unsupported_ompt_work_type",
                                          "unsupported_ompt_work_type",
                                          "unsupported_ompt_work_type",
                                          "unsupported_ompt_work_type" };
//
//--------------------------------------------------------------------------------------//
//
static const char* ompt_target_data_op_labels[] = {
    nullptr,
    "ompt_target_data_alloc",
    "ompt_target_data_transfer_to_dev",
    "ompt_target_data_transfer_from_dev",
    "ompt_target_data_delete",
    "unsupported_ompt_target_data_op_type",
    "unsupported_ompt_target_data_op_type",
    "unsupported_ompt_target_data_op_type",
    "unsupported_ompt_target_data_op_type",
    "unsupported_ompt_target_data_op_type",
    "unsupported_ompt_target_data_op_type"
};
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
                                                 "ompt_task_switch",
                                                 "ompt_taskwait_complete",
                                                 "unsupported_ompt_task_status_type",
                                                 "unsupported_ompt_task_status_type",
                                                 "unsupported_ompt_task_status_type",
                                                 "unsupported_ompt_task_status_type",
                                                 "unsupported_ompt_task_status_type",
                                                 "unsupported_ompt_task_status_type" };
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_mutex_t, const char*> ompt_mutex_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_lock),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_test_lock),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_nest_lock),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_test_nest_lock),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_critical),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_atomic),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_ordered)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_task_flag_t, const char*> ompt_task_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_initial),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_implicit),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_explicit),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_target),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_taskwait),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_undeferred),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_untied),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_final),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_mergeable),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_task_merged)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_target_map_flag_t, const char*> ompt_target_map_labels = {
    TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_to),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_from),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_alloc),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_release),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_delete),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_implicit)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_dependence_type_t, const char*> ompt_dependence_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_in),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_out),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_inout),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_mutexinoutset),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_source),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_sink),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_inoutset),
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
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_dispatch),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_emi),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_data_op_emi),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_submit_emi),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_map_emi),
    TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_error)
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
template <typename ApiT>
struct context_handler
{
    using api_type    = ApiT;
    using this_type   = context_handler<api_type>;
    using bundle_type = typename trait::ompt_handle<api_type>::type;
    static constexpr size_t max_supported_threads =
        trait::max_threads<ApiT, context_handler<ApiT>>::value;

public:
    template <typename KeyT, typename MappedT, typename HashT = std::hash<KeyT>>
    using uomap_t = std::unordered_map<KeyT, MappedT, HashT>;

    template <typename Tag, size_t N = 1>
    static auto& get_map_data(int64_t _tid = threading::get_id())
    {
        using map_type = std::array<uomap_t<uint64_t, bundle_type*>, N>;
        static std::array<map_type, max_supported_threads> _v = {};
        return _v.at(_tid % max_supported_threads);
    }

    template <typename Tag, size_t N = 1>
    static auto& get_data(int64_t _tid = threading::get_id(), size_t _n = 0)
    {
        return get_map_data<Tag, N>(_tid).at(_n % N);
    }

    // tags for above
    struct task_tag
    {};
    struct mutex_tag
    {};
    struct sync_region_tag
    {};
    struct master_tag
    {};
    struct target_tag
    {};
    struct thread_tag
    {};
    struct parallel_tag
    {};
    struct work_tag
    {};
    struct task_create_tag
    {};

    using cleanup_type_list =
        type_list<task_tag, mutex_tag, sync_region_tag, target_tag, thread_tag,
                  parallel_tag, work_tag, master_tag, task_create_tag>;

    template <typename... Args>
    static auto join(Args&&... _args)
    {
        return mpl::apply<std::string>::join(std::forward<Args>(_args)...);
    }

public:
    template <typename... Tp>
    static void cleanup(size_t _idx, type_list<Tp...>);
    static void cleanup();

    TIMEMORY_DEFAULT_OBJECT(context_handler)

    explicit context_handler(mode _v)
    : m_mode{ _v }
    {}

    //----------------------------------------------------------------------------------//
    // callback thread begin
    //----------------------------------------------------------------------------------//
    void operator()(ompt_thread_t thread_type, ompt_data_t* thread_data);

    //----------------------------------------------------------------------------------//
    // callback thread end
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* thread_data);

    //----------------------------------------------------------------------------------//
    // parallel begin
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                    ompt_data_t* parallel_data, unsigned int requested_parallelism,
                    int flags, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // parallel end
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* parallel_data, ompt_data_t* task_data, int flags,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback master
    //----------------------------------------------------------------------------------//
    void operator()(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback implicit task
    //----------------------------------------------------------------------------------//
    void operator()(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, unsigned int team_size,
                    unsigned int thread_num);

    //----------------------------------------------------------------------------------//
    // callback sync region
    //----------------------------------------------------------------------------------//
    void operator()(ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback mutex acquire
    //----------------------------------------------------------------------------------//
    void operator()(ompt_mutex_t kind, unsigned int hint, unsigned int impl,
                    ompt_wait_id_t wait_id, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback mutex acquired
    // callback mutex released
    //----------------------------------------------------------------------------------//
    void operator()(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback nest lock
    //----------------------------------------------------------------------------------//
    void operator()(ompt_scope_endpoint_t endpoint, ompt_wait_id_t wait_id,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback task create
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                    ompt_data_t* new_task_data, int flags, int has_dependences,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback task scheduler
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* prior_task_data, ompt_task_status_t prior_task_status,
                    ompt_data_t* next_task_data);

    //----------------------------------------------------------------------------------//
    // callback dispatch
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* parallel_data, ompt_data_t* task_data,
                    ompt_dispatch_t kind, ompt_data_t instance);

    //----------------------------------------------------------------------------------//
    // callback work
    //----------------------------------------------------------------------------------//
    void operator()(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data, uint64_t count,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback flush
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* thread_data, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback cancel
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* thread_data, int flags, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback target
    //----------------------------------------------------------------------------------//
    void operator()(ompt_target_t kind, ompt_scope_endpoint_t endpoint, int device_num,
                    ompt_data_t* task_data, ompt_id_t target_id, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback target data op
    //----------------------------------------------------------------------------------//
    void operator()(ompt_id_t target_id, ompt_id_t host_op_id,
                    ompt_target_data_op_t optype, void* src_addr, int src_device_num,
                    void* dest_addr, int dest_device_num, size_t bytes,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback target submit
    //----------------------------------------------------------------------------------//
    void operator()(ompt_id_t target_id, ompt_id_t host_op_id,
                    unsigned int requested_num_teams);

    //----------------------------------------------------------------------------------//
    // callback target mapping
    //----------------------------------------------------------------------------------//
    void operator()(ompt_id_t target_id, unsigned int nitems, void** host_addr,
                    void** device_addr, size_t* bytes, unsigned int* mapping_flags);

    //----------------------------------------------------------------------------------//
    // callback target device initialize
    //----------------------------------------------------------------------------------//
    void operator()(uint64_t device_num, const char* type, ompt_device_t* device,
                    ompt_function_lookup_t lookup, const char* documentation);

    //----------------------------------------------------------------------------------//
    // callback target device finalize
    //----------------------------------------------------------------------------------//
    void operator()(uint64_t device_num);

    //----------------------------------------------------------------------------------//
    // callback target device load
    //----------------------------------------------------------------------------------//
    void operator()(uint64_t device_num, const char* filename, int64_t offset_in_file,
                    void* vma_in_file, size_t bytes, void* host_addr, void* device_addr,
                    uint64_t module_id);

    //----------------------------------------------------------------------------------//
    // callback target device unload
    //----------------------------------------------------------------------------------//
    void operator()(uint64_t device_num, uint64_t module_id);

public:
    const std::string& key() const { return m_key; }
    void               set_mode(mode _v) { m_mode = _v; }

    friend std::ostream& operator<<(std::ostream& _os, const context_handler& _v)
    {
        _os << _v.m_key;
        return _os;
    }

protected:
    bool        m_enabled = trait::runtime_enabled<this_type>::get();
    mode        m_mode;
    std::string m_key = {};

    template <typename Ct, typename At>
    friend struct callback_connector;
};
}  // namespace openmp
//
//--------------------------------------------------------------------------------------//
//
namespace ompt
{
template <typename ApiT>
finalize_tool_func_t
configure(ompt_function_lookup_t lookup, int, ompt_data_t*);
}  // namespace ompt
}  // namespace tim
