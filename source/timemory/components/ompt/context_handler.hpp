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

#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/context.hpp"
#include "timemory/components/ompt/macros.hpp"
#include "timemory/components/ompt/tool.hpp"
#include "timemory/components/user_bundle.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/utility/type_list.hpp"

#include <cstdlib>

namespace tim
{
namespace openmp
{
//----------------------------------------------------------------------------------//

template <typename ApiT>
template <typename... Tp>
void
context_handler<ApiT>::cleanup(size_t _idx, type_list<Tp...>)
{
    auto _cleanup = [_idx](auto& _v) {
        for(auto& iitr : _v)
        {
            for(auto& itr : iitr)
            {
                if(itr.second)
                {
                    TIMEMORY_PRINTF(stderr, "[ompt] stopping %s on thread %li\n",
                                    itr.second->key().c_str(), _idx);
                    itr.second->stop();
                    delete itr.second;
                    itr.second = nullptr;
                }
            }
        }
    };

    TIMEMORY_FOLD_EXPRESSION(_cleanup(get_map_data<Tp>(_idx)));
}

//----------------------------------------------------------------------------------//

template <typename ApiT>
void
context_handler<ApiT>::cleanup()
{
    auto_lock_t _lk{ type_mutex<context_handler<ApiT>>() };
    for(size_t i = 0; i < max_supported_threads; ++i)
    {
        cleanup(max_supported_threads - i - 1, cleanup_type_list{});
    }
}

//----------------------------------------------------------------------------------//
// callback thread begin
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_thread_t thread_type, ompt_data_t* thread_data)
{
    if(!m_enabled)
        return;
    context_start(ompt_thread_type_labels[thread_type], get_data<thread_tag>(),
                  thread_data, thread_type, thread_data);
}

//----------------------------------------------------------------------------------//
// callback thread end
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* thread_data)
{
    if(!m_enabled)
        return;
    context_stop("ompt_thread_end", get_data<thread_tag>(), thread_data);
}

//----------------------------------------------------------------------------------//
// parallel begin
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                                  ompt_data_t* parallel_data,
                                  unsigned int requested_parallelism, int flags,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;
    context_start(join("", "ompt_parallel [parallelism=", requested_parallelism, ']'),
                  get_data<parallel_tag>(), task_data, task_data, task_frame,
                  parallel_data, requested_parallelism, flags, codeptr);
}

//----------------------------------------------------------------------------------//
// parallel end
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  int flags, const void* codeptr)
{
    if(!m_enabled)
        return;
    context_stop("ompt_parallel_end", get_data<parallel_tag>(), task_data, parallel_data,
                 task_data, flags, codeptr);
}

//----------------------------------------------------------------------------------//
// callback master
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_scope_endpoint_t endpoint,
                                  ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;
    context_endpoint("ompt_master", get_data<master_tag>(), endpoint, task_data, endpoint,
                     parallel_data, task_data, codeptr);
}

//----------------------------------------------------------------------------------//
// callback implicit task
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_scope_endpoint_t endpoint,
                                  ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  unsigned int team_size, unsigned int thread_num)
{
    if(!m_enabled)
        return;
    context_endpoint("ompt_implicit_task", get_data<task_tag>(), endpoint, task_data,
                     endpoint, parallel_data, task_data, team_size, thread_num);
}

//----------------------------------------------------------------------------------//
// callback sync region
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
                                  ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;
    constexpr size_t N = ompt_sync_region_reduction + 1;
    context_endpoint(ompt_sync_region_type_labels[kind],
                     get_data<sync_region_tag, N>(threading::get_id(), kind), endpoint,
                     task_data, kind, endpoint, parallel_data, task_data, codeptr);
}

//----------------------------------------------------------------------------------//
// callback mutex acquire
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_mutex_t kind, unsigned int hint, unsigned int impl,
                                  ompt_wait_id_t wait_id, const void* codeptr)
{
    if(!m_enabled)
        return;
    context_store<bundle_type>(ompt_mutex_type_labels[kind], kind, hint, impl, wait_id,
                               codeptr);
}

//----------------------------------------------------------------------------------//
// callback mutex acquired
// callback mutex released
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_mutex_t kind, ompt_wait_id_t wait_id,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;
    ompt_scope_endpoint_t endpoint;
    switch(m_mode)
    {
        case mode::begin_callback: endpoint = ompt_scope_begin; break;
        case mode::end_callback: endpoint = ompt_scope_end; break;
        default:
        {
            TIMEMORY_PRINTF(stderr,
                            "[ompt] ignoring mutex callback with unknown endpoint\n");
            return;
        }
    };

    ompt_data_t _task_data{};
    _task_data.value = wait_id;
    context_endpoint(ompt_mutex_type_labels[kind], get_data<mutex_tag>(), endpoint,
                     &_task_data, kind, wait_id, codeptr);
}

//----------------------------------------------------------------------------------//
// callback nest lock
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_scope_endpoint_t endpoint, ompt_wait_id_t wait_id,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;
    ompt_data_t _task_data{};
    _task_data.value = wait_id;
    context_endpoint("ompt_nested_lock", get_data<mutex_tag>(), endpoint, &_task_data,
                     endpoint, wait_id, codeptr);
}

//----------------------------------------------------------------------------------//
// callback task create
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                                  ompt_data_t* new_task_data, int flags,
                                  int has_dependences, const void* codeptr)
{
    if(!m_enabled)
        return;
    context_construct("ompt_task", get_data<task_tag>(), new_task_data, task_data,
                      task_frame, new_task_data, flags, has_dependences, codeptr);
}

//----------------------------------------------------------------------------------//
// callback task scheduler
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t*       prior_task_data,
                                  ompt_task_status_t prior_task_status,
                                  ompt_data_t*       next_task_data)
{
    if(!m_enabled)
        return;
    if(prior_task_data)
    {
        switch(prior_task_status)
        {
            case ompt_task_complete:
            case ompt_task_cancel:
            case ompt_task_detach:
                context_stop("ompt_task_schedule", get_data<task_tag>(), prior_task_data,
                             prior_task_data, prior_task_status, next_task_data);
                break;
            case ompt_task_early_fulfill:
            case ompt_task_late_fulfill:
                context_relaxed_stop("ompt_task_schedule", get_data<task_tag>(),
                                     prior_task_data, prior_task_data, prior_task_status,
                                     next_task_data);
            case ompt_task_yield:
            case ompt_task_switch:
            default: break;
        }
    }
    if(next_task_data)
    {
        context_start_constructed("ompt_task_schedule", get_data<task_tag>(),
                                  next_task_data, prior_task_data, prior_task_status,
                                  next_task_data);
    }
}

//----------------------------------------------------------------------------------//
// callback dispatch
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  ompt_dispatch_t kind, ompt_data_t instance)
{
    if(!m_enabled)
        return;
    context_store<bundle_type>(ompt_dispatch_type_labels[kind], parallel_data, task_data,
                               kind, instance);
}

//----------------------------------------------------------------------------------//
// callback work
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_work_t wstype, ompt_scope_endpoint_t endpoint,
                                  ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  uint64_t count, const void* codeptr)
{
    if(!m_enabled)
        return;
    context_endpoint(ompt_work_labels[wstype], get_data<work_tag>(), endpoint, task_data,
                     wstype, endpoint, parallel_data, task_data, count, codeptr);
}

//----------------------------------------------------------------------------------//
// callback flush
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* thread_data, const void* codeptr)
{
    if(!m_enabled)
        return;
    context_store<bundle_type>("ompt_flush", thread_data, codeptr);
}

//----------------------------------------------------------------------------------//
// callback cancel
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* thread_data, int flags,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;
    context_store<bundle_type>("ompt_cancel", thread_data, flags, codeptr);
}

//----------------------------------------------------------------------------------//
// callback target
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_target_t kind, ompt_scope_endpoint_t endpoint,
                                  int device_num, ompt_data_t* task_data,
                                  ompt_id_t target_id, const void* codeptr)
{
    if(!m_enabled)
        return;
    m_key = join('_', ompt_target_type_labels[kind], "dev", device_num);
    context_endpoint(m_key, get_data<target_tag>(), endpoint, task_data, kind, endpoint,
                     device_num, task_data, target_id, codeptr);
}

//----------------------------------------------------------------------------------//
// callback target data op
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_id_t target_id, ompt_id_t host_op_id,
                                  ompt_target_data_op_t optype, void* src_addr,
                                  int src_device_num, void* dest_addr,
                                  int dest_device_num, size_t bytes, const void* codeptr)
{
    if(!m_enabled)
        return;
    m_key = join('_', ompt_target_data_op_labels[optype], "src", src_device_num, "dest",
                 dest_device_num);
    context_store<bundle_type>(m_key, target_id, host_op_id, optype, src_addr,
                               src_device_num, dest_addr, dest_device_num, bytes,
                               codeptr);
}

//----------------------------------------------------------------------------------//
// callback target submit
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_id_t target_id, ompt_id_t host_op_id,
                                  unsigned int requested_num_teams)
{
    if(!m_enabled)
        return;
    context_store<bundle_type>("ompt_target_submit", target_id, host_op_id,
                               requested_num_teams);
}

//----------------------------------------------------------------------------------//
// callback target mapping
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_id_t target_id, unsigned int nitems,
                                  void** host_addr, void** device_addr, size_t* bytes,
                                  unsigned int* mapping_flags)
{
    if(!m_enabled)
        return;
    m_key = join('_', "ompt_target_mapping", target_id);
    context_store<bundle_type>(m_key, target_id, nitems, host_addr, device_addr, bytes,
                               mapping_flags);
}

//----------------------------------------------------------------------------------//
// callback target device initialize
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(uint64_t device_num, const char* type,
                                  ompt_device_t* device, ompt_function_lookup_t lookup,
                                  const char* documentation)
{
    if(!m_enabled)
        return;
    m_key = join('_', "ompt_device", device_num, type);
    context_store<bundle_type>(m_key, device_num, type, device, lookup, documentation);
}

//----------------------------------------------------------------------------------//
// callback target device finalize
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(uint64_t device_num)
{
    if(!m_enabled)
        return;
    context_store<bundle_type>("ompt_device_finalize", device_num);
}

//----------------------------------------------------------------------------------//
// callback target device load
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(uint64_t device_num, const char* filename,
                                  int64_t offset_in_file, void* vma_in_file, size_t bytes,
                                  void* host_addr, void* device_addr, uint64_t module_id)
{
    if(!m_enabled)
        return;
    m_key = join('_', "ompt_target_load", device_num, filename);
    context_store<bundle_type>(m_key, device_num, filename, offset_in_file, vma_in_file,
                               bytes, host_addr, device_addr, module_id);
}

//----------------------------------------------------------------------------------//
// callback target device unload
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(uint64_t device_num, uint64_t module_id)
{
    if(!m_enabled)
        return;
    m_key = join('_', "ompt_target_unload", device_num);
    context_store<bundle_type>(m_key, device_num, module_id);
}
}  // namespace openmp
}  // namespace tim
