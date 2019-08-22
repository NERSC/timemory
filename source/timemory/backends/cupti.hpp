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
 * \headerfile cupti.hpp "timemory/cupti.hpp"
 * Provides implementation of CUPTI routines.
 *
 */

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/device.hpp"
#include "timemory/details/cupti.hpp"
#include "timemory/details/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

//--------------------------------------------------------------------------------------//

#define CUDA_DRIVER_API_CALL(apiFuncCall)                                                \
    {                                                                                    \
        CUresult _status = apiFuncCall;                                                  \
        if(_status != CUDA_SUCCESS)                                                      \
        {                                                                                \
            fprintf(stderr, "%s:%d: error: function '%s' failed with error: %d.\n",      \
                    __FILE__, __LINE__, #apiFuncCall, _status);                          \
        }                                                                                \
    }

//--------------------------------------------------------------------------------------//

#define CUPTI_CALL(call)                                                                 \
    {                                                                                    \
        CUptiResult _status = call;                                                      \
        if(_status != CUPTI_SUCCESS)                                                     \
        {                                                                                \
            const char* errstr;                                                          \
            cuptiGetResultString(_status, &errstr);                                      \
            fprintf(stderr, "%s:%d: error: function '%s' failed with error: %s.\n",      \
                    __FILE__, __LINE__, #call, errstr);                                  \
        }                                                                                \
    }

//--------------------------------------------------------------------------------------//

#define CUPTI_BUFFER_SIZE (32 * 1024)
#define CUPTI_ALIGN_SIZE (8)
#define CUPTI_ALIGN_BUFFER(buffer, align)                                                \
    (((uintptr_t)(buffer) & ((align) -1))                                                \
         ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align) -1)))                   \
         : (buffer))

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace cupti
{
//--------------------------------------------------------------------------------------//

using string_t = std::string;
template <typename _Key, typename _Mapped>
using map_t    = std::map<_Key, _Mapped>;
using strvec_t = std::vector<string_t>;

//--------------------------------------------------------------------------------------//
/*
using stream_duration_t     = map_t<uint32_t, uint64_t>;
using stream_duration_ptr_t = std::unique_ptr<stream_duration_t>;

//--------------------------------------------------------------------------------------//

static stream_duration_ptr_t&
get_stream_kernel_duration()
{
    static thread_local stream_duration_ptr_t _instance =
        stream_duration_ptr_t(new stream_duration_t);
    return _instance;
}

//--------------------------------------------------------------------------------------//

static void CUPTIAPI
            buffer_requested(uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
    uint8_t* rawBuffer;

    *size          = 16 * 1024;
    rawBuffer      = (uint8_t*) malloc(*size + CUPTI_ALIGN_SIZE);
    *buffer        = CUPTI_ALIGN_BUFFER(rawBuffer, CUPTI_ALIGN_SIZE);
    *maxNumRecords = 0;
    if(*buffer == nullptr)
    {
        throw std::runtime_error("Error: out of memory\n");
    }
}

//--------------------------------------------------------------------------------------//

static void CUPTIAPI
            buffer_completed(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                             size_t size , size_t validSize)
{
    consume_parameters(ctx, size);
    CUpti_Activity* record = nullptr;

    // since we launched only 1 kernel, we should have only 1 kernel record
    CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

    CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*) record;
    if(kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL)
    {
        std::stringstream ss;
        ss << "Error: expected kernel activity record, got "
           << static_cast<int>(kernel->kind);
        throw std::runtime_error(os.str());
    }

    (*get_stream_kernel_duration())[streamId] = kernel->end - kernel->start;

    free(buffer);
}
*/

//--------------------------------------------------------------------------------------//

namespace impl
{
//--------------------------------------------------------------------------------------//

static uint64_t dummy_kernel_id = 0;

//--------------------------------------------------------------------------------------//

template <typename _Tp>
GLOBAL_CALLABLE void
warmup()
{
}

//--------------------------------------------------------------------------------------//

static data_metric_t
get_metric(CUpti_MetricID& id, CUpti_MetricValue& value)
{
    CUpti_MetricValueKind value_kind;
    size_t                value_kind_sz = sizeof(value_kind);
    CUPTI_CALL(cuptiMetricGetAttribute(id, CUPTI_METRIC_ATTR_VALUE_KIND, &value_kind_sz,
                                       &value_kind));
    data_metric_t ret;
    switch(value_kind)
    {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE: data::floating::set(ret, value); break;
        case CUPTI_METRIC_VALUE_KIND_UINT64:
            data::unsigned_integer::set(ret, value);
            break;
        case CUPTI_METRIC_VALUE_KIND_INT64: data::integer::set(ret, value); break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT: data::percent::set(ret, value); break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT: data::throughput::set(ret, value); break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            data::utilization::set(ret, value);
            break;
        default: break;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

static void
print_metric(std::ostream& os, CUpti_MetricID& id, CUpti_MetricValue& value)
{
    CUpti_MetricValueKind value_kind;
    size_t                value_kind_sz = sizeof(value_kind);
    CUPTI_CALL(cuptiMetricGetAttribute(id, CUPTI_METRIC_ATTR_VALUE_KIND, &value_kind_sz,
                                       &value_kind));
    switch(value_kind)
    {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE: os << value.metricValueDouble; break;
        case CUPTI_METRIC_VALUE_KIND_UINT64: os << value.metricValueUint64; break;
        case CUPTI_METRIC_VALUE_KIND_INT64: os << value.metricValueInt64; break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT: os << value.metricValuePercent; break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT: os << value.metricValueThroughput; break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            os << value.metricValueUtilizationLevel;
            break;
        default: std::cerr << "[error]: unknown value kind\n"; break;
    }
}

//--------------------------------------------------------------------------------------//
// Pass-specific data
//
struct pass_data_t
{
    // the set of event groups to collect for a pass
    CUpti_EventGroupSet* event_groups;
    // the number of entries in eventIdArray and eventValueArray
    uint32_t num_events;
    // array of event ids
    std::vector<CUpti_EventID> event_ids;
    // array of event values
    std::vector<uint64_t> event_values;
};

//--------------------------------------------------------------------------------------//
// data for the kernels
//
struct kernel_data_t
{
    using event_val_t  = std::vector<uint64_t>;
    using metric_val_t = std::vector<CUpti_MetricValue>;
    using metric_tup_t = std::vector<data_metric_t>;
    using pass_val_t   = std::vector<pass_data_t>;

    kernel_data_t()                     = default;
    kernel_data_t(const kernel_data_t&) = default;
    kernel_data_t(kernel_data_t&&)      = default;
    kernel_data_t& operator=(const kernel_data_t&) = default;
    kernel_data_t& operator=(kernel_data_t&&) = default;

    CUdevice m_device;
    int      m_metric_passes = 0;
    int      m_event_passes  = 0;
    int      m_current_pass  = 0;
    int      m_total_passes  = 0;
    string_t m_name          = "";

    pass_val_t   m_pass_data;
    event_val_t  m_event_values;
    metric_val_t m_metric_values;
    metric_tup_t m_metric_tuples;

    void clone(const kernel_data_t& rhs)
    {
        m_device        = rhs.m_device;
        m_metric_passes = rhs.m_metric_passes;
        m_event_passes  = rhs.m_event_passes;
        m_current_pass  = rhs.m_current_pass;
        m_total_passes  = rhs.m_total_passes;
        m_name          = rhs.m_name;

        m_pass_data     = rhs.m_pass_data;
        m_metric_values = rhs.m_metric_values;
        m_event_values.resize(rhs.m_event_values.size(), 0);
        m_metric_tuples.resize(rhs.m_metric_tuples.size(), data_metric_t());
    }

    kernel_data_t& operator+=(const kernel_data_t& rhs)
    {
        m_event_values.resize(rhs.m_event_values.size(), 0);
        for(uint64_t i = 0; i < rhs.m_event_values.size(); ++i)
            m_event_values[i] += rhs.m_event_values[i];

        m_metric_tuples.resize(rhs.m_metric_tuples.size(), data_metric_t());
        for(uint64_t i = 0; i < rhs.m_metric_tuples.size(); ++i)
            m_metric_tuples[i] += rhs.m_metric_tuples[i];

        return *this;
    }

    kernel_data_t& operator-=(const kernel_data_t& rhs)
    {
        m_event_values.resize(rhs.m_event_values.size(), 0);
        for(uint64_t i = 0; i < rhs.m_event_values.size(); ++i)
            m_event_values[i] -= rhs.m_event_values[i];

        m_metric_tuples.resize(rhs.m_metric_tuples.size(), data_metric_t());
        for(uint64_t i = 0; i < rhs.m_metric_tuples.size(); ++i)
            m_metric_tuples[i] -= rhs.m_metric_tuples[i];

        return *this;
    }

    friend kernel_data_t operator+(const kernel_data_t& lhs, const kernel_data_t& rhs)
    {
        return kernel_data_t(lhs) += rhs;
    }

    friend kernel_data_t operator-(const kernel_data_t& lhs, const kernel_data_t& rhs)
    {
        return kernel_data_t(lhs) -= rhs;
    }
};

//--------------------------------------------------------------------------------------//
// CUPTI subscriber
//
static void CUPTIAPI
            get_value_callback(void* userdata, CUpti_CallbackDomain /*domain*/, CUpti_CallbackId cbid,
                               const CUpti_CallbackData* cbInfo)
{
    using map_type = map_t<uint64_t, kernel_data_t>;
    static std::atomic<uint64_t> correlation_data;

    // This callback is enabled only for launch so we shouldn't see anything else.
    if((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
       (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) &&
       (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000) &&
       (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000))
    {
        char buf[512];
        sprintf(buf, "%s:%d: Unexpected cbid %d\n", __FILE__, __LINE__, cbid);
        throw std::runtime_error(buf);
    }
    // Skip execution if kernel name is NULL string
    if(cbInfo->symbolName == nullptr)
    {
        _LOG("Empty kernel name string. Skipping...");
        return;
    }

    static std::mutex            mtx;
    std::unique_lock<std::mutex> lk(mtx);

    map_type*     kernel_data = static_cast<map_type*>(userdata);
    kernel_data_t dummy       = (*kernel_data)[dummy_kernel_id];

    // if enter, assign unique correlation data ID
    if(cbInfo->callbackSite == CUPTI_API_ENTER)
        *cbInfo->correlationData = ++correlation_data;
    // get the correlation data ID
    uint64_t corr_data = *cbInfo->correlationData;

#if defined(DEBUG)
    printf("[kern] %s\n", cbInfo->symbolName);
    // construct a name
    std::stringstream _kernel_name_ss;
    const char*       _sym_name = cbInfo->symbolName;
    _kernel_name_ss << std::string(_sym_name) << "_" << cbInfo->contextUid << "_"
                    << corr_data;
    auto current_kernel_name = _kernel_name_ss.str();
#endif

    _LOG("... begin callback for %s...\n", current_kernel_name.c_str());
    if(cbInfo->callbackSite == CUPTI_API_ENTER)
    {
        _LOG("New kernel encountered: %s", current_kernel_name.c_str());
        kernel_data_t k_data = dummy;
        k_data.m_name        = corr_data;
        auto& pass_data      = k_data.m_pass_data;

        for(size_t j = 0; j < pass_data.size(); ++j)
        {
            for(uint32_t i = 0; i < pass_data[j].event_groups->numEventGroups; i++)
            {
                _LOG("  Enabling group %d", i);
                uint32_t all = 1;
                CUPTI_CALL(cuptiEventGroupSetAttribute(
                    pass_data[j].event_groups->eventGroups[i],
                    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all),
                    &all));
                CUPTI_CALL(
                    cuptiEventGroupEnable(pass_data[j].event_groups->eventGroups[i]));
            }
        }
        (*kernel_data)[corr_data] = k_data;
    }
    else if(cbInfo->callbackSite == CUPTI_API_EXIT)
    {
        auto& current_kernel = (*kernel_data)[corr_data];
        for(size_t current_pass = 0; current_pass < current_kernel.m_pass_data.size();
            ++current_pass)
        {
            auto& pass_data = current_kernel.m_pass_data[current_pass];

            for(uint32_t i = 0; i < pass_data.event_groups->numEventGroups; i++)
            {
                CUpti_EventGroup    group = pass_data.event_groups->eventGroups[i];
                CUpti_EventDomainID group_domain;
                uint32_t            numEvents, numInstances, numTotalInstances;
                size_t              groupDomainSize       = sizeof(group_domain);
                size_t              numEventsSize         = sizeof(numEvents);
                size_t              numInstancesSize      = sizeof(numInstances);
                size_t              numTotalInstancesSize = sizeof(numTotalInstances);

                CUPTI_CALL(cuptiEventGroupGetAttribute(
                    group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize,
                    &group_domain));
                CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
                    current_kernel.m_device, group_domain,
                    CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &numTotalInstancesSize,
                    &numTotalInstances));
                CUPTI_CALL(cuptiEventGroupGetAttribute(
                    group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &numInstancesSize,
                    &numInstances));
                CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                       CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                                       &numEventsSize, &numEvents));
                size_t         eventIdsSize = numEvents * sizeof(CUpti_EventID);
                CUpti_EventID* eventIds     = (CUpti_EventID*) malloc(eventIdsSize);
                CUPTI_CALL(cuptiEventGroupGetAttribute(
                    group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds));

                size_t    valuesSize = sizeof(uint64_t) * numInstances;
                uint64_t* values     = (uint64_t*) malloc(valuesSize);

                for(uint32_t j = 0; j < numEvents; j++)
                {
                    CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                                        eventIds[j], &valuesSize,
                                                        values));

                    // sum collect event values from all instances
                    uint64_t sum = 0;
                    for(uint32_t k = 0; k < numInstances; k++)
                        sum += values[k];

                    // normalize the event value to represent the total number of
                    // domain instances on the device
                    uint64_t normalized = (sum * numTotalInstances) / numInstances;

                    pass_data.event_ids.push_back(eventIds[j]);
                    pass_data.event_values.push_back(normalized);

// print collected value
#if defined(DEBUG)
                    {
                        char   eventName[128];
                        size_t eventNameSize = sizeof(eventName) - 1;
                        CUPTI_CALL(cuptiEventGetAttribute(eventIds[j],
                                                          CUPTI_EVENT_ATTR_NAME,
                                                          &eventNameSize, eventName));
                        eventName[eventNameSize] = '\0';
                        _DBG("\t%s = %llu (", eventName, (unsigned long long) sum);
                        for(int k = 0; k < numInstances && numInstances > 1; k++)
                        {
                            if(k != 0)
                                _DBG(", ");
                            _DBG("%llu", (unsigned long long) values[k]);
                        }
                        _DBG(")\n");
                        _LOG("\t%s (normalized) (%llu * %u) / %u = %llu", eventName,
                             (unsigned long long) sum, numTotalInstances, numInstances,
                             (unsigned long long) normalized);
                    }
#endif
                }
                free(values);
                free(eventIds);
            }
            ++(*kernel_data)[corr_data].m_current_pass;
        }
    }
    else
    {
        throw std::runtime_error("Unexpected callback site!");
    }
    _LOG("... ending callback for %s...\n", current_kernel_name.c_str());
}

//--------------------------------------------------------------------------------------//

}  // namespace impl

//--------------------------------------------------------------------------------------//

struct profiler
{
    using event_val_t  = impl::kernel_data_t::event_val_t;
    using metric_val_t = impl::kernel_data_t::metric_val_t;
    using results_t    = std::vector<result>;
    using ulong_t      = unsigned long long;

    profiler(const strvec_t& events, const strvec_t& metrics, const int device_num = 0)
    : m_device_num(device_num)
    , m_event_names(events)
    , m_metric_names(metrics)
    {
        int device_count = 0;

        // sync before starting
        tim::cuda::device_sync();

        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&device_count));
        if(device_count == 0)
        {
            fprintf(stderr, "There is no device supporting CUDA.\n");
            return;
        }
        if(events.size() + metrics.size() == 0)
        {
            fprintf(stderr, "No events or metrics were specified\n");
            return;
        }

        m_metric_ids.resize(metrics.size());
        m_event_ids.resize(events.size());

        // Init device, context and setup callback
        CUDA_DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
        CUDA_DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
        if(m_metric_names.size() > 0)
        {
            for(size_t i = 0; i < m_metric_names.size(); ++i)
                CUPTI_CALL(cuptiMetricGetIdFromName(m_device, m_metric_names[i].c_str(),
                                                    &m_metric_ids[i]));
            CUPTI_CALL(cuptiMetricCreateEventGroupSets(
                m_context, sizeof(CUpti_MetricID) * m_metric_names.size(),
                m_metric_ids.data(), &m_metric_pass_data));
            m_metric_passes = m_metric_pass_data->numSets;
        }

        if(m_event_names.size() > 0)
        {
            for(size_t i = 0; i < m_event_names.size(); ++i)
                CUPTI_CALL(cuptiEventGetIdFromName(m_device, m_event_names[i].c_str(),
                                                   &m_event_ids[i]));
            CUPTI_CALL(cuptiEventGroupSetsCreate(
                m_context, sizeof(CUpti_EventID) * m_event_ids.size(), m_event_ids.data(),
                &m_event_pass_data));
            m_event_passes = m_event_pass_data->numSets;
        }

        _LOG("# Metric Passes: %d\n", m_metric_passes);
        _LOG("# Event Passes: %d\n", m_event_passes);

        assert((m_metric_passes + m_event_passes) > 0);

        impl::kernel_data_t dummy_data;
        dummy_data.m_name          = "^^^__DUMMY__^^^";
        dummy_data.m_metric_passes = m_metric_passes;
        dummy_data.m_event_passes  = m_event_passes;
        dummy_data.m_device        = m_device;
        dummy_data.m_total_passes  = m_metric_passes + m_event_passes;
        dummy_data.m_pass_data.resize(m_metric_passes + m_event_passes);

        auto& pass_data = dummy_data.m_pass_data;
        for(int i = 0; i < m_metric_passes; ++i)
        {
            int total_events = 0;
            _LOG("[metric] Looking at set (pass) %d", i);
            uint32_t num_events      = 0;
            size_t   num_events_size = sizeof(num_events);
            for(uint32_t j = 0; j < m_metric_pass_data->sets[i].numEventGroups; ++j)
            {
                CUPTI_CALL(cuptiEventGroupGetAttribute(
                    m_metric_pass_data->sets[i].eventGroups[j],
                    CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &num_events_size, &num_events));
                _LOG("  Event Group %d, #Events = %d", j, num_events);
                total_events += num_events;
            }
            pass_data[i].event_groups = m_metric_pass_data->sets + i;
            pass_data[i].num_events   = total_events;
        }

        for(int i = 0; i < m_event_passes; ++i)
        {
            int total_events = 0;
            _LOG("[event] Looking at set (pass) %d", i);
            uint32_t num_events      = 0;
            size_t   num_events_size = sizeof(num_events);
            for(uint32_t j = 0; j < m_event_pass_data->sets[i].numEventGroups; ++j)
            {
                CUPTI_CALL(cuptiEventGroupGetAttribute(
                    m_event_pass_data->sets[i].eventGroups[j],
                    CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &num_events_size, &num_events));
                _LOG("  Event Group %d, #Events = %d", j, num_events);
                total_events += num_events;
            }
            pass_data[i + m_metric_passes].event_groups = m_event_pass_data->sets + i;
            pass_data[i + m_metric_passes].num_events   = total_events;
        }

        m_kernel_data[impl::dummy_kernel_id] = dummy_data;
        cuptiEnableKernelReplayMode(m_context);
        start();
        device::params<device::default_device> p(1, 1, 0, 0);
        device::launch(p, impl::warmup<int>);
        stop();
    }

    ~profiler()
    {
        /*
        for(auto& kitr : m_kernel_data)
        {
            //if(kitr.first != impl::dummy_kernel_id)
            //    continue;
            auto& pass_data = kitr.second.m_pass_data;
            for(int j = 0; j < pass_data.size(); ++j)
            {
            for(int i = 0; i < pass_data[j].event_groups->numEventGroups; i++)
            {
                CUPTI_CALL(cuptiEventGroupSetsDestroy(pass_data[j].event_groups->eventGroups[i]));
            }
            }
        }*/

        cuptiDisableKernelReplayMode(m_context);
        CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
    }

    int passes() { return m_metric_passes + m_event_passes; }

    void start()
    {
        if(m_is_running)
            return;
        m_is_running = true;

        tim::cuda::device_sync();

        CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                                  (CUpti_CallbackFunc) impl::get_value_callback,
                                  &m_kernel_data));

        CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
        CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

        CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000));
        CUPTI_CALL(
            cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000));

        tim::cuda::device_sync();
    }

    void stop()
    {
        if(!m_is_running)
            return;
        m_is_running = false;

        using event_id_map_t = std::map<CUpti_EventID, uint64_t>;

        tim::cuda::device_sync();
        for(auto& k : m_kernel_data)
        {
            if(k.first == impl::dummy_kernel_id)
                continue;

            auto& data         = k.second.m_pass_data;
            int   total_events = 0;
            for(int i = 0; i < m_metric_passes; ++i)
                total_events += data[i].num_events;

            CUpti_MetricValue metric_value;
            CUpti_EventID*    event_ids    = new CUpti_EventID[total_events];
            uint64_t*         event_values = new uint64_t[total_events];

            int running_sum = 0;
            for(int i = 0; i < m_metric_passes; ++i)
            {
                std::copy(data[i].event_ids.begin(), data[i].event_ids.end(),
                          event_ids + running_sum);
                std::copy(data[i].event_values.begin(), data[i].event_values.end(),
                          event_values + running_sum);
                running_sum += data[i].num_events;
            }

            for(size_t i = 0; i < m_metric_names.size(); ++i)
            {
                CUptiResult _status = cuptiMetricGetValue(
                    m_device, m_metric_ids[i], total_events * sizeof(CUpti_EventID),
                    event_ids, total_events * sizeof(uint64_t), event_values, 0,
                    &metric_value);
                if(_status != CUPTI_SUCCESS)
                {
                    fprintf(stderr, "Metric value retrieval failed for metric %s\n",
                            m_metric_names[i].c_str());
                    exit(-1);
                }
                k.second.m_metric_values.push_back(metric_value);
            }

            delete[] event_ids;
            delete[] event_values;

            event_id_map_t event_map;
            for(int i = m_metric_passes; i < (m_metric_passes + m_event_passes); ++i)
            {
                for(size_t j = 0; j < data[i].event_values.size(); ++j)
                    event_map[data[i].event_ids[j]] = data[i].event_values[j];
            }

            for(size_t i = 0; i < m_event_ids.size(); ++i)
                k.second.m_event_values.push_back(event_map[m_event_ids[i]]);
        }

        // Disable callback and unsubscribe
        CUPTI_CALL(cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
        CUPTI_CALL(cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
        CUPTI_CALL(cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000));
        CUPTI_CALL(
            cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000));
        CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
    }

    //----------------------------------------------------------------------------------//

    void print_event_values(std::ostream& os, bool print_names = true,
                            const char* kernel_separator = ";\n")
    {
        std::stringstream ss;
        for(auto& kitr : m_kernel_data)
        {
            if(kitr.first == impl::dummy_kernel_id)
                continue;
            for(size_t i = 0; i < m_event_names.size(); ++i)
            {
                if(print_names)
                    ss << "  (" << m_event_names.at(i) << ", " << std::setw(6)
                       << static_cast<ulong_t>(kitr.second.m_event_values.at(i)) << ") ";
                else
                    ss << std::setw(6)
                       << static_cast<ulong_t>(kitr.second.m_event_values.at(i)) << " ";
            }
            ss << kernel_separator;
        }
        os << ss.str() << std::endl;
    }

    //----------------------------------------------------------------------------------//

    void print_metric_values(std::ostream& os, bool print_names = true,
                             const char* kernel_separator = ";\n")
    {
        std::stringstream ss;
        for(auto& kitr : m_kernel_data)
        {
            if(kitr.first == impl::dummy_kernel_id)
                continue;
            for(size_t i = 0; i < m_metric_names.size(); ++i)
            {
                if(print_names)
                    ss << "  (" << m_metric_names[i] << ", ";
                ss << std::setw(6) << std::setprecision(2) << std::fixed;
                impl::print_metric(ss, m_metric_ids[i], kitr.second.m_metric_values[i]);
                ss << ((print_names) ? ") " : " ");
            }
            ss << kernel_separator;
        }
        os << ss.str() << std::endl;
    }

    //----------------------------------------------------------------------------------//

    void print_events_and_metrics(std::ostream& os, bool print_names = true,
                                  const char* kernel_separator = ";\n")
    {
        std::stringstream ss;
        print_event_values(ss, print_names, kernel_separator);
        print_metric_values(ss, print_names, kernel_separator);
        os << ss.str() << std::flush;
    }

    //----------------------------------------------------------------------------------//

    results_t get_events_and_metrics(const std::vector<std::string>& labels)
    {
        if(settings::verbose() > 2 || settings::debug())
            print_events_and_metrics(std::cout);

        results_t kern_data(labels.size());

        auto get_label_index = [&](const std::string& key) -> int64_t {
            for(int64_t i = 0; i < static_cast<int64_t>(labels.size()); ++i)
            {
                if(key == labels[i])
                    return i;
            }
            return -1;
        };

        for(auto& kitr : m_kernel_data)
        {
            if(kitr.first == impl::dummy_kernel_id)
                continue;
            for(size_t i = 0; i < m_event_names.size(); ++i)
            {
                std::string evt_name  = m_event_names[i].c_str();
                auto        label_idx = get_label_index(evt_name);
                if(label_idx < 0)
                    continue;
                auto         value = static_cast<uint64_t>(kitr.second.m_event_values[i]);
                data::metric ret;
                data::unsigned_integer::set(ret, value);
                kern_data[label_idx] += result(evt_name, ret, true);
            }

            for(size_t i = 0; i < m_metric_names.size(); ++i)
            {
                std::string met_name  = m_metric_names[i].c_str();
                auto        label_idx = get_label_index(met_name);
                if(label_idx < 0)
                    continue;
                auto ret =
                    impl::get_metric(m_metric_ids[i], kitr.second.m_metric_values[i]);
                kern_data[label_idx] += result(met_name, ret, false);
            }
        }
        return kern_data;
    }

    strvec_t get_kernel_names()
    {
        m_kernel_names.clear();
        for(auto& k : m_kernel_data)
        {
            if(k.first == impl::dummy_kernel_id)
                continue;
            m_kernel_names.push_back(std::to_string(k.first));
        }
        return m_kernel_names;
    }

    event_val_t get_event_values(const char* kernel_name)
    {
        if(m_kernel_data.count(atol(kernel_name)) != 0)
            return m_kernel_data[atol(kernel_name)].m_event_values;
        else
            return event_val_t{};
    }

    metric_val_t get_metric_values(const char* kernel_name)
    {
        if(m_kernel_data.count(atol(kernel_name)) != 0)
            return m_kernel_data[atol(kernel_name)].m_metric_values;
        else
            return metric_val_t{};
    }

    const strvec_t& get_event_names() const { return m_event_names; }
    const strvec_t& get_metric_names() const { return m_metric_names; }

private:
    bool                                 m_is_running    = false;
    int                                  m_device_num    = 0;
    int                                  m_metric_passes = 0;
    int                                  m_event_passes  = 0;
    strvec_t                             m_event_names;
    strvec_t                             m_metric_names;
    std::vector<CUpti_MetricID>          m_metric_ids;
    std::vector<CUpti_EventID>           m_event_ids;
    CUcontext                            m_context;
    CUdevice                             m_device;
    CUpti_SubscriberHandle               m_subscriber;
    CUpti_EventGroupSets*                m_metric_pass_data;
    CUpti_EventGroupSets*                m_event_pass_data;
    map_t<uint64_t, impl::kernel_data_t> m_kernel_data;
    strvec_t                             m_kernel_names;
};

//--------------------------------------------------------------------------------------//

inline strvec_t
available_metrics(CUdevice device);

//--------------------------------------------------------------------------------------//

inline strvec_t
available_events(CUdevice device);

//--------------------------------------------------------------------------------------//

namespace activity
{
//--------------------------------------------------------------------------------------//

inline const char*
memcpy_kind(CUpti_ActivityMemcpyKind kind)
{
    switch(kind)
    {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH";
        default: break;
    }

    return "<unknown>";
}

//--------------------------------------------------------------------------------------//

inline const char*
overhead_kind(CUpti_ActivityOverheadKind kind)
{
    switch(kind)
    {
        case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER: return "COMPILER";
        case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH: return "BUFFER_FLUSH";
        case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION: return "INSTRUMENTATION";
        case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE: return "RESOURCE";
        default: break;
    }

    return "<unknown>";
}

//--------------------------------------------------------------------------------------//

inline const char*
object_kind(CUpti_ActivityObjectKind kind)
{
    switch(kind)
    {
        case CUPTI_ACTIVITY_OBJECT_PROCESS: return "PROCESS";
        case CUPTI_ACTIVITY_OBJECT_THREAD: return "THREAD";
        case CUPTI_ACTIVITY_OBJECT_DEVICE: return "DEVICE";
        case CUPTI_ACTIVITY_OBJECT_CONTEXT: return "CONTEXT";
        case CUPTI_ACTIVITY_OBJECT_STREAM: return "STREAM";
        default: break;
    }

    return "<unknown>";
}

//--------------------------------------------------------------------------------------//

inline uint32_t
object_kind_id(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId* id)
{
    switch(kind)
    {
        case CUPTI_ACTIVITY_OBJECT_PROCESS: return id->pt.processId;
        case CUPTI_ACTIVITY_OBJECT_THREAD: return id->pt.threadId;
        case CUPTI_ACTIVITY_OBJECT_DEVICE: return id->dcs.deviceId;
        case CUPTI_ACTIVITY_OBJECT_CONTEXT: return id->dcs.contextId;
        case CUPTI_ACTIVITY_OBJECT_STREAM: return id->dcs.streamId;
        default: break;
    }

    return 0xffffffff;
}

//--------------------------------------------------------------------------------------//

inline const char*
compute_api_kind(CUpti_ActivityComputeApiKind kind)
{
    switch(kind)
    {
        case CUPTI_ACTIVITY_COMPUTE_API_CUDA: return "CUDA";
        case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS: return "CUDA_MPS";
        default: break;
    }

    return "<unknown>";
}

//--------------------------------------------------------------------------------------//

inline uint64_t
get_elapsed(CUpti_Activity* record)
{
    switch(record->kind)
    {
        break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy* memcpy = (CUpti_ActivityMemcpy*) record;
            return (memcpy->end) - (memcpy->start);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET:
        {
            CUpti_ActivityMemset* memset = (CUpti_ActivityMemset*) record;
            return (memset->end) - (memset->start);
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*) record;
            return (kernel->end) - (kernel->start);
            break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
            CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) record;
            return (api->end) - (api->start);
            break;
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
        {
            CUpti_ActivityOverhead* overhead = (CUpti_ActivityOverhead*) record;
            return (overhead->end) - (overhead->start);
            break;
        }
        case CUPTI_ACTIVITY_KIND_DEVICE:
        case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
        case CUPTI_ACTIVITY_KIND_CONTEXT:
        case CUPTI_ACTIVITY_KIND_NAME:
        case CUPTI_ACTIVITY_KIND_MARKER:
        case CUPTI_ACTIVITY_KIND_MARKER_DATA:
        default: break;
    }
    return 0;
}

//--------------------------------------------------------------------------------------//

inline void
print(CUpti_Activity* record)
{
    switch(record->kind)
    {
        break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy* memcpy = (CUpti_ActivityMemcpy*) record;
            printf(
                "MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, "
                "correlation %u/r%u\n",
                memcpy_kind((CUpti_ActivityMemcpyKind) memcpy->copyKind),
                (unsigned long long) (memcpy->start - start_timestamp()),
                (unsigned long long) (memcpy->end - start_timestamp()), memcpy->deviceId,
                memcpy->contextId, memcpy->streamId, memcpy->correlationId,
                memcpy->runtimeCorrelationId);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET:
        {
            CUpti_ActivityMemset* memset = (CUpti_ActivityMemset*) record;
            printf(
                "MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, "
                "correlation %u\n",
                memset->value, (unsigned long long) (memset->start - start_timestamp()),
                (unsigned long long) (memset->end - start_timestamp()), memset->deviceId,
                memset->contextId, memset->streamId, memset->correlationId);
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            const char* kindString =
                (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
            CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*) record;
            printf(
                "%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, "
                "correlation %u\n",
                kindString, kernel->name,
                (unsigned long long) (kernel->start - start_timestamp()),
                (unsigned long long) (kernel->end - start_timestamp()), kernel->deviceId,
                kernel->contextId, kernel->streamId, kernel->correlationId);
            printf(
                "    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, "
                "dynamic %u)\n",
                kernel->gridX, kernel->gridY, kernel->gridZ, kernel->blockX,
                kernel->blockY, kernel->blockZ, kernel->staticSharedMemory,
                kernel->dynamicSharedMemory);
            break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER:
        {
            CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) record;
            printf(
                "DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation "
                "%u\n",
                api->cbid, (unsigned long long) (api->start - start_timestamp()),
                (unsigned long long) (api->end - start_timestamp()), api->processId,
                api->threadId, api->correlationId);
            break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
            CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) record;
            printf(
                "RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation "
                "%u\n",
                api->cbid, (unsigned long long) (api->start - start_timestamp()),
                (unsigned long long) (api->end - start_timestamp()), api->processId,
                api->threadId, api->correlationId);
            break;
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
        {
            CUpti_ActivityOverhead* overhead = (CUpti_ActivityOverhead*) record;
            printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
                   overhead_kind(overhead->overheadKind),
                   (unsigned long long) overhead->start - start_timestamp(),
                   (unsigned long long) overhead->end - start_timestamp(),
                   object_kind(overhead->objectKind),
                   object_kind_id(overhead->objectKind, &overhead->objectId));
            break;
        }
        case CUPTI_ACTIVITY_KIND_DEVICE:
        case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
        case CUPTI_ACTIVITY_KIND_CONTEXT:
        case CUPTI_ACTIVITY_KIND_NAME:
        case CUPTI_ACTIVITY_KIND_MARKER:
        case CUPTI_ACTIVITY_KIND_MARKER_DATA:
        default: break;
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
static void CUPTIAPI
            request_buffer(uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
    uint8_t* bfr = (uint8_t*) malloc(CUPTI_BUFFER_SIZE + CUPTI_ALIGN_SIZE);
    if(bfr == nullptr)
        throw std::bad_alloc();

    *size          = CUPTI_BUFFER_SIZE;
    *buffer        = CUPTI_ALIGN_BUFFER(bfr, CUPTI_ALIGN_SIZE);
    *maxNumRecords = 0;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
static void CUPTIAPI
            buffer_completed(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t /*size*/,
                             size_t validSize)
{
    using lock_type = typename receiver<_Tp>::lock_type;

    CUptiResult     status;
    CUpti_Activity* record = nullptr;

    auto& _receiver = get_receiver<_Tp>();
    // obtain lock to keep data from being removed during update
    lock_type lk(_receiver.get_mutex());

    if(validSize > 0)
    {
        do
        {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if(status == CUPTI_SUCCESS)
            {
                if(settings::verbose() > 3 || settings::debug())
                    print(record);
                _receiver += get_elapsed(record);
            }
            else if(status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else
            {
                CUPTI_CALL(status);
            }
        } while(1);

        // report any records dropped from the queue
        size_t dropped = 0;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if(dropped != 0)
        {
            printf("[tim::cupti::activity::%s]> Dropped %u activity records\n",
                   __FUNCTION__, (unsigned int) dropped);
        }
    }

    free(buffer);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline std::vector<CUpti_ActivityKind>&
get_kind_types()
{
    static std::vector<CUpti_ActivityKind> _instance = {
        CUPTI_ACTIVITY_KIND_DEVICE, CUPTI_ACTIVITY_KIND_CONTEXT,
        CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
        CUPTI_ACTIVITY_KIND_NAME,   CUPTI_ACTIVITY_KIND_MARKER,
        CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_OVERHEAD
    };
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline void
scale_device_buffers(
    int64_t buffer_factor     = get_env("TIMEMORY_CUPTI_DEVICE_BUFFER_SIZE", 0),
    int64_t pool_limit_factor = get_env("TIMEMORY_CUPTI_DEVICE_BUFFER_POOL_LIMIT", 0))
{
    size_t deviceValue   = 0;
    size_t poolValue     = 0;
    size_t attrValueSize = sizeof(size_t);

    // Get and set activity attributes.
    // Attributes can be set by the CUPTI client to change behavior of the activity
    // API. Some attributes require to be set before any CUDA context is created to be
    // effective, e.g. to be applied to all device buffer allocations (see
    // documentation).

    // get the buffer size and increase
    CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                         &attrValueSize, &deviceValue));

    if(settings::verbose() > 1 || settings::debug())
        printf("[tim::cupti::activity::%s]> %s = %llu\n", __FUNCTION__,
               "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE",
               (long long unsigned) deviceValue);

    if(buffer_factor != 0)
    {
        size_t attrValue = deviceValue;
        if(buffer_factor > 1)
        {
            attrValue *= buffer_factor;
        }
        else if(buffer_factor < -1)
        {
            attrValue /= buffer_factor;
        }

        if(attrValue > 0)
            CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                                 &attrValueSize, &attrValue));
    }

    // get the buffer pool limit and increase
    CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                         &attrValueSize, &poolValue));

    if(settings::verbose() > 1 || settings::debug())
        printf("[tim::cupti::activity::%s]> %s = %llu\n", __FUNCTION__,
               "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned) poolValue);

    if(pool_limit_factor != 0)
    {
        size_t attrValue = poolValue;
        if(pool_limit_factor > 1)
        {
            attrValue *= pool_limit_factor;
        }
        else if(pool_limit_factor < -1)
        {
            attrValue /= pool_limit_factor;
        }
        if(attrValue > 0)
            CUPTI_CALL(
                cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                          &attrValueSize, &attrValue));
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
start_trace(_Tp* obj)
{
    auto& _receiver = get_receiver<_Tp>();

    if(_receiver.empty())
    {
        _receiver.set_kinds(get_kind_types<_Tp>());
        // Device activity record is created when CUDA initializes, so we
        // want to enable it before cuInit() or any CUDA runtime call.
        for(const auto& itr : _receiver.get_kinds())
            CUPTI_CALL(cuptiActivityEnable(itr));

        // Register callbacks for buffer requests and for buffers completed by CUPTI.
        CUPTI_CALL(
            cuptiActivityRegisterCallbacks(request_buffer<_Tp>, buffer_completed<_Tp>));

        auto& _start = start_timestamp();
        CUPTI_CALL(cuptiGetTimestamp(&_start));
    }
    _receiver.insert(obj);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
stop_trace(_Tp* obj)
{
    auto& _receiver = get_receiver<_Tp>();
    cuptiActivityFlushAll(0);
    _receiver.remove(obj);

    if(_receiver.empty())
    {
        // Device activity record is created when CUDA initializes, so we
        // want to enable it before cuInit() or any CUDA runtime call.
        for(const auto& itr : _receiver.get_kinds())
            CUPTI_CALL(cuptiActivityDisable(itr));
    }
}

//--------------------------------------------------------------------------------------//

}  // namespace activity

//--------------------------------------------------------------------------------------//

}  // namespace cupti

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#if !defined(__CUPTI_PROFILER_NAME_SHORT)
#    define __CUPTI_PROFILER_NAME_SHORT 128
#endif

//--------------------------------------------------------------------------------------//

inline tim::cupti::strvec_t
tim::cupti::available_metrics(CUdevice device)
{
    strvec_t              metric_names;
    uint32_t              numMetric;
    size_t                size;
    char                  metricName[__CUPTI_PROFILER_NAME_SHORT];
    CUpti_MetricValueKind metricKind;
    CUpti_MetricID*       metricIdArray;

    CUPTI_CALL(cuptiDeviceGetNumMetrics(device, &numMetric));
    size          = sizeof(CUpti_MetricID) * numMetric;
    metricIdArray = (CUpti_MetricID*) malloc(size);
    if(metricIdArray == nullptr)
    {
        printf("Memory could not be allocated for metric array");
        return metric_names;
    }

    CUPTI_CALL(cuptiDeviceEnumMetrics(device, &size, metricIdArray));

    for(uint32_t i = 0; i < numMetric; i++)
    {
        size = __CUPTI_PROFILER_NAME_SHORT;
        CUPTI_CALL(cuptiMetricGetAttribute(metricIdArray[i], CUPTI_METRIC_ATTR_NAME,
                                           &size, (void*) &metricName));
        size = sizeof(CUpti_MetricValueKind);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIdArray[i], CUPTI_METRIC_ATTR_VALUE_KIND,
                                           &size, (void*) &metricKind));
        if((metricKind == CUPTI_METRIC_VALUE_KIND_THROUGHPUT) ||
           (metricKind == CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL))
        {
            if(settings::verbose() > 0 || settings::debug())
                printf(
                    "Metric %s cannot be profiled as metric requires GPU"
                    "time duration for kernel run.\n",
                    metricName);
        }
        else
        {
            metric_names.push_back(metricName);
        }
    }
    free(metricIdArray);
    return std::move(metric_names);
}

//--------------------------------------------------------------------------------------//

inline tim::cupti::strvec_t
tim::cupti::available_events(CUdevice device)
{
    strvec_t             event_names;
    uint32_t             numDomains  = 0;
    uint32_t             num_events  = 0;
    uint32_t             totalEvents = 0;
    size_t               size;
    CUpti_EventDomainID* domainIdArray;
    CUpti_EventID*       eventIdArray;
    size_t               eventIdArraySize;
    char                 eventName[__CUPTI_PROFILER_NAME_SHORT];

    CUPTI_CALL(cuptiDeviceGetNumEventDomains(device, &numDomains));
    size          = sizeof(CUpti_EventDomainID) * numDomains;
    domainIdArray = (CUpti_EventDomainID*) malloc(size);
    if(domainIdArray == nullptr)
    {
        printf("Memory could not be allocated for domain array");
        return event_names;
    }
    CUPTI_CALL(cuptiDeviceEnumEventDomains(device, &size, domainIdArray));

    for(uint32_t i = 0; i < numDomains; i++)
    {
        CUPTI_CALL(cuptiEventDomainGetNumEvents(domainIdArray[i], &num_events));
        totalEvents += num_events;
    }

    eventIdArraySize = sizeof(CUpti_EventID) * totalEvents;
    eventIdArray     = (CUpti_EventID*) malloc(eventIdArraySize);

    totalEvents = 0;
    for(uint32_t i = 0; i < numDomains; i++)
    {
        // Query num of events available in the domain
        CUPTI_CALL(cuptiEventDomainGetNumEvents(domainIdArray[i], &num_events));
        size = num_events * sizeof(CUpti_EventID);
        CUPTI_CALL(cuptiEventDomainEnumEvents(domainIdArray[i], &size,
                                              eventIdArray + totalEvents));
        totalEvents += num_events;
    }

    for(uint32_t i = 0; i < totalEvents; i++)
    {
        size = __CUPTI_PROFILER_NAME_SHORT;
        CUPTI_CALL(cuptiEventGetAttribute(eventIdArray[i], CUPTI_EVENT_ATTR_NAME, &size,
                                          eventName));
        event_names.push_back(eventName);
    }
    free(domainIdArray);
    free(eventIdArray);
    return std::move(event_names);
}

#undef __CUPTI_PROFILER_NAME_SHORT
