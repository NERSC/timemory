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
#include "timemory/macros.hpp"
#include "timemory/utility.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
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

#define CUPTI_ALIGN_SIZE (8)
#define CUPTI_ALIGN_BUFFER(buffer, align)                                                \
    (((uintptr_t)(buffer) & ((align) -1))                                                \
         ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align) -1)))                   \
         : (buffer))

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace cupti
{
//--------------------------------------------------------------------------------------//

using string_t = std::string;
template <typename _Key, typename _Mapped>
using map_t                 = std::map<_Key, _Mapped>;
using strvec_t              = std::vector<string_t>;
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
    if(*buffer == NULL)
    {
        throw std::runtime_error("Error: out of memory\n");
    }
}

//--------------------------------------------------------------------------------------//

static void CUPTIAPI
            buffer_completed(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
                             size_t validSize)
{
    CUpti_Activity* record = nullptr;

    // since we launched only 1 kernel, we should have only 1 kernel record
    CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

    CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*) record;
    if(kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL)
    {
        std::stringstream ss;
        ss << "Error: expected kernel activity record, got "
           << static_cast<int>(kernel->kind);
        throw std::runtime_error(ss.str());
    }

    (*get_stream_kernel_duration())[streamId] = kernel->end - kernel->start;

    free(buffer);
}

//--------------------------------------------------------------------------------------//

namespace impl
{
//--------------------------------------------------------------------------------------//

static uint64_t dummy_kernel_id = 0;
using metric_tuple_t            = std::tuple<double, uint64_t, int64_t>;

//--------------------------------------------------------------------------------------//

__global__ void
warmup()
{
}

//--------------------------------------------------------------------------------------//

static metric_tuple_t
get_metric_tuple(CUpti_MetricID& id, CUpti_MetricValue& value)
{
    CUpti_MetricValueKind value_kind;
    size_t                value_kind_sz = sizeof(value_kind);
    CUPTI_CALL(cuptiMetricGetAttribute(id, CUPTI_METRIC_ATTR_VALUE_KIND, &value_kind_sz,
                                       &value_kind));
    metric_tuple_t ret(0.0, 0, 0);
    switch(value_kind)
    {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
            std::get<0>(ret) = value.metricValueDouble;
            break;
        case CUPTI_METRIC_VALUE_KIND_UINT64:
            std::get<1>(ret) = value.metricValueUint64;
            break;
        case CUPTI_METRIC_VALUE_KIND_INT64:
            std::get<2>(ret) = value.metricValueInt64;
            break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT:
            std::get<0>(ret) = value.metricValuePercent;
            break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
            std::get<0>(ret) = value.metricValueThroughput;
            break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            std::get<2>(ret) = value.metricValueUtilizationLevel;
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

static int
get_metric_tuple_index(CUpti_MetricID& id)
{
    CUpti_MetricValueKind value_kind;
    size_t                value_kind_sz = sizeof(value_kind);
    CUPTI_CALL(cuptiMetricGetAttribute(id, CUPTI_METRIC_ATTR_VALUE_KIND, &value_kind_sz,
                                       &value_kind));
    switch(value_kind)
    {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE: return 0; break;
        case CUPTI_METRIC_VALUE_KIND_UINT64: return 1; break;
        case CUPTI_METRIC_VALUE_KIND_INT64: return 2; break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT: return 0; break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT: return 0; break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL: return 2; break;
        default: break;
    }
    return -1;
}

//--------------------------------------------------------------------------------------//
// Generic tuple operations
//
template <size_t _N, typename _Tuple>
void
plus(_Tuple& lhs, const _Tuple& rhs)
{
    std::get<_N>(lhs) += std::get<_N>(rhs);
}

template <size_t _N, typename _Tuple>
void
minus(_Tuple& lhs, const _Tuple& rhs)
{
    std::get<_N>(lhs) -= std::get<_N>(rhs);
}

inline metric_tuple_t&
operator+=(metric_tuple_t& lhs, const metric_tuple_t& rhs)
{
    plus<0>(lhs, rhs);
    plus<1>(lhs, rhs);
    plus<2>(lhs, rhs);
    return lhs;
}

inline metric_tuple_t&
operator-=(metric_tuple_t& lhs, const metric_tuple_t& rhs)
{
    minus<0>(lhs, rhs);
    minus<1>(lhs, rhs);
    minus<2>(lhs, rhs);
    return lhs;
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
    using metric_tup_t = std::vector<metric_tuple_t>;
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
        m_metric_tuples.resize(rhs.m_metric_tuples.size(), metric_tuple_t());
    }

    kernel_data_t& operator+=(const kernel_data_t& rhs)
    {
        m_event_values.resize(rhs.m_event_values.size(), 0);
        for(uint64_t i = 0; i < rhs.m_event_values.size(); ++i)
            m_event_values[i] += rhs.m_event_values[i];

        m_metric_tuples.resize(rhs.m_metric_tuples.size(), metric_tuple_t());
        for(uint64_t i = 0; i < rhs.m_metric_tuples.size(); ++i)
            m_metric_tuples[i] += rhs.m_metric_tuples[i];

        return *this;
    }

    kernel_data_t& operator-=(const kernel_data_t& rhs)
    {
        m_event_values.resize(rhs.m_event_values.size(), 0);
        for(uint64_t i = 0; i < rhs.m_event_values.size(); ++i)
            m_event_values[i] -= rhs.m_event_values[i];

        m_metric_tuples.resize(rhs.m_metric_tuples.size(), metric_tuple_t());
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
       (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
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

        for(int j = 0; j < pass_data.size(); ++j)
        {
            for(int i = 0; i < pass_data[j].event_groups->numEventGroups; i++)
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
        for(int current_pass = 0; current_pass < current_kernel.m_pass_data.size();
            ++current_pass)
        {
            auto& pass_data = current_kernel.m_pass_data[current_pass];

            for(int i = 0; i < pass_data.event_groups->numEventGroups; i++)
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

struct result
{
    using data_t = std::tuple<uint64_t, int64_t, double>;

    bool        is_event_value = true;
    int         index          = 0;
    std::string name           = "unk";
    data_t      data           = data_t(0, 0, 0.0);

    result()              = default;
    ~result()             = default;
    result(const result&) = default;
    result(result&&)      = default;
    result& operator=(const result&) = default;
    result& operator=(result&&) = default;

    explicit result(const std::string& _name, const uint64_t& _data, bool _is = true)
    : is_event_value(_is)
    , index(0)
    , name(_name)
    , data({ _data, 0, 0.0 })
    {
    }

    explicit result(const std::string& _name, const int64_t& _data, bool _is = false)
    : is_event_value(_is)
    , index(1)
    , name(_name)
    , data({ 0, _data, 0.0 })
    {
    }

    explicit result(const std::string& _name, const double& _data, bool _is = false)
    : is_event_value(_is)
    , index(2)
    , name(_name)
    , data({ 0, 0, _data })
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const result& obj)
    {
        std::stringstream ss;
        ss << std::setprecision(2)
           << ((obj.index == 0)
                   ? std::get<0>(obj.data)
                   : ((obj.index == 1) ? std::get<1>(obj.data) : std::get<2>(obj.data)));
        ss << " " << obj.name;
        os << ss.str();
        return os;
    }

    bool operator==(const result& rhs) const { return (name == rhs.name); }
    bool operator!=(const result& rhs) const { return !(*this == rhs); }
    bool operator<(const result& rhs) const { return (name < rhs.name); }
    bool operator>(const result& rhs) const { return (name > rhs.name); }
    bool operator<=(const result& rhs) const { return !(*this > rhs); }
    bool operator>=(const result& rhs) const { return !(*this < rhs); }

    result& operator+=(const result& rhs)
    {
        if(name == "unk")
            return operator=(rhs);
        impl::plus<0>(data, rhs.data);
        impl::plus<1>(data, rhs.data);
        impl::plus<2>(data, rhs.data);
        return *this;
    }

    result& operator-=(const result& rhs)
    {
        if(name == "unk")
            return operator=(rhs);
        impl::minus<0>(data, rhs.data);
        impl::minus<1>(data, rhs.data);
        impl::minus<2>(data, rhs.data);
        return *this;
    }

    friend result operator+(const result& lhs, const result& rhs)
    {
        return result(lhs) += rhs;
    }

    friend result operator-(const result& lhs, const result& rhs)
    {
        return result(lhs) -= rhs;
    }
};

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

        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&device_count));
        if(device_count == 0)
        {
            fprintf(stderr, "There is no device supporting CUDA.\n");
            return;
        }

        m_metric_ids.resize(metrics.size());
        m_event_ids.resize(events.size());

        // Init device, context and setup callback
        CUDA_DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
        CUDA_DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
        CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                                  (CUpti_CallbackFunc) impl::get_value_callback,
                                  &m_kernel_data));
        CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
        CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

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
            for(int j = 0; j < m_metric_pass_data->sets[i].numEventGroups; ++j)
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
            for(int j = 0; j < m_event_pass_data->sets[i].numEventGroups; ++j)
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
        impl::warmup<<<1, 1>>>();
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

    void start() {}
    void stop()
    {
        using event_id_map_t = std::map<CUpti_EventID, uint64_t>;
        tim::cuda::stream_sync(0);
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
                auto value = static_cast<uint64_t>(kitr.second.m_event_values[i]);
                kern_data[label_idx] += result(evt_name, value, true);
            }

            for(size_t i = 0; i < m_metric_names.size(); ++i)
            {
                std::string met_name  = m_metric_names[i].c_str();
                auto        label_idx = get_label_index(met_name);
                if(label_idx < 0)
                    continue;
                auto idx = impl::get_metric_tuple_index(m_metric_ids[i]);
                auto ret = impl::get_metric_tuple(m_metric_ids[i],
                                                  kitr.second.m_metric_values[i]);
                switch(idx)
                {
                    case 0:
                        kern_data[label_idx] += result(met_name, std::get<0>(ret), false);
                        break;
                    case 1:
                        kern_data[label_idx] += result(met_name, std::get<1>(ret), false);
                        break;
                    case 2:
                        kern_data[label_idx] += result(met_name, std::get<2>(ret), false);
                        break;
                }
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

#if !defined(__CUPTI_PROFILER_NAME_SHORT)
#    define __CUPTI_PROFILER_NAME_SHORT 128
#endif

//--------------------------------------------------------------------------------------//

static strvec_t
available_metrics(CUdevice device)
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
    if(NULL == metricIdArray)
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

static strvec_t
available_events(CUdevice device)
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
    if(NULL == domainIdArray)
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

//--------------------------------------------------------------------------------------//

}  // namespace cupti

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
