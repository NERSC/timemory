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
using map_t = std::map<_Key, _Mapped>;

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

// User data for event collection callback
struct metric_data_t
{
    // the device where metric is being collected
    CUdevice device;
    // the set of event groups to collect for a pass
    CUpti_EventGroupSet* eventGroups;
    // the current number of events collected in eventIdArray and
    // eventValueArray
    uint32_t eventIdx;
    // the number of entries in eventIdArray and eventValueArray
    uint32_t numEvents;
    // array of event ids
    CUpti_EventID* eventIdArray;
    // array of event values
    uint64_t* eventValueArray;
};

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

using metric_tuple_t = std::tuple<double, uint64_t, int64_t>;

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
// data for the kernels
//
struct kernel_data_t
{
    using event_val_t  = std::vector<uint64_t>;
    using metric_val_t = std::vector<CUpti_MetricValue>;
    using metric_tup_t = std::vector<metric_tuple_t>;
    using pass_val_t   = std::vector<pass_data_t>;

    kernel_data_t() {}

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

    kernel_data_t(const kernel_data_t&) = default;
    kernel_data_t(kernel_data_t&&)      = default;

    kernel_data_t& operator=(const kernel_data_t&) = default;
    kernel_data_t& operator=(kernel_data_t&&) = default;

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
static void CUPTIAPI
            get_value_callback(void* userdata, CUpti_CallbackDomain /*domain*/, CUpti_CallbackId cbid,
                               const CUpti_CallbackData* cbInfo)
{
    // This callback is enabled only for launch so we shouldn't see
    // anything else.
    if((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
       (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {
        fprintf(stderr, "%s:%d: Unexpected cbid %d\n", __FILE__, __LINE__, cbid);
        return;
    }

    const char* _current_kernel_name = cbInfo->symbolName;

    // Skip execution if kernel name is NULL string
    // TODO: Make sure this is fine
    if(!_current_kernel_name)
    {
        _LOG("Empty kernel name string. Skipping...");
        return;
    }

    using map_type = map_t<string_t, kernel_data_t>;

    std::stringstream _kernel_name_ss;
    const char*       _sym_name  = cbInfo->symbolName;
    const char*       _func_name = cbInfo->functionName;
    _kernel_name_ss << std::string(_sym_name) << "_" << cbInfo->contextUid << "_"
                    << cbInfo->correlationId << "_" << std::string(_func_name);
    auto  current_kernel_name = _kernel_name_ss.str();
    auto* kernel_data         = static_cast<map_type*>(userdata);
    _LOG("... begin callback for %s...\n", current_kernel_name.c_str());

    if(cbInfo->callbackSite == CUPTI_API_ENTER)
    {
        _LOG("CUPTI_API_ENTER... starting callback for %s...\n",
             current_kernel_name.c_str());

        // If this is kernel name hasn't been seen before
        if(kernel_data->count(current_kernel_name) == 0)
        {
            _LOG("New kernel encountered: %s", current_kernel_name.c_str());

            const char*   dummy_kernel_name = "^^ DUMMY ^^";
            kernel_data_t dummy             = (*kernel_data)[dummy_kernel_name];
            kernel_data_t k_data            = dummy;

            k_data.m_name = current_kernel_name;

            auto& pass_data = k_data.m_pass_data;

            CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                                                   CUPTI_EVENT_COLLECTION_MODE_KERNEL));

            for(uint32_t i = 0; i < pass_data[0].event_groups->numEventGroups; i++)
            {
                _LOG("  Enabling group %d", i);
                uint32_t all = 1;
                CUPTI_CALL(cuptiEventGroupSetAttribute(
                    pass_data[0].event_groups->eventGroups[i],
                    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all),
                    &all));
                CUPTI_CALL(
                    cuptiEventGroupEnable(pass_data[0].event_groups->eventGroups[i]));

                (*kernel_data)[current_kernel_name] = k_data;
            }
        }
        else
        {
            auto&       current_kernel = (*kernel_data)[current_kernel_name];
            auto const& pass_data      = current_kernel.m_pass_data;

            int current_pass = current_kernel.m_current_pass;
            if(current_pass >= current_kernel.m_total_passes)
                return;

            _LOG("Current pass for %s: %d", current_kernel_name.c_str(), current_pass);

            CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                                                   CUPTI_EVENT_COLLECTION_MODE_KERNEL));

            for(uint32_t i = 0; i < pass_data[current_pass].event_groups->numEventGroups;
                i++)
            {
                _LOG("  Enabling group %d", i);
                uint32_t all = 1;
                CUPTI_CALL(cuptiEventGroupSetAttribute(
                    pass_data[current_pass].event_groups->eventGroups[i],
                    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all),
                    &all));
                CUPTI_CALL(cuptiEventGroupEnable(
                    pass_data[current_pass].event_groups->eventGroups[i]));
            }
        }
        _LOG("CUPTI_API_ENTER... ending callback for %s...\n",
             current_kernel_name.c_str());
    }
    else if(cbInfo->callbackSite == CUPTI_API_EXIT)
    {
        _LOG("CUPTI_API_EXIT... starting callback for %s...\n",
             current_kernel_name.c_str());
        auto& current_kernel = (*kernel_data)[current_kernel_name];
        int   current_pass   = current_kernel.m_current_pass;

        if(current_pass >= current_kernel.m_total_passes)
            return;

        auto& pass_data = current_kernel.m_pass_data[current_pass];

        for(uint32_t i = 0; i < pass_data.event_groups->numEventGroups; i++)
        {
            CUpti_EventGroup    group = pass_data.event_groups->eventGroups[i];
            CUpti_EventDomainID group_domain;
            uint32_t            num_events;
            uint32_t            num_instances;
            uint32_t            num_total_instances;
            CUpti_EventID*      event_ids;
            size_t              group_domain_size        = sizeof(group_domain);
            size_t              num_events_size          = sizeof(num_events);
            size_t              num_instances_size       = sizeof(num_instances);
            size_t              num_total_instances_size = sizeof(num_total_instances);
            uint64_t*           values;
            uint64_t            normalized;
            uint64_t            sum;
            size_t              values_size;
            size_t              event_ids_size;

            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                   CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                                   &group_domain_size, &group_domain));
            CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
                current_kernel.m_device, group_domain,
                CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &num_total_instances_size,
                &num_total_instances));
            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                   CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                                   &num_instances_size, &num_instances));
            CUPTI_CALL(cuptiEventGroupGetAttribute(
                group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &num_events_size, &num_events));
            event_ids_size = num_events * sizeof(CUpti_EventID);
            event_ids      = (CUpti_EventID*) malloc(event_ids_size);
            CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                                   &event_ids_size, event_ids));

            values_size = sizeof(uint64_t) * num_instances;
            values      = (uint64_t*) malloc(values_size);

            for(uint32_t j = 0; j < num_events; j++)
            {
                CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                                    event_ids[j], &values_size, values));

                // sum collect event values from all instances
                sum = 0;
                for(uint32_t k = 0; k < num_instances; k++)
                    sum += values[k];

                // normalize the event value to represent the total number of
                // domain instances on the device
                normalized = (sum * num_total_instances) / num_instances;

                pass_data.event_ids.push_back(event_ids[j]);
                pass_data.event_values.push_back(normalized);

                // print collected value
                {
                    char   eventName[128];
                    size_t eventNameSize = sizeof(eventName) - 1;
                    CUPTI_CALL(cuptiEventGetAttribute(event_ids[j], CUPTI_EVENT_ATTR_NAME,
                                                      &eventNameSize, eventName));
                    eventName[127] = '\0';
                    _DBG("\t%s = %llu (", eventName, (unsigned long long) sum);
                    if(num_instances > 1)
                    {
                        for(uint32_t k = 0; k < num_instances; k++)
                        {
                            if(k != 0)
                                _DBG(", ");
                            _DBG("%llu", (unsigned long long) values[k]);
                        }
                    }

                    _DBG(")\n");
                    _LOG("\t%s (normalized) (%llu * %u) / %u = %llu", eventName,
                         (unsigned long long) sum, num_total_instances, num_instances,
                         (unsigned long long) normalized);
                }
            }
            free(values);
            free(event_ids);
        }

        for(uint32_t i = 0; i < pass_data.event_groups->numEventGroups; i++)
        {
            _LOG("  Disabling group %d", i);
            CUPTI_CALL(cuptiEventGroupDisable(pass_data.event_groups->eventGroups[i]));
        }
        ++(*kernel_data)[current_kernel_name].m_current_pass;
        _LOG("CUPTI_API_EXIT... ending callback for %s...\n",
             current_kernel_name.c_str());
    }
    _LOG("... ending callback for %s...\n", current_kernel_name.c_str());
}

//--------------------------------------------------------------------------------------//

static void
print_metric(CUpti_MetricID& id, CUpti_MetricValue& value, std::ostream& s)
{
    CUpti_MetricValueKind value_kind;
    size_t                value_kind_sz = sizeof(value_kind);
    CUPTI_CALL(cuptiMetricGetAttribute(id, CUPTI_METRIC_ATTR_VALUE_KIND, &value_kind_sz,
                                       &value_kind));
    switch(value_kind)
    {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE: s << value.metricValueDouble; break;
        case CUPTI_METRIC_VALUE_KIND_UINT64: s << value.metricValueUint64; break;
        case CUPTI_METRIC_VALUE_KIND_INT64: s << value.metricValueInt64; break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT: s << value.metricValuePercent; break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT: s << value.metricValueThroughput; break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            s << value.metricValueUtilizationLevel;
            break;
        default: std::cerr << "[error]: unknown value kind\n"; break;
    }
}

//--------------------------------------------------------------------------------------//

static void CUPTIAPI
            get_metric_value_callback(void* userdata, CUpti_CallbackDomain domain,
                                      CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo)
{
    metric_data_t* metricData = static_cast<metric_data_t*>(userdata);

    // This callback is enabled only for launch so we shouldn't see
    // anything else.
    if((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
       (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {
        printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
        exit(-1);
    }

    // on entry, enable all the event groups being collected this pass,
    // for metrics we collect for all instances of the event
    if(cbInfo->callbackSite == CUPTI_API_ENTER)
    {
        cudaDeviceSynchronize();

        CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                                               CUPTI_EVENT_COLLECTION_MODE_KERNEL));

        for(uint32_t i = 0; i < metricData->eventGroups->numEventGroups; i++)
        {
            uint32_t all = 1;
            CUPTI_CALL(cuptiEventGroupSetAttribute(
                metricData->eventGroups->eventGroups[i],
                CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all));
            CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
        }
    }

    // on exit, read and record event values
    if(cbInfo->callbackSite == CUPTI_API_EXIT)
    {
        cudaDeviceSynchronize();

        // for each group, read the event values from the group and record
        // in metricData
        for(uint32_t i = 0; i < metricData->eventGroups->numEventGroups; i++)
        {
            CUpti_EventGroup    group = metricData->eventGroups->eventGroups[i];
            CUpti_EventDomainID groupDomain;
            uint32_t            numEvents, numInstances, numTotalInstances;
            CUpti_EventID*      eventIds;
            size_t              groupDomainSize       = sizeof(groupDomain);
            size_t              numEventsSize         = sizeof(numEvents);
            size_t              numInstancesSize      = sizeof(numInstances);
            size_t              numTotalInstancesSize = sizeof(numTotalInstances);
            uint64_t *          values, normalized, sum;
            size_t              valuesSize, eventIdsSize;

            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                   CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                                   &groupDomainSize, &groupDomain));
            CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
                metricData->device, groupDomain,
                CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &numTotalInstancesSize,
                &numTotalInstances));
            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                   CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                                   &numInstancesSize, &numInstances));
            CUPTI_CALL(cuptiEventGroupGetAttribute(
                group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &numEventsSize, &numEvents));
            eventIdsSize = numEvents * sizeof(CUpti_EventID);
            eventIds     = (CUpti_EventID*) malloc(eventIdsSize);
            CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                                   &eventIdsSize, eventIds));

            valuesSize = sizeof(uint64_t) * numInstances;
            values     = (uint64_t*) malloc(valuesSize);

            for(uint32_t j = 0; j < numEvents; j++)
            {
                CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                                    eventIds[j], &valuesSize, values));
                if(metricData->eventIdx >= metricData->numEvents)
                {
                    fprintf(stderr,
                            "error: too many events collected, metric expects only %d\n",
                            (int) metricData->numEvents);
                    exit(-1);
                }

                // sum collect event values from all instances
                sum = 0;
                for(uint32_t k = 0; k < numInstances; k++)
                    sum += values[k];

                // normalize the event value to represent the total number of
                // domain instances on the device
                normalized = (sum * numTotalInstances) / numInstances;

                metricData->eventIdArray[metricData->eventIdx]    = eventIds[j];
                metricData->eventValueArray[metricData->eventIdx] = normalized;
                metricData->eventIdx++;

                // print collected value
                {
                    char   eventName[128];
                    size_t eventNameSize = sizeof(eventName) - 1;
                    CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
                                                      &eventNameSize, eventName));
                    eventName[127] = '\0';
                    printf("\t%s = %llu (", eventName, (unsigned long long) sum);
                    if(numInstances > 1)
                    {
                        for(uint32_t k = 0; k < numInstances; k++)
                        {
                            if(k != 0)
                                printf(", ");
                            printf("%llu", (unsigned long long) values[k]);
                        }
                    }

                    printf(")\n");
                    printf("\t%s (normalized) (%llu * %u) / %u = %llu\n", eventName,
                           (unsigned long long) sum, numTotalInstances, numInstances,
                           (unsigned long long) normalized);
                }
            }

            free(values);
        }

        for(uint32_t i = 0; i < metricData->eventGroups->numEventGroups; i++)
            CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
    }
}

//--------------------------------------------------------------------------------------//

}  // namespace impl

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

    bool operator==(const result& rhs) const { return (name == rhs.name); }
    bool operator!=(const result& rhs) const { return !(*this == rhs); }
    bool operator<(const result& rhs) const { return (name < rhs.name); }
    bool operator>(const result& rhs) const { return (name > rhs.name); }
    bool operator<=(const result& rhs) const { return !(*this > rhs); }
    bool operator>=(const result& rhs) const { return !(*this < rhs); }

    result& operator+=(const result& rhs)
    {
        impl::plus<0>(data, rhs.data);
        impl::plus<1>(data, rhs.data);
        impl::plus<2>(data, rhs.data);
        return *this;
    }

    result& operator-=(const result& rhs)
    {
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

struct metric_profiler
{
    metric_profiler() {}

    void start(int _devNum = 0, const std::string& _metric = "ipc")
    {
        // printf("Usage: %s [device_num] [metric_name]\n", argv[0]);

        // make sure activity is enabled before any CUDA API
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

        CUDA_DRIVER_API_CALL(cuInit(0));
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
        if(deviceCount == 0)
        {
            fprintf(stderr, "There is no device supporting CUDA.\n");
            return;
        }

        deviceNum = _devNum;
        printf("CUDA Device Number: %d\n", deviceNum);

        CUDA_DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
        CUDA_DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
        printf("CUDA Device Name: %s\n", deviceName);

        CUDA_DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

        // Get the name of the metric to collect
        metricName = _metric.c_str();

        // need to collect duration of kernel execution without any event
        // collection enabled (some metrics need kernel duration as part of
        // calculation). The only accurate way to do this is by using the
        // activity API.
        // CUPTI_CALL(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
        // cudaDeviceSynchronize();
        // CUPTI_CALL(cuptiActivityFlushAll(0));

        // setup launch callback for event collection
        CUPTI_CALL(cuptiSubscribe(&subscriber,
                                  (CUpti_CallbackFunc) impl::get_metric_value_callback,
                                  &metricData));
        CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
        CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

        // allocate space to hold all the events needed for the metric
        CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName, &metricId));
        CUPTI_CALL(cuptiMetricGetNumEvents(metricId, &metricData.numEvents));
        metricData.device = device;
        metricData.eventIdArray =
            (CUpti_EventID*) malloc(metricData.numEvents * sizeof(CUpti_EventID));
        metricData.eventValueArray =
            (uint64_t*) malloc(metricData.numEvents * sizeof(uint64_t));
        metricData.eventIdx = 0;

        // get the number of passes required to collect all the events
        // needed for the metric and the event groups for each pass
        CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metricId), &metricId,
                                                   &passData));
    }

    void stop()
    {
        // for(pass = 0; pass < passData->numSets; pass++)
        //{
        //    printf("Pass %u\n", pass);
        //    metricData.eventGroups = passData->sets + pass;
        //    runPass();
        //}

        if(metricData.eventIdx != metricData.numEvents)
        {
            char buffer[1024];
            sprintf(buffer, "error: expected %u metric events, got %u\n",
                    metricData.numEvents, metricData.eventIdx);
            throw std::runtime_error(std::string(buffer));
        }

        uint64_t kernelDuration = 0;
        // use all the collected events to calculate the metric value
        CUPTI_CALL(cuptiMetricGetValue(
            device, metricId, metricData.numEvents * sizeof(CUpti_EventID),
            metricData.eventIdArray, metricData.numEvents * sizeof(uint64_t),
            metricData.eventValueArray, kernelDuration, &metricValue));

        // print metric value, we format based on the value kind
        {
            CUpti_MetricValueKind valueKind;
            size_t                valueKindSize = sizeof(valueKind);
            CUPTI_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
                                               &valueKindSize, &valueKind));
            switch(valueKind)
            {
                case CUPTI_METRIC_VALUE_KIND_DOUBLE:
                    printf("Metric %s = %f\n", metricName, metricValue.metricValueDouble);
                    break;
                case CUPTI_METRIC_VALUE_KIND_UINT64:
                    printf("Metric %s = %llu\n", metricName,
                           (unsigned long long) metricValue.metricValueUint64);
                    break;
                case CUPTI_METRIC_VALUE_KIND_INT64:
                    printf("Metric %s = %lld\n", metricName,
                           (long long) metricValue.metricValueInt64);
                    break;
                case CUPTI_METRIC_VALUE_KIND_PERCENT:
                    printf("Metric %s = %f%%\n", metricName,
                           metricValue.metricValuePercent);
                    break;
                case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
                    printf("Metric %s = %llu bytes/sec\n", metricName,
                           (unsigned long long) metricValue.metricValueThroughput);
                    break;
                case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
                    printf("Metric %s = utilization level %u\n", metricName,
                           (unsigned int) metricValue.metricValueUtilizationLevel);
                    break;
                default:
                {
                    char buffer[1024];
                    sprintf(buffer, "error: unknown value kind\n");
                    throw std::runtime_error(std::string(buffer));
                }
            }
        }

        CUPTI_CALL(cuptiUnsubscribe(subscriber));
    }

private:
    CUpti_SubscriberHandle subscriber;
    CUcontext              context     = 0;
    CUdevice               device      = 0;
    int                    deviceNum   = 0;
    int                    deviceCount = 0;
    char                   deviceName[32];
    const char*            metricName = "ipc";
    CUpti_MetricID         metricId;
    CUpti_EventGroupSets*  passData;
    metric_data_t          metricData;
    unsigned int           pass = 0;
    CUpti_MetricValue      metricValue;
};

//--------------------------------------------------------------------------------------//

struct profiler
{
    using strvec_t     = std::vector<string_t>;
    using event_val_t  = impl::kernel_data_t::event_val_t;
    using metric_val_t = impl::kernel_data_t::metric_val_t;
    using results_t    = std::vector<result>;

    profiler(const strvec_t& events, const strvec_t& metrics, const int device_num = 0)
    : m_device_num(device_num)
    , m_num_metrics(metrics.size())
    , m_num_events(events.size())
    , m_event_names(events)
    , m_metric_names(metrics)
    {
        init();
    }

    ~profiler()
    {
        // Disable callback and unsubscribe
        CUPTI_CALL(cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
        CUPTI_CALL(cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
        CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
        CUDA_DRIVER_API_CALL(cuCtxDestroy(m_context));
    }

    void init()
    {
        int device_count = 0;
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&device_count));
        if(device_count == 0)
        {
            fprintf(stderr, "There is no device supporting CUDA.\n");
            return;
        }

        m_metric_ids.resize(m_num_metrics);
        m_event_ids.resize(m_num_events);

        // Init device, context and setup callback
        CUDA_DRIVER_API_CALL(cuDeviceGet(&m_device, m_device_num));
        CUDA_DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
        CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                                  (CUpti_CallbackFunc) impl::get_value_callback,
                                  &m_kernel_data));
        CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
        CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

        CUpti_MetricID* metric_ids =
            (CUpti_MetricID*) calloc(sizeof(CUpti_MetricID), m_num_metrics);
        for(int i = 0; i < m_num_metrics; ++i)
        {
            CUPTI_CALL(cuptiMetricGetIdFromName(m_device, m_metric_names[i].c_str(),
                                                &metric_ids[i]));
        }

        CUpti_EventID* event_ids =
            (CUpti_EventID*) calloc(sizeof(CUpti_EventID), m_num_events);
        for(int i = 0; i < m_num_events; ++i)
        {
            CUPTI_CALL(cuptiEventGetIdFromName(m_device, m_event_names[i].c_str(),
                                               &event_ids[i]));
        }

        if(m_num_metrics > 0)
        {
            CUPTI_CALL(cuptiMetricCreateEventGroupSets(
                m_context, sizeof(CUpti_MetricID) * m_num_metrics, metric_ids,
                &m_metric_pass_data));
            m_metric_passes = m_metric_pass_data->numSets;

            std::copy(metric_ids, metric_ids + m_num_metrics, m_metric_ids.begin());
        }
        if(m_num_events > 0)
        {
            CUPTI_CALL(cuptiEventGroupSetsCreate(m_context,
                                                 sizeof(CUpti_EventID) * m_num_events,
                                                 event_ids, &m_event_pass_data));
            m_event_passes = m_event_pass_data->numSets;

            std::copy(event_ids, event_ids + m_num_events, m_event_ids.begin());
        }

        _LOG("# Metric Passes: %d\n", m_metric_passes);
        _LOG("# Event Passes: %d\n", m_event_passes);

        assert((m_metric_passes + m_event_passes) > 0);

        impl::kernel_data_t dummy_data;
        const char*         dummy_kernel_name = "^^ DUMMY ^^";
        dummy_data.m_name                     = dummy_kernel_name;
        dummy_data.m_metric_passes            = m_metric_passes;
        dummy_data.m_event_passes             = m_event_passes;
        dummy_data.m_device                   = m_device;
        dummy_data.m_total_passes             = m_metric_passes + m_event_passes;
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

        m_kernel_data[dummy_kernel_name] = dummy_data;
        free(metric_ids);
        free(event_ids);
    }

    int passes() { return m_metric_passes + m_event_passes; }

    void start() {}

    void stop()
    {
        const char* dummy_kernel_name = "^^ DUMMY ^^";
        for(auto& k : m_kernel_data)
        {
            auto& data = k.second.m_pass_data;

            if(k.first == dummy_kernel_name)
                continue;

            int total_events = 0;
            for(int i = 0; i < m_metric_passes; ++i)
            {
                // total_events += m_metric_data[i].num_events;
                total_events += data[i].num_events;
            }
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

            for(int i = 0; i < m_num_metrics; ++i)
            {
                CUptiResult _status = cuptiMetricGetValue(
                    m_device, m_metric_ids[i], total_events * sizeof(CUpti_EventID),
                    event_ids, total_events * sizeof(uint64_t), event_values, 0,
                    &metric_value);
                if(_status != CUPTI_SUCCESS)
                {
                    fprintf(stderr, "Metric value retrieval failed for metric %s\n",
                            m_metric_names[i].c_str());
                    return;
                }
                k.second.m_metric_values.push_back(metric_value);
                k.second.m_metric_tuples.push_back(
                    impl::get_metric_tuple(m_metric_ids[i], metric_value));
            }

            delete[] event_ids;
            delete[] event_values;

            map_t<CUpti_EventID, uint64_t> event_map;
            for(int i = m_metric_passes; i < (m_metric_passes + m_event_passes); ++i)
            {
                for(uint32_t j = 0; j < data[i].num_events; ++j)
                {
                    event_map[data[i].event_ids[j]] = data[i].event_values[j];
                }
            }

            for(int i = 0; i < m_num_events; ++i)
            {
                k.second.m_event_values.push_back(event_map[m_event_ids[i]]);
            }
        }
    }

    void print_event_values(std::ostream& os, bool print_names = true,
                            const char* kernel_separator = "\n")
    {
        using ull_t                   = unsigned long long;
        const char* dummy_kernel_name = "^^ DUMMY ^^";

        std::stringstream ss;
        for(auto const& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_name)
                continue;

            if(m_num_events <= 0)
                return;

            for(int i = 0; i < m_num_events; ++i)
            {
                if(print_names)
                    ss << "(" << m_event_names.at(i) << ","
                       << (ull_t) m_kernel_data.at(k.first).m_event_values.at(i) << ") ";
                else
                    ss << (ull_t) m_kernel_data.at(k.first).m_event_values.at(i) << " ";
                if(i + 1 < m_num_events)
                    ss << kernel_separator;
            }
        }
        os << ss.str() << std::endl;
    }

    void print_metric_values(std::ostream& os, bool print_names = true,
                             const char* kernel_separator = "\n")
    {
        if(m_num_metrics <= 0)
            return;

        const char*       dummy_kernel_name = "^^ DUMMY ^^";
        std::stringstream ss;
        for(auto const& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_name)
                continue;

            for(int i = 0; i < m_num_metrics; ++i)
            {
                if(print_names)
                    ss << "(" << m_metric_names[i] << ",";

                impl::print_metric(m_metric_ids[i],
                                   m_kernel_data[k.first].m_metric_values[i], ss);

                if(print_names)
                    ss << ") ";
                else
                    ss << " ";
                if(i + 1 < m_num_events)
                    ss << kernel_separator;
            }
        }
        os << ss.str() << std::endl;
    }

    results_t get_events_and_metrics(const std::vector<std::string>& labels)
    {
        results_t kern_data(labels.size());
        if(m_num_events <= 0 && m_num_metrics <= 0)
            return kern_data;

        using ull_t                   = unsigned long long;
        const char* dummy_kernel_name = "^^ DUMMY ^^";
        auto        get_label_index   = [&](const std::string& key) -> int64_t {
            for(int64_t i = 0; i < static_cast<int64_t>(labels.size()); ++i)
            {
                if(key == labels[i])
                    return i;
            }
            return -1;
        };

        for(auto const& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_name)
                continue;
            for(int i = 0; i < m_num_events; ++i)
            {
                std::string evt_name  = m_event_names[i].c_str();
                auto        label_idx = get_label_index(evt_name);
                if(label_idx < 0)
                    continue;
                auto value = static_cast<uint64_t>(
                    m_kernel_data.find(k.first)->second.m_event_values[i]);
                kern_data[label_idx] = result(evt_name, value, true);
            }

            for(int i = 0; i < m_num_metrics; ++i)
            {
                std::string met_name  = m_metric_names[i].c_str();
                auto        label_idx = get_label_index(met_name);
                if(label_idx < 0)
                    continue;
                auto idx = impl::get_metric_tuple_index(m_metric_ids[i]);
                auto ret = impl::get_metric_tuple(
                    m_metric_ids[i],
                    m_kernel_data.find(k.first)->second.m_metric_values[i]);
                switch(idx)
                {
                    case 0:
                        kern_data[label_idx] = result(met_name, std::get<0>(ret), false);
                        break;
                    case 1:
                        kern_data[label_idx] = result(met_name, std::get<1>(ret), false);
                        break;
                    case 2:
                        kern_data[label_idx] = result(met_name, std::get<2>(ret), false);
                        break;
                }
            }
        }
        return kern_data;
    }

    template <typename stream>
    void print_events_and_metrics(stream& s, bool print_names = true,
                                  const char* kernel_separator = "; ")
    {
        if(m_num_events <= 0 && m_num_metrics <= 0)
            return;

        using ull_t                   = unsigned long long;
        const char* dummy_kernel_name = "^^ DUMMY ^^";
        for(auto const& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_name)
                continue;

            for(int i = 0; i < m_num_events; ++i)
            {
                if(print_names)
                    s << "(" << m_event_names[i] << ","
                      << (ull_t) m_kernel_data[k.first].m_event_values[i] << ") ";
                else
                    s << (ull_t) m_kernel_data[k.first].m_event_values[i] << " ";
            }

            for(int i = 0; i < m_num_metrics; ++i)
            {
                if(print_names)
                    s << "(" << m_metric_names[i] << ",";

                impl::print_metric(m_metric_ids[i],
                                   m_kernel_data[k.first].m_metric_values[i], s);

                if(print_names)
                    s << ") ";
                else
                    s << " ";
            }

            s << kernel_separator;
        }
        printf("\n");
    }

    std::vector<string_t> get_kernel_names()
    {
        if(m_kernel_names.size() == 0)
        {
            const char* dummy_kernel_name = "^^ DUMMY ^^";
            for(auto const& k : m_kernel_data)
            {
                if(k.first == dummy_kernel_name)
                    continue;
                m_kernel_names.push_back(k.first);
            }
        }
        return m_kernel_names;
    }

    event_val_t get_event_values(const char* kernel_name)
    {
        if(m_num_events > 0)
            return m_kernel_data[kernel_name].m_event_values;
        else
            return event_val_t{};
    }

    metric_val_t get_metric_values(const char* kernel_name)
    {
        if(m_num_metrics > 0)
            return m_kernel_data[kernel_name].m_metric_values;
        else
            return metric_val_t{};
    }

    const strvec_t& get_event_names() const { return m_event_names; }
    const strvec_t& get_metric_names() const { return m_metric_names; }

private:
    int m_device_num;
    int m_num_metrics;
    int m_num_events;
    int m_num_kernels   = 0;
    int m_metric_passes = 0;
    int m_event_passes  = 0;

    strvec_t                    m_event_names;
    strvec_t                    m_metric_names;
    std::vector<CUpti_MetricID> m_metric_ids;
    std::vector<CUpti_EventID>  m_event_ids;

    CUcontext              m_context;
    CUdevice               m_device;
    CUpti_SubscriberHandle m_subscriber;

    CUpti_EventGroupSets* m_metric_pass_data;
    CUpti_EventGroupSets* m_event_pass_data;

    // Kernel-specific (indexed by name) trace data
    map_t<string_t, impl::kernel_data_t> m_kernel_data;
    strvec_t                             m_kernel_names;
};

//--------------------------------------------------------------------------------------//

#if !defined(__CUPTI_PROFILER_NAME_SHORT)
#    define __CUPTI_PROFILER_NAME_SHORT 128
#endif

//--------------------------------------------------------------------------------------//

static std::vector<string_t>
available_metrics(CUdevice device)
{
    std::vector<string_t> metric_names;
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

static std::vector<string_t>
available_events(CUdevice device)
{
    std::vector<string_t> event_names;
    uint32_t              numDomains  = 0;
    uint32_t              num_events  = 0;
    uint32_t              totalEvents = 0;
    size_t                size;
    CUpti_EventDomainID*  domainIdArray;
    CUpti_EventID*        eventIdArray;
    size_t                eventIdArraySize;
    char                  eventName[__CUPTI_PROFILER_NAME_SHORT];

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
