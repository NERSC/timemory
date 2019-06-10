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

#include "timemory/macros.hpp"
#include "timemory/utility.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(TIMEMORY_USE_CUPTI)
#    include <cuda.h>
#    include <cuda_runtime.h>
#    include <cupti.h>
#endif

//--------------------------------------------------------------------------------------//

#define DRIVER_API_CALL(apiFuncCall)                                                     \
    {                                                                                    \
        CUresult _status = apiFuncCall;                                                  \
        if(_status != CUDA_SUCCESS)                                                      \
        {                                                                                \
            fprintf(stderr, "%s:%d: error: function '%s' failed with error: %d.\n",      \
                    __FILE__, __LINE__, #apiFuncCall, _status);                          \
        }                                                                                \
    }

//--------------------------------------------------------------------------------------//

#define RUNTIME_API_CALL(apiFuncCall)                                                    \
    {                                                                                    \
        cudaError_t _status = apiFuncCall;                                               \
        if(_status != cudaSuccess)                                                       \
        {                                                                                \
            fprintf(stderr, "%s:%d: error: function '%s' failed with error: %s.\n",      \
                    __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));      \
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

#if defined(DEBUG)
template <typename... Args>
inline void
_LOG(const char* msg, Args&&... args)
{
    fprintf(stderr, "[Log]: ");
    fprintf(stderr, msg, std::forward<Args>(args)...);
    fprintf(stderr, "\n");
}

//--------------------------------------------------------------------------------------//

inline void
_LOG(const char* msg)
{
    fprintf(stderr, "[Log]: %s\n", msg);
}

//--------------------------------------------------------------------------------------//

template <typename... Args>
inline void
_DBG(const char* msg, Args&&... args)
{
    fprintf(stderr, msg, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

inline void
_DBG(const char* msg)
{
    fprintf(stderr, "%s", msg);
}
#else
#    define _LOG(...)                                                                    \
        {                                                                                \
        }
#    define _DBG(...)                                                                    \
        {                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace cupti
{
//--------------------------------------------------------------------------------------//

using string_t = std::string;

template <typename _Key, typename _Mapped>
using uomap = std::unordered_map<_Key, _Mapped>;

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

//--------------------------------------------------------------------------------------//
// data for the kernels
//
struct kernel_data_t
{
    using event_val_t  = std::vector<uint64_t>;
    using metric_val_t = std::vector<CUpti_MetricValue>;

    kernel_data_t() {}

    std::vector<pass_data_t> m_pass_data;
    string_t                 m_name;

    int      m_metric_passes = 0;
    int      m_event_passes  = 0;
    int      m_current_pass  = 0;
    int      m_total_passes  = 0;
    CUdevice m_device;

    event_val_t  m_event_values;
    metric_val_t m_metric_values;

    kernel_data_t& operator+=(const kernel_data_t& rhs)
    {
        for(uint64_t i = 0; i < m_event_values.size(); ++i)
        {
            m_event_values[i] += rhs.m_event_values[i];
        }
        return *this;
    }

    kernel_data_t& operator-=(const kernel_data_t& rhs)
    {
        for(uint64_t i = 0; i < m_event_values.size(); ++i)
        {
            m_event_values[i] -= rhs.m_event_values[i];
        }
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
static CUPTIAPI void
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

#if defined(TIMEMORY_DEMANGLE)
    // lambda for demangling a string when delimiting
    auto _demangle = [](string_t _str) {
        auto _to_str = [](char* cstr) {
            std::stringstream ss;
            ss << cstr;
            return ss.str();
        };

        const int _max_len = 256;
        int       _ret     = 0;
        size_t    _len     = 0;
        char*     _buf     = new char[_max_len];
        char*     _demang  = abi::__cxa_demangle(_str.c_str(), _buf, &_len, &_ret);

        if(_len > 0 && _len < _max_len)
            _buf[_len] = '\0';

        if(_ret == 0 && (_len > 0 || _demang))
            _str = _to_str((_len > 0) ? _buf : _demang);

        delete[] _buf;
        return _str;
    };
    auto _demangled_name = _demangle(_current_kernel_name);
    _LOG("  Demangled name: %s", _demangled_name.c_str());
    auto current_kernel_name = _demangled_name.c_str();
#else
    auto current_kernel_name = _current_kernel_name;
#endif

    using uomap_type  = uomap<string_t, kernel_data_t>;
    auto* kernel_data = static_cast<uomap_type*>(userdata);
    _LOG("... begin callback for %s...\n", current_kernel_name);

    if(cbInfo->callbackSite == CUPTI_API_ENTER)
    {
        _LOG("CUPTI_API_ENTER... starting callback for %s...\n", current_kernel_name);
        // If this is kernel name hasn't been seen before
        if(kernel_data->count(current_kernel_name) == 0)
        {
            _LOG("New kernel encountered: %s", current_kernel_name);

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

            _LOG("Current pass for %s: %d", current_kernel_name, current_pass);

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
        _LOG("CUPTI_API_ENTER... ending callback for %s...\n", current_kernel_name);
    }
    else if(cbInfo->callbackSite == CUPTI_API_EXIT)
    {
        _LOG("CUPTI_API_EXIT... starting callback for %s...\n", current_kernel_name);
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
        _LOG("CUPTI_API_EXIT... ending callback for %s...\n", current_kernel_name);
    }
    _LOG("... ending callback for %s...\n", current_kernel_name);
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

}  // namespace impl

//--------------------------------------------------------------------------------------//

struct profiler
{
    typedef std::vector<string_t> strvec_t;
    using event_val_t  = impl::kernel_data_t::event_val_t;
    using metric_val_t = impl::kernel_data_t::metric_val_t;

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
        DRIVER_API_CALL(cuCtxDestroy(m_context));
        CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
    }

    void init()
    {
        int device_count = 0;
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        DRIVER_API_CALL(cuDeviceGetCount(&device_count));
        if(device_count == 0)
        {
            fprintf(stderr, "There is no device supporting CUDA.\n");
            return;
        }

        m_metric_ids.resize(m_num_metrics);
        m_event_ids.resize(m_num_events);

        // Init device, context and setup callback
        DRIVER_API_CALL(cuDeviceGet(&m_device, m_device_num));
        DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
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
            }

            delete[] event_ids;
            delete[] event_values;

            uomap<CUpti_EventID, uint64_t> event_map;
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

        // Disable callback and unsubscribe
        CUPTI_CALL(cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
        CUPTI_CALL(cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                       CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
        CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
    }

    template <typename stream>
    void print_event_values(stream& s, bool print_names = true,
                            const char* kernel_separator = "; ")
    {
        using ull_t                   = unsigned long long;
        const char* dummy_kernel_name = "^^ DUMMY ^^";

        for(auto const& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_name)
                continue;

            if(m_num_events <= 0)
                return;

            for(int i = 0; i < m_num_events; ++i)
            {
                if(print_names)
                    s << "(" << m_event_names[i] << ","
                      << (ull_t) m_kernel_data[k.first].m_event_values[i] << ") ";
                else
                    s << (ull_t) m_kernel_data[k.first].m_event_values[i] << " ";
            }
            s << kernel_separator;
        }
        printf("\n");
    }

    template <typename stream>
    void print_metric_values(stream& s, bool print_names = true,
                             const char* kernel_separator = "; ")
    {
        if(m_num_metrics <= 0)
            return;

        const char* dummy_kernel_name = "^^ DUMMY ^^";
        for(auto const& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_name)
                continue;

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

    const strvec_t&             m_event_names;
    const strvec_t&             m_metric_names;
    std::vector<CUpti_MetricID> m_metric_ids;
    std::vector<CUpti_EventID>  m_event_ids;

    CUcontext              m_context;
    CUdevice               m_device;
    CUpti_SubscriberHandle m_subscriber;

    CUpti_EventGroupSets* m_metric_pass_data;
    CUpti_EventGroupSets* m_event_pass_data;

    // Kernel-specific (indexed by name) trace data
    uomap<string_t, impl::kernel_data_t> m_kernel_data;
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
