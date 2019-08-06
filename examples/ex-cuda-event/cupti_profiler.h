#pragma once

#include <atomic>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cupti.h>

#define CUDA_DRIVER_API_CALL(apiFuncCall)                                                \
    do                                                                                   \
    {                                                                                    \
        CUresult _status = apiFuncCall;                                                  \
        if(_status != CUDA_SUCCESS)                                                      \
        {                                                                                \
            fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",         \
                    __FILE__, __LINE__, #apiFuncCall, _status);                          \
            exit(-1);                                                                    \
        }                                                                                \
    } while(0)

#define CUDA_RUNTIME_API_CALL(apiFuncCall)                                               \
    do                                                                                   \
    {                                                                                    \
        cudaError_t _status = apiFuncCall;                                               \
        if(_status != cudaSuccess)                                                       \
        {                                                                                \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",         \
                    __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));      \
            exit(-1);                                                                    \
        }                                                                                \
    } while(0)

#define CUPTI_CALL(call)                                                                 \
    do                                                                                   \
    {                                                                                    \
        CUptiResult _status = call;                                                      \
        if(_status != CUPTI_SUCCESS)                                                     \
        {                                                                                \
            const char* errstr;                                                          \
            cuptiGetResultString(_status, &errstr);                                      \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",         \
                    __FILE__, __LINE__, #call, errstr);                                  \
            exit(-1);                                                                    \
        }                                                                                \
    } while(0)

#ifdef DEBUG
template <typename... Args>
void
_LOG(const char* msg, Args&&... args)
{
    fprintf(stderr, "[Log]: ");
    fprintf(stderr, msg, args...);
    fprintf(stderr, "\n");
}
void
_LOG(const char* msg)
{
    fprintf(stderr, "[Log]: %s\n", msg);
}
template <typename... Args>
void
_DBG(const char* msg, Args&&... args)
{
    fprintf(stderr, msg, args...);
}
void
_DBG(const char* msg)
{
    fprintf(stderr, "%s", msg);
}
#else
#    define _LOG(...)
#    define _DBG(...)
#endif

namespace cupti_profiler
{
static uint64_t dummy_kernel_id = 0;

namespace detail
{
// Pass-specific data
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

struct kernel_data_t
{
    typedef std::vector<uint64_t>          event_val_t;
    typedef std::vector<CUpti_MetricValue> metric_val_t;

    kernel_data_t()
    : m_current_pass(0)
    {
    }

    std::vector<pass_data_t> m_pass_data;
    std::string              m_name;

    int      m_metric_passes;
    int      m_event_passes;
    int      m_current_pass;
    int      m_total_passes;
    CUdevice m_device;

    event_val_t  m_event_values;
    metric_val_t m_metric_values;
};

void CUPTIAPI
     get_value_callback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                        const CUpti_CallbackData* cbInfo)
{
    using kernel_data_t = detail::kernel_data_t;
    using map_type      = std::map<uint64_t, kernel_data_t>;
    static std::atomic<uint64_t> correlation_data;
    // This callback is enabled only for launch so we shouldn't see
    // anything else.
    if((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
       (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {
        char buf[512];
        sprintf(buf, "%s:%d: Unexpected cbid %d\n", __FILE__, __LINE__, cbid);
        throw std::runtime_error(buf);
    }

    // Skip execution if kernel name is NULL string
    // TODO: Make sure this is fine
    if(cbInfo->symbolName == nullptr)
    {
        _LOG("Empty kernel name string. Skipping...");
        return;
    }

    map_type* kernel_data = static_cast<map_type*>(userdata);

    kernel_data_t dummy = (*kernel_data)[dummy_kernel_id];

    // if enter, assign unique correlation data ID
    if(cbInfo->callbackSite == CUPTI_API_ENTER)
        *cbInfo->correlationData = ++correlation_data;

    // get the correlation data ID
    uint64_t corr_data = *cbInfo->correlationData;
    _LOG("[kern] %s\n", cbInfo->symbolName);

    char current_kernel_name[1024];
    sprintf(current_kernel_name, "kernel_%li_i", static_cast<long>(corr_data));

    if(cbInfo->callbackSite == CUPTI_API_ENTER)
    {
        _LOG("New kernel encountered: %s", current_kernel_name);
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
}

template <typename stream_t>
void
print_metric(CUpti_MetricID& id, CUpti_MetricValue& value, stream_t& s)
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
        default: std::cerr << "[error]: unknown value kind\n"; exit(-1);
    }
}

__global__ void
warmup()
{
}

}  // namespace detail

struct profiler
{
    typedef std::vector<std::string> strvec_t;
    using event_val_t  = detail::kernel_data_t::event_val_t;
    using metric_val_t = detail::kernel_data_t::metric_val_t;

    profiler(const strvec_t& events, const strvec_t& metrics, const int device_num = 0)
    : m_event_names(events)
    , m_metric_names(metrics)
    , m_device_num(device_num)
    , m_metric_passes(0)
    , m_event_passes(0)
    {
        int device_count = 0;

        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&device_count));
        if(device_count == 0)
        {
            fprintf(stderr, "There is no device supporting CUDA.\n");
            exit(1);
        }

        m_metric_ids.resize(metrics.size());
        m_event_ids.resize(events.size());

        // Init device, context and setup callback
        CUDA_DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
        CUDA_DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
        CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                                  (CUpti_CallbackFunc) detail::get_value_callback,
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

        detail::kernel_data_t dummy_data;
        dummy_data.m_name          = dummy_kernel_id;
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

        m_kernel_data[dummy_kernel_id] = dummy_data;

        cuptiEnableKernelReplayMode(m_context);
        detail::warmup<<<1, 1>>>();
    }

    ~profiler() {}

    int get_passes() { return m_metric_passes + m_event_passes; }

    void start() {}

    void stop()
    {
        cudaStreamSynchronize(0);
        for(auto& k : m_kernel_data)
        {
            auto& data = k.second.m_pass_data;

            if(k.first == dummy_kernel_id)
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

            for(int i = 0; i < m_metric_names.size(); ++i)
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

            std::map<CUpti_EventID, uint64_t> event_map;
            for(int i = m_metric_passes; i < (m_metric_passes + m_event_passes); ++i)
            {
                for(int j = 0; j < data[i].event_values.size(); ++j)
                {
                    event_map[data[i].event_ids[j]] = data[i].event_values[j];
                }
            }

            for(int i = 0; i < m_event_ids.size(); ++i)
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
                            const char* kernel_separator = ";\n")
    {
        using ull_t = unsigned long long;

        for(auto const& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_id)
                continue;

            for(int i = 0; i < m_event_names.size(); ++i)
            {
                if(print_names)
                    s << "  (" << m_event_names[i] << ", " << std::setw(6)
                      << (ull_t) k.second.m_event_values[i] << ") ";
                else
                    s << std::setw(6) << (ull_t) k.second.m_event_values[i] << " ";
            }
            s << kernel_separator;
        }
        printf("\n");
    }

    template <typename stream>
    void print_metric_values(stream& s, bool print_names = true,
                             const char* kernel_separator = ";\n")
    {
        for(auto& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_id)
                continue;

            for(int i = 0; i < m_metric_names.size(); ++i)
            {
                if(print_names)
                    s << "  (" << m_metric_names[i] << ", ";

                s << std::setw(6) << std::setprecision(2) << std::fixed;
                detail::print_metric(m_metric_ids[i], k.second.m_metric_values[i], s);

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
                                  const char* kernel_separator = ";\n")
    {
        using ull_t = unsigned long long;
        for(auto& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_id)
                continue;

            for(int i = 0; i < m_event_names.size(); ++i)
            {
                if(print_names)
                    s << "  (" << m_event_names[i] << ", "
                      << (ull_t) k.second.m_event_values[i] << ") ";
                else
                    s << (ull_t) k.second.m_event_values[i] << " ";
            }

            for(int i = 0; i < m_metric_names.size(); ++i)
            {
                if(print_names)
                    s << "  (" << m_metric_names[i] << ",";

                detail::print_metric(m_metric_ids[i], k.second.m_metric_values[i], s);

                if(print_names)
                    s << ") ";
                else
                    s << " ";
            }

            s << kernel_separator;
        }
        printf("\n");
    }

    std::vector<std::string> get_kernel_names()
    {
        m_kernel_names.clear();
        for(auto& k : m_kernel_data)
        {
            if(k.first == dummy_kernel_id)
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

private:
    int                         m_device_num;
    const strvec_t&             m_event_names;
    const strvec_t&             m_metric_names;
    std::vector<CUpti_MetricID> m_metric_ids;
    std::vector<CUpti_EventID>  m_event_ids;

    CUcontext              m_context;
    CUdevice               m_device;
    CUpti_SubscriberHandle m_subscriber;

    CUpti_EventGroupSets* m_metric_pass_data;
    CUpti_EventGroupSets* m_event_pass_data;

    int m_metric_passes, m_event_passes;
    // Kernel-specific (indexed by name) trace data
    std::map<uint64_t, detail::kernel_data_t> m_kernel_data;
    std::vector<std::string>                  m_kernel_names;
};

#ifndef __CUPTI_PROFILER_NAME_SHORT
#    define __CUPTI_PROFILER_NAME_SHORT 128
#endif

std::vector<std::string>
available_metrics(CUdevice device)
{
    std::vector<std::string> metric_names;
    uint32_t                 numMetric;
    size_t                   size;
    char                     metricName[__CUPTI_PROFILER_NAME_SHORT];
    CUpti_MetricValueKind    metricKind;
    CUpti_MetricID*          metricIdArray;

    CUPTI_CALL(cuptiDeviceGetNumMetrics(device, &numMetric));
    size          = sizeof(CUpti_MetricID) * numMetric;
    metricIdArray = (CUpti_MetricID*) malloc(size);
    if(NULL == metricIdArray)
    {
        printf("Memory could not be allocated for metric array");
        exit(-1);
    }

    CUPTI_CALL(cuptiDeviceEnumMetrics(device, &size, metricIdArray));

    for(int i = 0; i < numMetric; i++)
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

std::vector<std::string>
available_events(CUdevice device)
{
    std::vector<std::string> event_names;
    uint32_t                 numDomains = 0, numEvents = 0, totalEvents = 0;
    size_t                   size;
    CUpti_EventDomainID*     domainIdArray;
    CUpti_EventID*           eventIdArray;
    size_t                   eventIdArraySize;
    char                     eventName[__CUPTI_PROFILER_NAME_SHORT];

    CUPTI_CALL(cuptiDeviceGetNumEventDomains(device, &numDomains));
    size          = sizeof(CUpti_EventDomainID) * numDomains;
    domainIdArray = (CUpti_EventDomainID*) malloc(size);
    if(NULL == domainIdArray)
    {
        printf("Memory could not be allocated for domain array");
        exit(-1);
    }
    CUPTI_CALL(cuptiDeviceEnumEventDomains(device, &size, domainIdArray));

    for(int i = 0; i < numDomains; i++)
    {
        CUPTI_CALL(cuptiEventDomainGetNumEvents(domainIdArray[i], &numEvents));
        totalEvents += numEvents;
    }

    eventIdArraySize = sizeof(CUpti_EventID) * totalEvents;
    eventIdArray     = (CUpti_EventID*) malloc(eventIdArraySize);

    totalEvents = 0;
    for(int i = 0; i < numDomains; i++)
    {
        // Query num of events available in the domain
        CUPTI_CALL(cuptiEventDomainGetNumEvents(domainIdArray[i], &numEvents));
        size = numEvents * sizeof(CUpti_EventID);
        CUPTI_CALL(cuptiEventDomainEnumEvents(domainIdArray[i], &size,
                                              eventIdArray + totalEvents));
        totalEvents += numEvents;
    }

    for(int i = 0; i < totalEvents; i++)
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

}  // namespace cupti_profiler
