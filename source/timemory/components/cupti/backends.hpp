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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/**
 * \file timemory/components/cupti/backends.hpp
 * \brief Implementation of the cupti functions/utilities
 */

#pragma once

#include "timemory/backends/types/cupti.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace stl
{
//
inline cupti::profiler::results_t&
operator+=(cupti::profiler::results_t& lhs, const cupti::profiler::results_t& rhs)
{
    assert(lhs.size() == rhs.size());
    const auto _N = ::std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        lhs[i] += rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//

inline cupti::profiler::results_t
operator-(const cupti::profiler::results_t& lhs, const cupti::profiler::results_t& rhs)
{
    assert(lhs.size() == rhs.size());
    cupti::profiler::results_t tmp = lhs;
    const auto                 _N  = ::std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        tmp[i] -= rhs[i];
    return tmp;
}
//
}  // namespace stl
}  // namespace tim
//
//--------------------------------------------------------------------------------------//

/*
#include "timemory/backends/device.hpp"
#include "timemory/components/cuda/backends.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include "timemory/components/cupti/backends/activity.hpp"
#include "timemory/components/cupti/backends/callback.hpp"
#include "timemory/components/cupti/backends/common.hpp"
#include "timemory/components/cupti/backends/enums.hpp"
#include "timemory/components/cupti/backends/macros.hpp"

//======================================================================================//
//
namespace tim
{
namespace cupti
{
//======================================================================================//
//
//                              Callback Profiler
//
//======================================================================================//

namespace callback
{
//--------------------------------------------------------------------------------------//

struct profiler
{
    using ulong_t          = unsigned long long;
    using kernel_map_t     = cupti::map_t<uint64_t, impl::kernel_data_t>;
    using metric_id_vector = std::vector<metric_id_t>;
    using event_id_vector  = std::vector<event_id_t>;
    using event_val_t      = impl::kernel_data_t::event_val_t;
    using metric_val_t     = impl::kernel_data_t::metric_val_t;
    using results_t        = std::vector<result>;
    using kernel_pair_t    = std::pair<std::string, results_t>;
    using kernel_results_t = std::vector<kernel_pair_t>;

    profiler(const strvec_t& events, const strvec_t& metrics, const int device_num = 0)
    : m_device_num(device_num)
    , m_event_names(events)
    , m_metric_names(metrics)
    {
#if defined(TIMEMORY_USE_CUPTI)
        int device_count = 0;

        // sync before starting
        cuda::device_sync();

        TIMEMORY_CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGetCount(&device_count));

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
        TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&m_device, m_device_num));
        // TIMEMORY_CUDA_DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
        TIMEMORY_CUDA_DRIVER_API_CALL(cuDevicePrimaryCtxRetain(&m_context, m_device));

        if(m_metric_names.size() > 0)
        {
            for(size_t i = 0; i < m_metric_names.size(); ++i)
                TIMEMORY_CUPTI_CALL(cuptiMetricGetIdFromName(m_device,
m_metric_names[i].c_str(), &m_metric_ids[i]));
            TIMEMORY_CUPTI_CALL(cuptiMetricCreateEventGroupSets(
                m_context, sizeof(metric_id_t) * m_metric_names.size(),
                m_metric_ids.data(), &m_metric_pass_data));
            m_metric_passes = m_metric_pass_data->numSets;
        }

        if(m_event_names.size() > 0)
        {
            for(size_t i = 0; i < m_event_names.size(); ++i)
                TIMEMORY_CUPTI_CALL(cuptiEventGetIdFromName(m_device,
m_event_names[i].c_str(), &m_event_ids[i]));
            TIMEMORY_CUPTI_CALL(cuptiEventGroupSetsCreate(m_context,
                                                 sizeof(event_id_t) * m_event_ids.size(),
                                                 m_event_ids.data(), &m_event_pass_data));
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
                TIMEMORY_CUPTI_CALL(cuptiEventGroupGetAttribute(
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
                TIMEMORY_CUPTI_CALL(cuptiEventGroupGetAttribute(
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
        static std::atomic<int> _once(0);
        if(_once++ == 0)
        {
            // store the initial data so that warmup does not show up
            kernel_map_t _initial = m_kernel_data;
            start();
            device::params<device::default_device> p(1, 1, 0, 0);
            device::launch(p, impl::warmup<int>);
            stop();
            // revert to original
            m_kernel_data = _initial;
        }
#endif
    }

    ~profiler()
    {
#if defined(TIMEMORY_USE_CUPTI)
        if(m_is_running)
            stop();

        cuptiDisableKernelReplayMode(m_context);
        TIMEMORY_CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
        // TIMEMORY_CUDA_DRIVER_API_CALL(cuDevicePrimaryCtxRelease(m_device));
#endif
    }

    profiler(const profiler&) = delete;
    profiler(profiler&&)      = delete;
    profiler& operator=(const profiler&) = delete;
    profiler& operator=(profiler&&) = delete;

    int passes() { return m_metric_passes + m_event_passes; }

private:
    using mutex_t = std::recursive_mutex;
    using lock_t  = std::unique_lock<mutex_t>;

    static mutex_t& get_mutex()
    {
        static mutex_t _instance;
        return _instance;
    }

    static bool& is_subscribed()
    {
        static bool _instance = false;
        return _instance;
    }

    static std::atomic<int32_t>& subscribed_count()
    {
        static std::atomic<int32_t> _instance(0);
        return _instance;
    }

public:
    void start()
    {
#if defined(TIMEMORY_USE_CUPTI)
        if(m_is_running)
            return;
        m_is_running = true;

        cuda::device_sync();

        if(!is_subscribed())
        {
            is_subscribed() = true;
            TIMEMORY_CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                                      (CUpti_CallbackFunc) impl::get_value_callback,
                                      &m_kernel_data));

            TIMEMORY_CUPTI_CALL(cuptiEnableCallback(1, m_subscriber,
CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
            TIMEMORY_CUPTI_CALL(
                cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

            TIMEMORY_CUPTI_CALL(
                cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000));
            TIMEMORY_CUPTI_CALL(cuptiEnableCallback(
                1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000));
        }

        cuda::device_sync();
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_CUPTI)
        if(!m_is_running)
            return;
        m_is_running = false;

        using event_id_map_t = std::map<event_id_t, uint64_t>;

        cuda::device_sync();

        if(is_subscribed())
        {
            is_subscribed() = false;
            // Disable callback and unsubscribe
            TIMEMORY_CUPTI_CALL(cuptiEnableCallback(0, m_subscriber,
CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
            TIMEMORY_CUPTI_CALL(
                cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
            TIMEMORY_CUPTI_CALL(
                cuptiEnableCallback(0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000));
            TIMEMORY_CUPTI_CALL(cuptiEnableCallback(
                0, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000));
            TIMEMORY_CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
        }

        for(auto& k : m_kernel_data)
        {
            if(k.first == impl::dummy_kernel_id)
                continue;

            auto& data         = k.second.m_pass_data;
            int   total_events = 0;
            for(int i = 0; i < m_metric_passes; ++i)
                total_events += data[i].num_events;

            metric_value_t metric_value;
            event_id_t*    event_ids    = new event_id_t[total_events];
            uint64_t*      event_values = new uint64_t[total_events];

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
                    m_device, m_metric_ids[i], total_events * sizeof(event_id_t),
                    event_ids, total_events * sizeof(uint64_t), event_values, 0,
                    &metric_value);
                if(_status != CUPTI_SUCCESS)
                {
                    char buff[512];
                    sprintf(buff, "Metric value retrieval failed for metric %s\n",
                            m_metric_names[i].c_str());
                    throw std::runtime_error(buff);
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
#endif
    }

    //----------------------------------------------------------------------------------//

    void print_event_values(std::ostream& os, bool print_names = true,
                            const char* kernel_separator = ";\n")
    {
#if defined(TIMEMORY_USE_CUPTI)
        std::stringstream ss;
        for(auto& kitr : m_kernel_data)
        {
            if(kitr.first == impl::dummy_kernel_id)
                continue;
            ss << kitr.second.m_name << " : ";
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
#else
        consume_parameters(os, print_names, kernel_separator);
#endif
    }

    //----------------------------------------------------------------------------------//

    void print_metric_values(std::ostream& os, bool print_names = true,
                             const char* kernel_separator = ";\n")
    {
#if defined(TIMEMORY_USE_CUPTI)
        std::stringstream ss;
        for(auto& kitr : m_kernel_data)
        {
            if(kitr.first == impl::dummy_kernel_id)
                continue;
            ss << kitr.second.m_name << " : ";
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
#else
        consume_parameters(os, print_names, kernel_separator);
#endif
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

#if defined(TIMEMORY_USE_CUPTI)
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
                {
                    printf("[%s:'%s'@%i]> Skipping metric '%s'...\n", __FUNCTION__,
                           __FILE__, __LINE__, evt_name.c_str());
                    continue;
                }
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
                {
                    printf("[%s:'%s'@%i]> Skipping metric '%s'...\n", __FUNCTION__,
                           __FILE__, __LINE__, met_name.c_str());
                    continue;
                }
                auto ret =
                    impl::get_metric(m_metric_ids[i], kitr.second.m_metric_values[i]);
                result _result(met_name, ret, false);
                kern_data[label_idx] += _result;
            }
        }
#endif
        return kern_data;
    }

    //----------------------------------------------------------------------------------//

    kernel_results_t get_kernel_events_and_metrics(const std::vector<std::string>& labels)
    {
        if(settings::verbose() > 2 || settings::debug())
            print_events_and_metrics(std::cout);

        kernel_results_t kern_data;

#if defined(TIMEMORY_USE_CUPTI)
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
            results_t _tmp_data(labels.size());
            for(size_t i = 0; i < m_event_names.size(); ++i)
            {
                std::string evt_name  = m_event_names[i];
                auto        label_idx = get_label_index(evt_name);
                if(label_idx < 0)
                    continue;
                auto         value = static_cast<uint64_t>(kitr.second.m_event_values[i]);
                data::metric ret;
                data::unsigned_integer::set(ret, value);
                _tmp_data[label_idx] += result(evt_name, ret, true);
            }

            for(size_t i = 0; i < m_metric_names.size(); ++i)
            {
                std::string met_name  = m_metric_names[i].c_str();
                auto        label_idx = get_label_index(met_name);
                if(label_idx < 0)
                    continue;
                auto ret =
                    impl::get_metric(m_metric_ids[i], kitr.second.m_metric_values[i]);
                result _result(met_name, ret, false);
                _tmp_data[label_idx] += _result;
            }

            kern_data.push_back(kernel_pair_t{ kitr.second.m_name, _tmp_data });
        }
#else
        consume_parameters(labels);
#endif
        return kern_data;
    }

    strvec_t get_kernel_names()
    {
#if defined(TIMEMORY_USE_CUPTI)
        m_kernel_names.clear();
        for(auto& k : m_kernel_data)
        {
            if(k.first == impl::dummy_kernel_id)
                continue;
            m_kernel_names.push_back(k.second.m_name);
        }
#endif
        return m_kernel_names;
    }

    event_val_t get_event_values(const char* kernel_name)
    {
#if defined(TIMEMORY_USE_CUPTI)
        if(m_kernel_data.count(atol(kernel_name)) != 0)
            return m_kernel_data[atol(kernel_name)].m_event_values;
        else
        {
            for(auto& kitr : m_kernel_data)
            {
                if(kitr.second.m_name == std::string(kernel_name))
                    return kitr.second.m_event_values;
            }
        }
#else
        consume_parameters(kernel_name);
#endif
        return event_val_t{};
    }

    metric_val_t get_metric_values(const char* kernel_name)
    {
#if defined(TIMEMORY_USE_CUPTI)
        if(m_kernel_data.count(atol(kernel_name)) != 0)
            return m_kernel_data[atol(kernel_name)].m_metric_values;
        else
        {
            for(auto& kitr : m_kernel_data)
            {
                if(kitr.second.m_name == std::string(kernel_name))
                    return kitr.second.m_metric_values;
            }
        }
#else
        consume_parameters(kernel_name);
#endif
        return metric_val_t{};
    }

    const strvec_t& get_event_names() const { return m_event_names; }
    const strvec_t& get_metric_names() const { return m_metric_names; }

    bool                get_is_running() const { return m_is_running; }
    int                 get_device_num() const { return m_device_num; }
    context_t           get_context() const { return m_context; }
    device_t            get_device() const { return m_device; }
    subscriber_handle_t get_subscriber() const { return m_subscriber; }
    event_group_sets_t* get_metric_pass_data() const { return m_metric_pass_data; }
    event_group_sets_t* get_event_pass_data() const { return m_event_pass_data; }

private:
    bool                m_is_running    = false;
    int                 m_device_num    = 0;
    int                 m_metric_passes = 0;
    int                 m_event_passes  = 0;
    strvec_t            m_event_names   = {};
    strvec_t            m_metric_names  = {};
    metric_id_vector    m_metric_ids    = {};
    event_id_vector     m_event_ids     = {};
    context_t           m_context;
    device_t            m_device;
    subscriber_handle_t m_subscriber;
    event_group_sets_t* m_metric_pass_data = nullptr;
    event_group_sets_t* m_event_pass_data  = nullptr;
    kernel_map_t        m_kernel_data      = {};
    strvec_t            m_kernel_names     = {};
};

}  // namespace callback

//======================================================================================//

namespace activity
{
//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_USE_CUPTI)

inline void
initialize_trace(const std::vector<activity_kind_t>&)
{}

//--------------------------------------------------------------------------------------//

inline void
finalize_trace(const std::vector<activity_kind_t>&)
{}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
start_trace(_Tp*, bool = false)
{}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
stop_trace(_Tp*)
{}

//--------------------------------------------------------------------------------------//

inline void
request_buffer(uint8_t**, size_t*, size_t*)
{}

//--------------------------------------------------------------------------------------//

inline void
buffer_completed(context_t, uint32_t, uint8_t*, size_t, size_t)
{}

//--------------------------------------------------------------------------------------//

inline void set_device_buffers(size_t, size_t) {}

#else

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

inline int64_t
get_elapsed(CUpti_Activity* record)
{
#    define _CUPTI_CAST_RECORD(ToType, var) ToType* var = (ToType*) record

    switch(record->kind)
    {
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityMemcpy, obj);
            return obj->end - obj->start;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityMemset, obj);
            return obj->end - obj->start;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityKernel4, obj);
            return obj->end - obj->start;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityAPI, obj);
            return obj->end - obj->start;
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityOverhead, obj);
            return obj->end - obj->start;
        }
        case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityCdpKernel, obj);
            return obj->end - obj->start;
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
#    undef _CUPTI_CAST_RECORD
}

//--------------------------------------------------------------------------------------//

inline const char*
get_name(CUpti_Activity* record)
{
#    define _CUPTI_CAST_RECORD(ToType, var) ToType* var = (ToType*) record

    switch(record->kind)
    {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityKernel4, obj);
            return obj->name;
        }
        case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityCdpKernel, obj);
            return obj->name;
        }
        case CUPTI_ACTIVITY_KIND_NAME:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityName, obj);
            return obj->name;
        }
        case CUPTI_ACTIVITY_KIND_MARKER:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityMarker2, obj);
            return obj->name;
        }
        case CUPTI_ACTIVITY_KIND_DEVICE:
        {
            _CUPTI_CAST_RECORD(CUpti_ActivityDevice2, obj);
            return obj->name;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: { return "cudaMemcpy";
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: { return "cudaMemset";
        }
        case CUPTI_ACTIVITY_KIND_DRIVER: { return "cudaDriver";
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME: { return "cudaRuntime";
        }
        case CUPTI_ACTIVITY_KIND_CONTEXT: { return "cudaContext";
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD: { return "cuptiOverhead";
        }
        case CUPTI_ACTIVITY_KIND_MARKER_DATA:
        case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
        default: break;
    }
    return "";
#    undef _CUPTI_CAST_RECORD
}

//--------------------------------------------------------------------------------------//

inline const char*
get_kind_extra(CUpti_Activity* record)
{
    switch(record->kind)
    {
        case CUPTI_ACTIVITY_KIND_CONTEXT:
        {
            CUpti_ActivityContext* context = (CUpti_ActivityContext*) record;
            return compute_api_kind(
                (CUpti_ActivityComputeApiKind) context->computeApiKind);
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy* memcpy = (CUpti_ActivityMemcpy*) record;
            return memcpy_kind((CUpti_ActivityMemcpyKind) memcpy->copyKind);
        }
        case CUPTI_ACTIVITY_KIND_NAME:
        {
            CUpti_ActivityName* name = (CUpti_ActivityName*) record;
            return object_kind(name->objectKind);
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
        {
            CUpti_ActivityOverhead* overhead = (CUpti_ActivityOverhead*) record;
            return overhead_kind(overhead->overheadKind);
        }
        case CUPTI_ACTIVITY_KIND_MARKER_DATA:
        default: break;
    }
    return "";
}

//--------------------------------------------------------------------------------------//

inline void
print(CUpti_Activity* record)
{
    switch(record->kind)
    {
        case CUPTI_ACTIVITY_KIND_DEVICE:
        {
            CUpti_ActivityDevice2* device = (CUpti_ActivityDevice2*) record;
            printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, "
                   "size %u MB), "
                   "multiprocessors %u, clock %u MHz\n",
                   device->name, device->id, device->computeCapabilityMajor,
                   device->computeCapabilityMinor,
                   (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
                   (unsigned int) (device->globalMemorySize / 1024 / 1024),
                   device->numMultiprocessors,
                   (unsigned int) (device->coreClockRate / 1000));
            break;
        }
        case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
        {
            CUpti_ActivityDeviceAttribute* attribute =
                (CUpti_ActivityDeviceAttribute*) record;
            printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
                   attribute->attribute.cupti, attribute->deviceId,
                   (unsigned long long) attribute->value.vUint64);
            break;
        }
        case CUPTI_ACTIVITY_KIND_CONTEXT:
        {
            CUpti_ActivityContext* context = (CUpti_ActivityContext*) record;
            printf(
                "CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
                context->contextId, context->deviceId,
                compute_api_kind((CUpti_ActivityComputeApiKind) context->computeApiKind),
                (int) context->nullStreamId);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy* memcpy = (CUpti_ActivityMemcpy*) record;
            printf(
                "MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation "
                "%u/r%u\n",
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
                "%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation "
                "%u\n",
                kindString, kernel->name,
                (unsigned long long) (kernel->start - start_timestamp()),
                (unsigned long long) (kernel->end - start_timestamp()), kernel->deviceId,
                kernel->contextId, kernel->streamId, kernel->correlationId);
            printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, "
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
                "DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
                api->cbid, (unsigned long long) (api->start - start_timestamp()),
                (unsigned long long) (api->end - start_timestamp()), api->processId,
                api->threadId, api->correlationId);
            break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
            CUpti_ActivityAPI* api = (CUpti_ActivityAPI*) record;
            printf(
                "RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
                api->cbid, (unsigned long long) (api->start - start_timestamp()),
                (unsigned long long) (api->end - start_timestamp()), api->processId,
                api->threadId, api->correlationId);
            break;
        }
        case CUPTI_ACTIVITY_KIND_NAME:
        {
            CUpti_ActivityName* name = (CUpti_ActivityName*) record;
            switch(name->objectKind)
            {
                case CUPTI_ACTIVITY_OBJECT_CONTEXT:
                    printf("NAME  %s %u %s id %u, name %s\n",
                           object_kind(name->objectKind),
                           object_kind_id(name->objectKind, &name->objectId),
                           object_kind(CUPTI_ACTIVITY_OBJECT_DEVICE),
                           object_kind_id(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
                           name->name);
                    break;
                case CUPTI_ACTIVITY_OBJECT_STREAM:
                    printf("NAME %s %u %s %u %s id %u, name %s\n",
                           object_kind(name->objectKind),
                           object_kind_id(name->objectKind, &name->objectId),
                           object_kind(CUPTI_ACTIVITY_OBJECT_CONTEXT),
                           object_kind_id(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
                           object_kind(CUPTI_ACTIVITY_OBJECT_DEVICE),
                           object_kind_id(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
                           name->name);
                    break;
                default:
                    printf("NAME %s id %u, name %s\n", object_kind(name->objectKind),
                           object_kind_id(name->objectKind, &name->objectId), name->name);
                    break;
            }
            break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER:
        {
            CUpti_ActivityMarker2* marker = (CUpti_ActivityMarker2*) record;
            printf("MARKER id %u [ %llu ], name %s, domain %s\n", marker->id,
                   (unsigned long long) marker->timestamp, marker->name, marker->domain);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER_DATA:
        {
            CUpti_ActivityMarkerData* marker = (CUpti_ActivityMarkerData*) record;
            printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
                   marker->id, marker->color, marker->category,
                   (unsigned long long) marker->payload.metricValueUint64,
                   marker->payload.metricValueDouble);
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
        default:
            fprintf(stderr,
                    "[cupti::activity::%s]> Warning!! Unknown activity record: %i\n",
                    __FUNCTION__, (int) record->kind);
            break;
    }
}

//--------------------------------------------------------------------------------------//

static void CUPTIAPI
request_buffer(uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
    uint8_t* bfr = (uint8_t*) malloc(TIMEMORY_CUPTI_BUFFER_SIZE +
TIMEMORY_CUPTI_ALIGN_SIZE); if(bfr == nullptr)
    {
        unsigned long long sz = TIMEMORY_CUPTI_BUFFER_SIZE + TIMEMORY_CUPTI_ALIGN_SIZE;
        fprintf(stderr, "[%s:%s:%i]> malloc unable to allocate %llu bytes\n",
                __FUNCTION__, __FILE__, __LINE__, sz);
        throw std::bad_alloc();
    }

    *size = TIMEMORY_CUPTI_BUFFER_SIZE;
    *buffer = TIMEMORY_CUPTI_ALIGN_BUFFER(bfr, TIMEMORY_CUPTI_ALIGN_SIZE);
    *maxNumRecords = 0;
}

//--------------------------------------------------------------------------------------//

static void CUPTIAPI
buffer_completed(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
                 size_t validSize)
{
    CUptiResult status;
    CUpti_Activity* record = nullptr;

    // obtain lock to keep data from being removed during update
    using lock_type = typename receiver::holder_type;
    auto& _receiver = get_receiver();

    if(validSize > 0)
    {
        do
        {
            lock_type lk(_receiver);
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if(status == CUPTI_SUCCESS)
            {
                using name_pair_t = std::tuple<std::string, int64_t>;
                if(settings::verbose() > 3 || settings::debug())
                    print(record);
                std::string _name = get_name(record);
                auto _time = get_elapsed(record);
                auto _extra = get_kind_extra(record);
                std::stringstream ss;
                if(std::strlen(_extra) > 0)
                {
                    ss << _name << "_" << _extra;
                    _name = ss.str();
                }
                auto _name_len = _name.length();
                switch(record->kind)
                {
                    case CUPTI_ACTIVITY_KIND_OVERHEAD:
                        _receiver -= name_pair_t{ _name, _time };
                        break;
                    case CUPTI_ACTIVITY_KIND_KERNEL:
                    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                    case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
                        _receiver += name_pair_t{ _name, _time };
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMCPY:
                        _receiver += name_pair_t{ _name, _time };
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMSET:
                        _receiver += name_pair_t{ _name, _time };
                        break;
                    case CUPTI_ACTIVITY_KIND_RUNTIME:
                        _receiver += name_pair_t{ _name, _time };
                        break;
                    case CUPTI_ACTIVITY_KIND_NAME:
                    case CUPTI_ACTIVITY_KIND_DRIVER:
                    case CUPTI_ACTIVITY_KIND_CONTEXT:
                    case CUPTI_ACTIVITY_KIND_MARKER:
                    case CUPTI_ACTIVITY_KIND_DEVICE:
                    case CUPTI_ACTIVITY_KIND_MARKER_DATA:
                    case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
                    default:
                    {
                        if(_name_len > 0 && _time > 0)
                        {
                            _receiver += name_pair_t{ _name, _time };
                            break;
                        }
                        else if(_name_len == 0 && _time > 0)
                        {
                            std::stringstream _ss;
                            _ss << "CUPTI_ACTIVITY_KIND_ENUM_"
                                << static_cast<int>(record->kind);
                            _receiver += name_pair_t{ _ss.str(), _time };
                            break;
                        }
                    }
                }
            }
            else if(status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else
            {
                TIMEMORY_CUPTI_CALL(status);
            }
        } while(1);

        // report any records dropped from the queue
        size_t dropped = 0;
        TIMEMORY_CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if(dropped != 0)
        {
            printf("[tim::cupti::activity::%s]> Dropped %u activity records\n",
                   __FUNCTION__, (unsigned int) dropped);
        }
    }

    free(buffer);
}

//--------------------------------------------------------------------------------------//

inline void
set_device_buffers(size_t _buffer_size, size_t _pool_limit)
{
    size_t deviceValue = 0;
    size_t poolValue = 0;
    size_t attrValueSize = sizeof(size_t);

    // Get and set activity attributes.
    // Attributes can be set by the CUPTI client to change behavior of the activity
    // API. Some attributes require to be set before any CUDA context is created to be
    // effective, e.g. to be applied to all device buffer allocations (see
    // documentation).

    // get the buffer size and increase
    TIMEMORY_CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                         &attrValueSize, &deviceValue));

    // get the buffer pool limit and increase
    TIMEMORY_CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                         &attrValueSize, &poolValue));

    if(_buffer_size != deviceValue)
    {
        TIMEMORY_CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                             &attrValueSize, &_buffer_size));
        if(settings::verbose() > 1 || settings::debug())
            printf("[tim::cupti::activity::%s]> %s = %llu\n", __FUNCTION__,
                   "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE",
                   (long long unsigned) _buffer_size);
        get_buffer_size() = _buffer_size;
    }

    if(_pool_limit != poolValue)
    {
        TIMEMORY_CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                             &attrValueSize, &_pool_limit));
        if(settings::verbose() > 1 || settings::debug())
            printf("[tim::cupti::activity::%s]> %s = %llu\n", __FUNCTION__,
                   "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT",
                   (long long unsigned) _pool_limit);
        get_buffer_pool_limit() = _pool_limit;
    }
}

//--------------------------------------------------------------------------------------//

inline void
initialize_trace(const std::vector<activity_kind_t>& _kind_types)
{
    size_t f_buffer = get_env("TIMEMORY_CUPTI_DEVICE_BUFFER_SIZE", get_buffer_size());
    size_t f_pool_limit =
        get_env("TIMEMORY_CUPTI_DEVICE_BUFFER_POOL_LIMIT", get_buffer_pool_limit());

    enable(_kind_types);
    register_callbacks(request_buffer, buffer_completed);
    set_device_buffers(f_buffer, f_pool_limit);
    start_timestamp() = cupti::activity::get_timestamp();
    init_driver();
}

//--------------------------------------------------------------------------------------//

inline void
finalize_trace(const std::vector<activity_kind_t>& _kind_types)
{
    disable(_kind_types);
    get_receiver().clear();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
start_trace(_Tp* obj, bool flush)
{
    auto& _receiver = get_receiver();
    // clang-format off
    if(flush) { TIMEMORY_CUPTI_CALL(cuptiActivityFlushAll(0)); }
    // clang-format on
    _receiver.insert(obj);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
stop_trace(_Tp* obj)
{
    auto& _receiver = get_receiver();
    cuda::device_sync();
    TIMEMORY_CUPTI_CALL(cuptiActivityFlushAll(0));
    _receiver.remove(obj);
}

#endif

}  // namespace activity
}  // namespace cupti
}  // namespace tim
//
//======================================================================================//
*/
