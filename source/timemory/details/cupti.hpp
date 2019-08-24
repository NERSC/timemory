// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/device.hpp"
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

#if defined(TIMEMORY_USE_CUPTI)
#    include <cupti.h>

//--------------------------------------------------------------------------------------//

#    if !defined(CUDA_DRIVER_API_CALL)
#        define CUDA_DRIVER_API_CALL(apiFuncCall)                                        \
            {                                                                            \
                CUresult _status = apiFuncCall;                                          \
                if(_status != CUDA_SUCCESS)                                              \
                {                                                                        \
                    fprintf(stderr,                                                      \
                            "%s:%d: error: function '%s' failed with error: %d.\n",      \
                            __FILE__, __LINE__, #apiFuncCall, _status);                  \
                }                                                                        \
            }
#    endif

//--------------------------------------------------------------------------------------//

#    if !defined(CUPTI_CALL)
#        define CUPTI_CALL(call)                                                         \
            {                                                                            \
                CUptiResult _status = call;                                              \
                if(_status != CUPTI_SUCCESS)                                             \
                {                                                                        \
                    const char* errstr;                                                  \
                    cuptiGetResultString(_status, &errstr);                              \
                    fprintf(stderr,                                                      \
                            "%s:%d: error: function '%s' failed with error: %s.\n",      \
                            __FILE__, __LINE__, #call, errstr);                          \
                }                                                                        \
            }
#    endif

#else  // !defined(TIMEMORY_USE_CUPTI)

typedef enum
{
    CUPTI_ACTIVITY_KIND_INVALID                       = 0,
    CUPTI_ACTIVITY_KIND_MEMCPY                        = 1,
    CUPTI_ACTIVITY_KIND_MEMSET                        = 2,
    CUPTI_ACTIVITY_KIND_KERNEL                        = 3,
    CUPTI_ACTIVITY_KIND_DRIVER                        = 4,
    CUPTI_ACTIVITY_KIND_RUNTIME                       = 5,
    CUPTI_ACTIVITY_KIND_EVENT                         = 6,
    CUPTI_ACTIVITY_KIND_METRIC                        = 7,
    CUPTI_ACTIVITY_KIND_DEVICE                        = 8,
    CUPTI_ACTIVITY_KIND_CONTEXT                       = 9,
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL             = 10,
    CUPTI_ACTIVITY_KIND_NAME                          = 11,
    CUPTI_ACTIVITY_KIND_MARKER                        = 12,
    CUPTI_ACTIVITY_KIND_MARKER_DATA                   = 13,
    CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR                = 14,
    CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS                 = 15,
    CUPTI_ACTIVITY_KIND_BRANCH                        = 16,
    CUPTI_ACTIVITY_KIND_OVERHEAD                      = 17,
    CUPTI_ACTIVITY_KIND_CDP_KERNEL                    = 18,
    CUPTI_ACTIVITY_KIND_PREEMPTION                    = 19,
    CUPTI_ACTIVITY_KIND_ENVIRONMENT                   = 20,
    CUPTI_ACTIVITY_KIND_EVENT_INSTANCE                = 21,
    CUPTI_ACTIVITY_KIND_MEMCPY2                       = 22,
    CUPTI_ACTIVITY_KIND_METRIC_INSTANCE               = 23,
    CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION         = 24,
    CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER        = 25,
    CUPTI_ACTIVITY_KIND_FUNCTION                      = 26,
    CUPTI_ACTIVITY_KIND_MODULE                        = 27,
    CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE              = 28,
    CUPTI_ACTIVITY_KIND_SHARED_ACCESS                 = 29,
    CUPTI_ACTIVITY_KIND_PC_SAMPLING                   = 30,
    CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO       = 31,
    CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION       = 32,
    CUPTI_ACTIVITY_KIND_OPENACC_DATA                  = 33,
    CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH                = 34,
    CUPTI_ACTIVITY_KIND_OPENACC_OTHER                 = 35,
    CUPTI_ACTIVITY_KIND_CUDA_EVENT                    = 36,
    CUPTI_ACTIVITY_KIND_STREAM                        = 37,
    CUPTI_ACTIVITY_KIND_SYNCHRONIZATION               = 38,
    CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION          = 39,
    CUPTI_ACTIVITY_KIND_NVLINK                        = 40,
    CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT           = 41,
    CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE  = 42,
    CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC          = 43,
    CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE = 44,
    CUPTI_ACTIVITY_KIND_MEMORY                        = 45,
    CUPTI_ACTIVITY_KIND_PCIE                          = 46,
    CUPTI_ACTIVITY_KIND_OPENMP                        = 47,
    CUPTI_ACTIVITY_KIND_COUNT                         = 48,
    CUPTI_ACTIVITY_KIND_FORCE_INT                     = 0x7fffffff
} _tim_activity_kind_t;

#    if !defined(CUDA_DRIVER_API_CALL)
#        define CUDA_DRIVER_API_CALL(...)
#    endif
#    if !defined(CUPTI_CALL)
#        define CUPTI_CALL(...)
#    endif

#endif  // !defined(TIMEMORY_USE_CUPTI

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

#if defined(TIMEMORY_USE_CUPTI)
using metric_value_t  = CUpti_MetricValue;
using activity_kind_t = CUpti_ActivityKind;
using context_t       = CUcontext;
using device_t        = CUdevice;
#else
typedef enum
{
    CUPTI_METRIC_VALUE_UTILIZATION_IDLE      = 0,
    CUPTI_METRIC_VALUE_UTILIZATION_LOW       = 2,
    CUPTI_METRIC_VALUE_UTILIZATION_MID       = 5,
    CUPTI_METRIC_VALUE_UTILIZATION_HIGH      = 8,
    CUPTI_METRIC_VALUE_UTILIZATION_MAX       = 10,
    CUPTI_METRIC_VALUE_UTILIZATION_FORCE_INT = 0x7fffffff
} MetricValueUtilizationLevel;

typedef union {
    double                      metricValueDouble;
    uint64_t                    metricValueUint64;
    int64_t                     metricValueInt64;
    double                      metricValuePercent;
    uint64_t                    metricValueThroughput;
    MetricValueUtilizationLevel metricValueUtilizationLevel;
} metric_value_t;

struct _CUcontext
{
};

struct _CUdevice
{
};

using context_t       = _CUcontext;
using device_t        = _CUdevice;
using activity_kind_t = _tim_activity_kind_t;

inline strvec_t available_metrics(device_t) { return strvec_t{}; }
inline strvec_t available_events(device_t) { return strvec_t{}; }

#endif

//--------------------------------------------------------------------------------------//

inline void
init_driver()
{
#if defined(TIMEMORY_USE_CUPTI)
    static std::atomic<short> _once;
    if(_once++ > 0)
        return;

    if(settings::debug())
        printf("[cupt::%s]> Initializing driver...\n", __FUNCTION__);
    CUDA_DRIVER_API_CALL(cuInit(0));
#endif
}

//--------------------------------------------------------------------------------------//

inline device_t
get_device(int devid)
{
    device_t device;
#if defined(TIMEMORY_USE_CUPTI)
    CUDA_DRIVER_API_CALL(cuDeviceGet(&device, devid));
#else
    consume_parameters(devid);
#endif
    return device;
}

//--------------------------------------------------------------------------------------//

namespace data
{
//--------------------------------------------------------------------------------------//

union metric_u {
    int64_t  integer_v = 0;
    uint64_t unsigned_integer_v;
    double   percent_v;
    double   floating_v;
    int64_t  throughput_v;
    int64_t  utilization_v;
};

//--------------------------------------------------------------------------------------//

struct metric
{
    metric_u data  = {};
    uint64_t count = 1;
    uint64_t index = 0;
};

//--------------------------------------------------------------------------------------//

struct unsigned_integer
{
    using type                      = uint64_t;
    static constexpr uint64_t index = 0;

    static type& get(metric& obj) { return obj.data.unsigned_integer_v; }
    static type  cget(const metric& obj) { return obj.data.unsigned_integer_v; }
    static void  set(metric& obj, metric_value_t& value)
    {
        get(obj)  = value.metricValueUint64;
        obj.index = index;
    }
    static void set(metric& obj, const type& value)
    {
        get(obj)  = value;
        obj.index = index;
    }
    static void print(std::ostream& os, const metric& obj) { os << cget(obj); }
    static type get_data(const metric& obj) { return obj.data.unsigned_integer_v; }
};

//--------------------------------------------------------------------------------------//

struct integer
{
    using type                      = int64_t;
    static constexpr uint64_t index = 1;

    static type& get(metric& obj) { return obj.data.integer_v; }
    static type  cget(const metric& obj) { return obj.data.integer_v; }
    static void  set(metric& obj, metric_value_t& value)
    {
        get(obj)  = value.metricValueInt64;
        obj.index = index;
    }
    static void print(std::ostream& os, const metric& obj) { os << cget(obj); }
    static type get_data(const metric& obj) { return obj.data.integer_v; }
};

//--------------------------------------------------------------------------------------//

struct percent
{
    using type                      = double;
    static constexpr uint64_t index = 2;

    static type& get(metric& obj) { return obj.data.percent_v; }
    static type  cget(const metric& obj) { return obj.data.percent_v; }
    static void  set(metric& obj, metric_value_t& value)
    {
        get(obj)  = value.metricValuePercent;
        obj.index = index;
    }
    static void print(std::ostream& os, const metric& obj)
    {
        auto val = cget(obj);
        val /= obj.count;
        os << val << " %";
    }
    static type get_data(const metric& obj) { return obj.data.percent_v / obj.count; }
};

//--------------------------------------------------------------------------------------//

struct floating
{
    using type                      = double;
    static constexpr uint64_t index = 3;

    static type& get(metric& obj) { return obj.data.floating_v; }
    static type  cget(const metric& obj) { return obj.data.floating_v; }
    static void  set(metric& obj, metric_value_t& value)
    {
        get(obj)  = value.metricValueDouble;
        obj.index = index;
    }
    static void print(std::ostream& os, const metric& obj) { os << cget(obj); }
    static type get_data(const metric& obj) { return obj.data.floating_v; }
};

//--------------------------------------------------------------------------------------//

struct throughput
{
    using type                      = int64_t;
    static constexpr uint64_t index = 4;

    static type& get(metric& obj) { return obj.data.throughput_v; }
    static type  cget(const metric& obj) { return obj.data.throughput_v; }
    static void  set(metric& obj, metric_value_t& value)
    {
        get(obj)  = value.metricValueThroughput;
        obj.index = index;
    }
    static void print(std::ostream& os, const metric& obj) { os << cget(obj); }
    static type get_data(const metric& obj) { return obj.data.throughput_v; }
};

//--------------------------------------------------------------------------------------//

struct utilization
{
    using type                      = int64_t;
    static constexpr uint64_t index = 5;

    static type& get(metric& obj) { return obj.data.utilization_v; }
    static type  cget(const metric& obj) { return obj.data.utilization_v; }
    static void  set(metric& obj, metric_value_t& value)
    {
        get(obj)  = value.metricValueUtilizationLevel;
        obj.index = index;
    }
    static void print(std::ostream& os, const metric& obj) { os << cget(obj); }
    static type get_data(const metric& obj) { return obj.data.utilization_v; }
};

//--------------------------------------------------------------------------------------//

using data_types =
    std::tuple<integer, unsigned_integer, percent, floating, throughput, utilization>;

}  // namespace data

//--------------------------------------------------------------------------------------//

namespace impl
{
//--------------------------------------------------------------------------------------//

using data_metric_t = data::metric;

//--------------------------------------------------------------------------------------//
// Generic tuple operations
//
template <typename _Ret, typename _Tp, typename... _Types,
          typename std::enable_if<(sizeof...(_Types) == 0), int>::type = 0>
void
_get(_Ret& val, const data_metric_t& lhs)
{
    if(lhs.index == _Tp::index)
        val = static_cast<_Ret>(_Tp::get_data(lhs));
}

//--------------------------------------------------------------------------------------//

template <typename _Ret, typename _Tp, typename... _Types,
          typename std::enable_if<(sizeof...(_Types) > 0), int>::type = 0>
void
_get(_Ret& val, const data_metric_t& lhs)
{
    if(lhs.index == _Tp::index)
        val = static_cast<_Ret>(_Tp::get_data(lhs));
    else
        _get<_Ret, _Types...>(val, lhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Types,
          typename std::enable_if<(sizeof...(_Types) == 0), int>::type = 0>
void
_print(std::ostream& os, const data_metric_t& lhs)
{
    if(lhs.index == _Tp::index)
        _Tp::print(os, lhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Types,
          typename std::enable_if<(sizeof...(_Types) > 0), int>::type = 0>
void
_print(std::ostream& os, const data_metric_t& lhs)
{
    if(lhs.index == _Tp::index)
        _Tp::print(os, lhs);
    else
        _print<_Types...>(os, lhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Types,
          typename std::enable_if<(sizeof...(_Types) == 0), int>::type = 0>
void
_plus(data_metric_t& lhs, const data_metric_t& rhs)
{
    if(lhs.index == _Tp::index)
        _Tp::get(lhs) += _Tp::cget(rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Types,
          typename std::enable_if<(sizeof...(_Types) > 0), int>::type = 0>
void
_plus(data_metric_t& lhs, const data_metric_t& rhs)
{
    if(lhs.index == _Tp::index)
        _Tp::get(lhs) += _Tp::cget(rhs);
    else
        _plus<_Types...>(lhs, rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Types,
          typename std::enable_if<(sizeof...(_Types) == 0), int>::type = 0>
void
_minus(data_metric_t& lhs, const data_metric_t& rhs)
{
    if(lhs.index == _Tp::index)
        _Tp::get(lhs) -= _Tp::cget(rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Types,
          typename std::enable_if<(sizeof...(_Types) > 0), int>::type = 0>
void
_minus(data_metric_t& lhs, const data_metric_t& rhs)
{
    if(lhs.index == _Tp::index)
        _Tp::get(lhs) -= _Tp::cget(rhs);
    else
        _minus<_Types...>(lhs, rhs);
}

//--------------------------------------------------------------------------------------//

inline data_metric_t&
operator+=(data_metric_t& lhs, const data_metric_t& rhs)
{
    impl::_plus<data::unsigned_integer, data::integer, data::percent, data::floating,
                data::throughput, data::utilization>(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

inline data_metric_t&
operator-=(data_metric_t& lhs, const data_metric_t& rhs)
{
    impl::_minus<data::unsigned_integer, data::integer, data::percent, data::floating,
                 data::throughput, data::utilization>(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

}  // namespace impl

//--------------------------------------------------------------------------------------//

inline void
print(std::ostream& os, const impl::data_metric_t& lhs)
{
    impl::_print<data::unsigned_integer, data::integer, data::percent, data::floating,
                 data::throughput, data::utilization>(os, lhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline _Tp
get(const impl::data_metric_t& lhs)
{
    _Tp value = _Tp(0.0);
    impl::_get<_Tp, data::unsigned_integer, data::integer, data::percent, data::floating,
               data::throughput, data::utilization>(value, lhs);
    return value;
}

//--------------------------------------------------------------------------------------//

inline void
plus(impl::data_metric_t& lhs, const impl::data_metric_t& rhs)
{
    impl::_plus<data::unsigned_integer, data::integer, data::percent, data::floating,
                data::throughput, data::utilization>(lhs, rhs);
    lhs.count += rhs.count;
}

//--------------------------------------------------------------------------------------//

inline void
minus(impl::data_metric_t& lhs, const impl::data_metric_t& rhs)
{
    impl::_minus<data::unsigned_integer, data::integer, data::percent, data::floating,
                 data::throughput, data::utilization>(lhs, rhs);
    lhs.count -= rhs.count;
}

//--------------------------------------------------------------------------------------//

struct result
{
    using data_t = data::metric;

    bool        is_event_value = true;
    int         index          = 0;
    std::string name           = "unk";
    data_t      data;

    result()              = default;
    ~result()             = default;
    result(const result&) = default;
    result(result&&)      = default;
    result& operator=(const result&) = default;
    result& operator=(result&&) = default;

    explicit result(const std::string& _name, const data_t& _data, bool _is = true)
    : is_event_value(_is)
    , index(_data.index)
    , name(_name)
    , data(_data)
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const result& obj)
    {
        std::stringstream ss;
        ss << std::setprecision(2);
        print(ss, obj.data);
        ss << " " << obj.name;
        os << ss.str();
        return os;
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(serializer::make_nvp("is_event_value", is_event_value),
           serializer::make_nvp("index", index), serializer::make_nvp("name", name),
           serializer::make_nvp("data", get<double>(data)));
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
        plus(data, rhs.data);
        return *this;
    }

    result& operator-=(const result& rhs)
    {
        if(name == "unk")
            return operator=(rhs);
        minus(data, rhs.data);
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

#if !defined(TIMEMORY_USE_CUPTI)
namespace impl
{
struct kernel_data_t
{
    using event_val_t  = std::vector<uint64_t>;
    using metric_val_t = std::vector<metric_value_t>;
    using metric_tup_t = std::vector<data_metric_t>;

    kernel_data_t()                     = default;
    kernel_data_t(const kernel_data_t&) = default;
    kernel_data_t(kernel_data_t&&)      = default;
    kernel_data_t& operator=(const kernel_data_t&) = default;
    kernel_data_t& operator=(kernel_data_t&&) = default;

    void                 clone(const kernel_data_t&) {}
    kernel_data_t&       operator+=(const kernel_data_t&) { return *this; }
    kernel_data_t&       operator-=(const kernel_data_t&) { return *this; }
    friend kernel_data_t operator+(const kernel_data_t& lhs, const kernel_data_t& rhs)
    {
        return kernel_data_t(lhs) += rhs;
    }
    friend kernel_data_t operator-(const kernel_data_t& lhs, const kernel_data_t& rhs)
    {
        return kernel_data_t(lhs) -= rhs;
    }
};
}
struct profiler
{
    using event_val_t  = impl::kernel_data_t::event_val_t;
    using metric_val_t = impl::kernel_data_t::metric_val_t;
    using results_t    = std::vector<result>;
    using ulong_t      = unsigned long long;

    profiler(const strvec_t&, const strvec_t&, const int = 0) {}

    ~profiler() {}

    int  passes() { return 0; }
    void start() {}

    void      stop() {}
    void      print_event_values(std::ostream&, bool = true, const char* = ";\n") {}
    void      print_metric_values(std::ostream&, bool = true, const char* = ";\n") {}
    void      print_events_and_metrics(std::ostream&, bool = true, const char* = ";\n") {}
    results_t get_events_and_metrics(const std::vector<std::string>&)
    {
        return results_t{};
    }
    strvec_t        get_kernel_names() { return strvec_t{}; }
    event_val_t     get_event_values(const char*) { return event_val_t{}; }
    metric_val_t    get_metric_values(const char*) { return metric_val_t{}; }
    const strvec_t& get_event_names() const { return m_event_names; }
    const strvec_t& get_metric_names() const { return m_metric_names; }

private:
    bool     m_is_running    = false;
    int      m_device_num    = 0;
    int      m_metric_passes = 0;
    int      m_event_passes  = 0;
    strvec_t m_event_names;
    strvec_t m_metric_names;
};

#endif

namespace activity
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
class receiver
{
public:
    using mutex_type     = std::mutex;
    using lock_type      = std::unique_lock<mutex_type>;
    using data_type      = std::list<_Tp*>;
    using size_type      = typename data_type::size_type;
    using iterator       = typename data_type::iterator;
    using const_iterator = typename data_type::const_iterator;

    // value_type is not used but keeping it here bc of plans to use something
    // similar later for a "thread_value" to distinguish traditional additions
    // vs. addition from other threads when they merge
    /*
    template <typename _Up>
    class value_type
    {
    public:
        value_type() = delete;

        explicit value_type(const _Up& _val)
        : m_value(_val)
        {
        }
        explicit value_type(_Up&& val)
        : m_value(std::move(val))
        {
        }

        value_type(const value_type&) = default;
        value_type(value_type&&)      = default;

        value_type& operator=(const value_type&) = default;
        value_type& operator=(value_type&&) = default;

        _Up&       get() { return m_value; }
        const _Up& get() const { return m_value; }

    private:
        _Up m_value;
    };
    */

    struct lock_holder
    {
        lock_holder(receiver& _recv)
        : m_recv(_recv)
        {
            m_lock                 = new lock_type(m_recv.m_mutex);
            m_recv.m_external_hold = true;
        }

        ~lock_holder()
        {
            m_recv.m_external_hold = false;
            delete m_lock;
        }

    private:
        receiver&  m_recv;
        lock_type* m_lock = nullptr;
    };

    friend struct lock_holder;
    using holder_type = lock_holder;

public:
    receiver()  = default;
    ~receiver() = default;

    receiver(const receiver&) = delete;
    receiver(receiver&&)      = default;

    receiver& operator=(const receiver&) = delete;
    receiver& operator=(receiver&&) = default;

    inline void insert(_Tp* obj)
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        if(find(obj) == end())
            m_data.insert(m_data.end(), obj);
    }

    inline void remove(_Tp* obj)
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        for(auto itr = m_data.begin(); itr != m_data.end(); ++itr)
        {
            if(*itr == obj)
            {
                m_data.erase(itr);
                return;
            }
        }
    }

    inline void clear()
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        m_data.clear();
    }

    inline void reset()
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        m_data.clear();
        m_elapsed = 0;
    }

    inline size_type size() const
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        return m_data.size();
    }

    inline bool empty() const
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        return m_data.empty();
    }

    uint64_t get()
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        return m_elapsed;
    }

    // this operator is invoked from the CUPTI callback which implements an external
    // hold to make sure all the buffers get added before allowing other operations
    // such as insert/remove to proceed
    template <typename _Up>
    inline receiver& operator+=(const _Up& rhs)
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock() && !m_external_hold)
            lk.lock();
        m_elapsed += rhs;
        return *this;
    }

    template <typename _Up>
    inline receiver& operator-=(const _Up& rhs)
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        m_elapsed -= rhs;
        return *this;
    }

protected:
    iterator begin() { return m_data.begin(); }
    iterator end() { return m_data.end(); }

    const_iterator begin() const { return m_data.begin(); }
    const_iterator end() const { return m_data.end(); }

    iterator find(const _Tp* obj)
    {
        for(auto itr = begin(); itr != end(); ++itr)
        {
            if(*itr == obj)
                return itr;
        }
        return end();
    }

    const_iterator find(const _Tp* obj) const
    {
        for(auto itr = begin(); itr != end(); ++itr)
        {
            if(*itr == obj)
                return itr;
        }
        return end();
    }

protected:
    mutable bool       m_external_hold = false;
    uint64_t           m_elapsed       = 0;
    mutable mutex_type m_mutex;
    data_type          m_data;
};

//--------------------------------------------------------------------------------------//

// Timestamp at trace initialization time. Used to normalized other timestamps
inline uint64_t&
start_timestamp()
{
    static uint64_t _instance = 0;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline receiver<_Tp>&
get_receiver()
{
    static receiver<_Tp> _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline size_t&
get_buffer_size()
{
    static size_t deviceValue = 0;

#if defined(TIMEMORY_USE_CUPTI)
    if(deviceValue == 0)
    {
        size_t attrValueSize = sizeof(size_t);
        // get the buffer size and increase
        CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                             &attrValueSize, &deviceValue));
        if(settings::verbose() > 1 || settings::debug())
            printf("[tim::cupti::activity::%s]> %s = %llu\n", __FUNCTION__,
                   "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE",
                   (long long unsigned) deviceValue);
        deviceValue *= 2;
    }
#endif
    return deviceValue;
}

//--------------------------------------------------------------------------------------//

inline size_t&
get_buffer_pool_limit()
{
    static size_t poolValue = 0;

#if defined(TIMEMORY_USE_CUPTI)
    if(poolValue == 0)
    {
        size_t attrValueSize = sizeof(size_t);
        // get the buffer pool limit and increase
        CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                             &attrValueSize, &poolValue));
        if(settings::verbose() > 1 || settings::debug())
            printf("[tim::cupti::activity::%s]> %s = %llu\n", __FUNCTION__,
                   "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT",
                   (long long unsigned) poolValue);
        poolValue *= 2;
    }
#endif
    return poolValue;
}

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_USE_CUPTI)

template <typename _Tp>
inline void
initialize_trace(const std::vector<activity_kind_t>&)
{
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
finalize_trace(const std::vector<activity_kind_t>&)
{
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
start_trace(_Tp*)
{
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
stop_trace(_Tp*)
{
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
request_buffer(uint8_t**, size_t*, size_t*)
{
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
buffer_completed(context_t, uint32_t, uint8_t*, size_t, size_t)
{
}

//--------------------------------------------------------------------------------------//

inline void set_device_buffers(size_t, size_t) {}

#else

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
initialize_trace(const std::vector<activity_kind_t>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
finalize_trace(const std::vector<activity_kind_t>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
start_trace(_Tp*);

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
stop_trace(_Tp*);

//--------------------------------------------------------------------------------------//

template <typename _Tp>
static void CUPTIAPI
            request_buffer(uint8_t**, size_t*, size_t*);

//--------------------------------------------------------------------------------------//

template <typename _Tp>
static void CUPTIAPI
            buffer_completed(CUcontext, uint32_t, uint8_t*, size_t, size_t);

//--------------------------------------------------------------------------------------//

inline void set_device_buffers(size_t, size_t);

#endif

//--------------------------------------------------------------------------------------//

inline void
enable(const std::vector<activity_kind_t>& _kind_types)
{
#if defined(TIMEMORY_USE_CUPTI)
    // Device activity record is created when CUDA initializes, so we
    // want to enable it before cuInit() or any CUDA runtime call.
    for(const auto& itr : _kind_types)
    {
        if(settings::debug())
            std::cout << "[cupti::activity::enable]> Enabling " << static_cast<int>(itr)
                      << "..." << std::endl;
        auto ret = cuptiActivityEnable(itr);
        CUPTI_CALL(ret);
    }
#else
    consume_parameters(_kind_types);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
disable(const std::vector<activity_kind_t>& _kind_types)
{
#if defined(TIMEMORY_USE_CUPTI)
    // Device activity record is created when CUDA initializes, so we
    // want to enable it before cuInit() or any CUDA runtime call.
    for(const auto& itr : _kind_types)
    {
        auto ret = cuptiActivityDisable(itr);
        CUPTI_CALL(ret);
    }
#else
    consume_parameters(_kind_types);
#endif
}

//--------------------------------------------------------------------------------------//

template <typename _ReqBuffFunc, typename _BuffCompFunc>
inline void
register_callbacks(_ReqBuffFunc _reqbuff, _BuffCompFunc _buffcomp)
{
#if defined(TIMEMORY_USE_CUPTI)
    // typedef void (*BuffFunc)(uint8_t**, size_t*, size_t*);
    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(_reqbuff, _buffcomp));
#else
    consume_parameters(_reqbuff, _buffcomp);
#endif
}

//--------------------------------------------------------------------------------------//

inline uint64_t
get_timestamp()
{
    uint64_t _start = 0;
#if defined(TIMEMORY_USE_CUPTI)
    CUPTI_CALL(cuptiGetTimestamp(&_start));
#endif
    return _start;
}

//--------------------------------------------------------------------------------------//

}  // namespace activity

//--------------------------------------------------------------------------------------//

}  // namespace cupti

//--------------------------------------------------------------------------------------//

}  // namespace tim
