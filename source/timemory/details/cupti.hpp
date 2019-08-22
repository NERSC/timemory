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
#endif

namespace tim
{
namespace cupti
{
//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUPTI)
using metric_value_t  = CUpti_MetricValue;
using activity_kind_t = CUpti_ActivityKind;
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

using activity_kind_t = int;
#endif

//--------------------------------------------------------------------------------------//

using string_t = std::string;
template <typename _Key, typename _Mapped>
using map_t    = std::map<_Key, _Mapped>;
using strvec_t = std::vector<string_t>;

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

namespace activity
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
class receiver
{
public:
    using mutex_type       = std::mutex;
    using lock_type        = std::unique_lock<mutex_type>;
    using data_type        = std::list<_Tp*>;
    using size_type        = typename data_type::size_type;
    using iterator         = typename data_type::iterator;
    using const_iterator   = typename data_type::const_iterator;
    using kind_vector_type = std::vector<activity_kind_t>;
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

public:
    receiver()  = default;
    ~receiver() = default;

    receiver(const receiver&) = delete;
    receiver(receiver&&)      = default;

    receiver& operator=(const receiver&) = delete;
    receiver& operator=(receiver&&) = default;

    inline void insert(_Tp* obj)
    {
        lock_type lk(m_mutex);
        if(find(obj) == end())
            m_data.insert(m_data.end(), obj);
    }

    inline void remove(_Tp* obj)
    {
        lock_type lk(m_mutex);
        for(auto itr = m_data.begin(); itr != m_data.end(); ++itr)
        {
            if(*itr == obj)
            {
                m_data.erase(itr);
                return;
            }
        }
    }

    inline size_type size() const
    {
        lock_type lk(m_mutex);
        return m_data.size();
    }
    inline bool empty() const
    {
        lock_type lk(m_mutex);
        return m_data.empty();
    }

    template <typename _Up>
    inline receiver& operator+=(const _Up& rhs)
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        m_elapsed += rhs;
        // for(auto& itr : m_data)
        //    (*itr) += value_type<_Up>(rhs);
        return *this;
    }

    template <typename _Up>
    inline receiver& operator-=(const _Up& rhs)
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        m_elapsed -= rhs;
        // for(auto& itr : m_data)
        //    (*itr) -= value_type<_Up>(rhs);
        return *this;
    }

    mutex_type& get_mutex() { return m_mutex; }

    uint64_t get()
    {
        lock_type lk(m_mutex, std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        return m_elapsed;
    }

    void set_kinds(const kind_vector_type& _kinds) { m_kinds = _kinds; }
    const kind_vector_type& get_kinds() const { return m_kinds; }

protected:
    iterator begin() { return m_data.begin(); }
    iterator end() { return m_data.end(); }

    const_iterator begin() const { return m_data.begin(); }
    const_iterator end() const { return m_data.end(); }

    iterator find(const _Tp* obj)
    {
        lock_type lk(m_mutex);
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
    uint64_t           m_elapsed = 0;
    mutable mutex_type m_mutex;
    data_type          m_data;
    kind_vector_type   m_kinds;
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

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_USE_CUPTI)

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

#else

// declare

template <typename _Tp>
inline void
start_trace(_Tp*);

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
stop_trace(_Tp*);

#endif

//--------------------------------------------------------------------------------------//

}  // namespace activity

//--------------------------------------------------------------------------------------//

}  // namespace cupti

//--------------------------------------------------------------------------------------//

}  // namespace tim
