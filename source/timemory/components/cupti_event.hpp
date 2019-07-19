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
 * \headerfile cupti_event.hpp "timemory/cupti_event.hpp"
 * Provides implementation of CUPTI routines.
 *
 */

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/cupti.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          CUPTI component
//
//--------------------------------------------------------------------------------------//

struct cupti_event : public base<cupti_event, cupti::profiler::results_t>
{
    using size_type     = std::size_t;
    using string_t      = std::string;
    using kernel_data_t = cupti::result;
    using value_type    = cupti::profiler::results_t;
    using entry_type    = typename value_type::value_type;
    using base_type     = base<cupti_event, value_type>;
    using this_type     = cupti_event;
    using event_count   = static_counted_object<cupti_event>;
    // short-hard for vectors
    using strvec_t  = std::vector<string_t>;
    using intvec_t  = std::vector<int>;
    using profptr_t = std::shared_ptr<cupti::profiler>;
    using profvec_t = std::vector<profptr_t>;
    // function for setting device, metrics, and events to record
    using event_func_t  = std::function<strvec_t()>;
    using metric_func_t = std::function<strvec_t()>;
    using device_func_t = std::function<intvec_t()>;

    static const short                   precision = 3;
    static const short                   width     = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::dec | std::ios_base::showpoint;

    static event_func_t& get_event_setter()
    {
        static event_func_t _instance = []() { return strvec_t(); };
        return _instance;
    }

    static metric_func_t& get_metric_setter()
    {
        static metric_func_t _instance = []() { return strvec_t(); };
        return _instance;
    }

    static device_func_t& get_device_setter()
    {
        static device_func_t _instance = []() {
            intvec_t devices(tim::cuda::device_count(), 0);
            std::iota(devices.begin(), devices.end(), 0);
            return devices;
        };
        return _instance;
    }

    // size_type size() const { return (is_transient) ? accum.size() : value.size(); }

    cupti_event(const strvec_t& events  = get_event_setter()(),
                const strvec_t& metrics = get_metric_setter()(),
                const intvec_t& devices = get_device_setter()())
    {
        init(events, metrics, devices);
    }

    ~cupti_event() { clear(); }

    void clear()
    {
        // for(auto& itr : m_profilers)
        //    delete itr;
        m_profilers.clear();
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    // template <typename Archive>
    // void serialize(Archive& ar, const unsigned int)
    //{}

    static int64_t unit() { return 1; }
    // leave these empty
    static string_t label() { return ""; }
    static string_t descript() { return ""; }
    static string_t display_unit() { return ""; }

    static value_type record() { return value_type{}; }

    string_t compute_display() const
    {
        auto _compute_display = [&](std::ostream& os, const cupti::result& obj) {
            auto _idx   = obj.index;
            auto _label = obj.name;
            auto _prec  = base_type::get_precision();
            auto _width = base_type::get_width();
            auto _flags = base_type::get_format_flags();

            std::stringstream ss, ssv, ssi;
            ssv.setf(_flags);
            ssv << std::setw(_width) << std::setprecision(_prec);
            switch(_idx)
            {
                case 0: ssv << std::get<0>(obj.data); break;
                case 1: ssv << std::get<1>(obj.data); break;
                case 2: ssv << std::get<2>(obj.data); break;
                default: ssv << -1; break;
            }
            if(!_label.empty())
                ssi << " " << _label;
            ss << ssv.str() << ssi.str();
            os << ss.str();
        };

        const auto&       _data = (is_transient) ? accum : value;
        std::stringstream ss;
        for(size_type i = 0; i < _data.size(); ++i)
        {
            _compute_display(ss, _data[i]);
            if(i + 1 < _data.size())
                ss << ", ";
        }
        return ss.str();
    }

    template <typename _Tp>
    using array_t = std::vector<_Tp>;

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    array_t<string_t> label_array()
    {
        array_t<string_t> arr;
        auto              contains = [&](const string_t& entry) {
            return std::find(arr.begin(), arr.end(), entry) != arr.end();
        };
        auto insert = [&](const string_t& entry) {
            if(!contains(entry))
                arr.push_back(entry);
        };
        for(const auto& profiler : m_profilers)
        {
            for(const auto& itr : profiler->get_event_names())
                insert(itr);
            for(const auto& itr : profiler->get_metric_names())
                insert(itr);
        }
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    array_t<string_t> descript_array() { return label_array(); }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    array_t<string_t> display_unit_array()
    {
        return array_t<string_t>(m_profilers.size(), "");
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    array_t<int64_t> unit_array()
    {
        array_t<int64_t> arr;
        for(size_type i = 0; i < m_labels.size(); ++i)
            arr[i] = 1;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        set_started();
        for(size_type i = 0; i < m_profilers.size(); ++i)
            m_profilers[i]->start();
    }

    void stop()
    {
        value_type tmp;
        for(size_type i = 0; i < m_profilers.size(); ++i)
        {
            m_profilers[i]->stop();
            if(tmp.size() == 0)
            {
                tmp = m_profilers[i]->get_events_and_metrics(m_labels);
            } else if(tmp.size() == m_labels.size())
            {
                auto ret = m_profilers[i]->get_events_and_metrics(m_labels);
                for(size_t j = 0; j < m_labels.size(); ++j)
                    tmp[j] += ret[j];
            } else
            {
                fprintf(stderr, "Warning! mis-matched size in cupti_event::%s @ %s:%i\n",
                        __FUNCTION__, __FILE__, __LINE__);
            }
        }

        if(accum.size() == 0)
        {
            accum = tmp;
        } else
        {
            for(size_type i = 0; i < tmp.size(); ++i)
            {
                accum[i] += (tmp[i] - value[i]);
            }
        }
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < m_labels.size(); ++i)
            accum[i] += rhs.accum[i];
        for(size_type i = 0; i < m_labels.size(); ++i)
            value[i] += rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < m_labels.size(); ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < m_labels.size(); ++i)
            value[i] -= rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    value_type serialization() { return accum; }

    //----------------------------------------------------------------------------------//
    //

    /*
    cupti_event(const cupti_event& rhs)
    : base_type(rhs)
    , m_events(rhs.m_events)
    , m_metrics(rhs.m_metrics)
    , m_devices(rhs.m_devices)
    , m_labels(rhs.m_labels)
    {}

    cupti_event& operator=(const cupti_event& rhs)
    {
        if(this != &rhs)
        {
            base_type::operator=(rhs);
            m_events           = rhs.m_events;
            m_metrics          = rhs.m_metrics;
            m_devices          = rhs.m_devices;
            m_labels           = rhs.m_labels;
        }
        return *this;
    }
    */

    cupti_event(const cupti_event&) = default;
    cupti_event& operator=(const cupti_event&) = default;
    cupti_event(cupti_event&&)                 = default;
    cupti_event& operator=(cupti_event&&) = default;

private:
    profvec_t m_profilers;
    strvec_t  m_events;
    strvec_t  m_metrics;
    intvec_t  m_devices;
    strvec_t  m_labels;

    template <typename _Tp>
    struct writer
    {
        _Tp& obj;
        writer(_Tp& _obj)
        : obj(_obj)
        {}
        friend std::ostream& operator<<(std::ostream& os, const writer& obj)
        {
            for(size_type i = 0; i < obj.obj.size(); ++i)
            {
                os << obj.obj[i];
                if(i + 1 < obj.obj.size())
                    os << ", ";
            }
            return os;
        }
    };

    void init(const strvec_t& events, const strvec_t& metrics, const intvec_t& devices)
    {
        clear();
        for(const auto& device : devices)
            m_profilers.push_back(
                profptr_t(new cupti::profiler(events, metrics, device)));

        m_devices = devices;
        m_events  = events;
        m_metrics = metrics;
        m_labels  = label_array();
        value.resize(m_labels.size());
        for(size_type i = 0; i < m_labels.size(); ++i)
            value[i].name = m_labels[i];
        std::cout << "Devices : " << writer<intvec_t>(m_devices) << std::endl;
        std::cout << "Event   : " << writer<strvec_t>(m_events) << std::endl;
        std::cout << "Metrics : " << writer<strvec_t>(m_metrics) << std::endl;
        std::cout << "Labels  : " << writer<strvec_t>(m_labels) << std::endl;
    }
};

}  // namespace component
}  // namespace tim
