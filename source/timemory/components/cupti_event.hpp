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

struct cupti_event
: public base<cupti_event, cupti::profiler::results_t, policy::thread_init,
              policy::thread_finalize>
{
    using size_type     = std::size_t;
    using string_t      = std::string;
    using kernel_data_t = cupti::result;
    using value_type    = cupti::profiler::results_t;
    using entry_type    = typename value_type::value_type;
    using base_type =
        base<cupti_event, value_type, policy::thread_init, policy::thread_finalize>;
    using this_type   = cupti_event;
    using event_count = static_counted_object<cupti_event>;
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

    static const profvec_t& get_profilers() { return get_private_profilers(); }
    static const strvec_t&  get_events() { return get_private_events(); }
    static const strvec_t&  get_metrics() { return get_private_metrics(); }
    static const intvec_t&  get_devices() { return get_private_devices(); }
    static const strvec_t&  get_labels() { return get_private_labels(); }
    static void             invoke_thread_init() { init(); }
    static void             invoke_thread_finalize() {}

    // size_type size() const { return (is_transient) ? accum.size() : value.size(); }

    explicit cupti_event()
    {
        value.resize(m_labels.size());
        for(size_type i = 0; i < m_labels.size(); ++i)
            value[i].name = m_labels[i];
    }

    ~cupti_event() {}

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

    std::vector<double> get() const
    {
        std::vector<double> values;
        const auto&         _data = (is_transient) ? accum : value;
        for(auto itr : _data)
        {
            switch(itr.index)
            {
                case 0: values.push_back(std::get<0>(itr.data)); break;
                case 1: values.push_back(std::get<1>(itr.data)); break;
                case 2: values.push_back(std::get<2>(itr.data)); break;
            }
        }
        return values;
    }

    template <typename _Tp>
    using array_t = std::vector<_Tp>;

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    static array_t<string_t> label_array()
    {
        array_t<string_t> arr;
        auto              contains = [&](const string_t& entry) {
            return std::find(arr.begin(), arr.end(), entry) != arr.end();
        };
        auto insert = [&](const string_t& entry) {
            if(!contains(entry))
                arr.push_back(entry);
        };
        for(const auto& profiler : get_profilers())
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
    static array_t<string_t> descript_array() { return label_array(); }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    static array_t<string_t> display_unit_array()
    {
        return array_t<string_t>(get_profilers().size(), "");
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    static array_t<int64_t> unit_array()
    {
        return array_t<int64_t>(get_labels().size(), 1);
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

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto _get = [&](const value_type& _data) {
            std::vector<double> values;
            for(auto itr : _data)
            {
                switch(itr.index)
                {
                    case 0: values.push_back(std::get<0>(itr.data)); break;
                    case 1: values.push_back(std::get<1>(itr.data)); break;
                    case 2: values.push_back(std::get<2>(itr.data)); break;
                }
            }
            return values;
        };
        array_t<double> _disp  = _get(accum);
        array_t<double> _value = _get(value);
        array_t<double> _accum = _get(accum);
        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("value", _value),
           serializer::make_nvp("accum", _accum), serializer::make_nvp("display", _disp));
    }

    //----------------------------------------------------------------------------------//
    //
    cupti_event(const cupti_event&) = default;
    cupti_event(cupti_event&&)      = default;
    cupti_event& operator           =(const cupti_event& rhs)
    {
        if(this != &rhs)
            base_type::operator=(rhs);
        return *this;
    }
    cupti_event& operator=(cupti_event&&) = default;

private:
    const profvec_t& m_profilers = get_profilers();
    const strvec_t&  m_events    = get_events();
    const strvec_t&  m_metrics   = get_metrics();
    const intvec_t&  m_devices   = get_devices();
    const strvec_t&  m_labels    = get_labels();

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

    static profvec_t& get_private_profilers()
    {
        static thread_local profvec_t _instance;
        return _instance;
    }

    static strvec_t& get_private_events()
    {
        static thread_local strvec_t _instance;
        return _instance;
    }

    static strvec_t& get_private_metrics()
    {
        static thread_local strvec_t _instance;
        return _instance;
    }

    static intvec_t& get_private_devices()
    {
        static thread_local intvec_t _instance;
        return _instance;
    }

    static strvec_t& get_private_labels()
    {
        static thread_local strvec_t _instance;
        return _instance;
    }

    static void init()
    {
        clear();
        auto& _profilers = get_private_profilers();
        auto& _events    = get_private_events();
        auto& _metrics   = get_private_metrics();
        auto& _devices   = get_private_devices();
        auto& _labels    = get_private_labels();

        _events  = get_event_setter()();
        _metrics = get_metric_setter()();
        _devices = get_device_setter()();

        for(const auto& _device : _devices)
            _profilers.push_back(
                profptr_t(new cupti::profiler(_events, _metrics, _device)));

        _labels = label_array();
        std::cout << "Devices : " << writer<intvec_t>(_devices) << std::endl;
        std::cout << "Event   : " << writer<strvec_t>(_events) << std::endl;
        std::cout << "Metrics : " << writer<strvec_t>(_metrics) << std::endl;
        std::cout << "Labels  : " << writer<strvec_t>(_labels) << std::endl;
    }

    static void clear()
    {
        get_private_labels().clear();
        get_private_devices().clear();
        get_private_metrics().clear();
        get_private_events().clear();
        get_private_profilers().clear();
    }
};

}  // namespace component
}  // namespace tim
