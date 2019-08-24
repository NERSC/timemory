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
 * \headerfile cupti_counters.hpp "timemory/cupti_counters.hpp"
 * Provides implementation of CUPTI routines.
 *
 */

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

#include <algorithm>
#include <iterator>
#include <numeric>
#include <set>
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

//#if defined(TIMEMORY_USE_CUPTI)

struct cupti_counters
: public base<cupti_counters, cupti::profiler::results_t, policy::global_init,
              policy::global_init, policy::serialization>
{
    // required aliases
    using value_type = cupti::profiler::results_t;
    using this_type  = cupti_counters;
    using base_type  = base<cupti_counters, value_type, policy::global_init,
                           policy::global_init, policy::serialization>;

    // custom aliases
    using size_type     = std::size_t;
    using string_t      = std::string;
    using kernel_data_t = cupti::result;
    using entry_type    = typename value_type::value_type;
    // short-hand for vectors
    using strvec_t  = std::vector<string_t>;
    using intvec_t  = std::vector<int>;
    using profptr_t = std::shared_ptr<cupti::profiler>;
    using profvec_t = std::vector<profptr_t>;
    /// a tuple of the <devices, events, metrics>
    using tuple_type = std::tuple<intvec_t, strvec_t, strvec_t>;
    /// a tuple of the available events and metrics on a specific device
    using device_tuple_type = std::tuple<int, strvec_t, strvec_t>;
    /// function for setting device, metrics, and events to record
    using event_func_t  = std::function<strvec_t()>;
    using metric_func_t = std::function<strvec_t()>;
    using device_func_t = std::function<intvec_t()>;
    /// function for setting all of device, metrics, and events
    using initializer_type = std::function<tuple_type()>;

    static const short                   precision = 3;
    static const short                   width     = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::dec | std::ios_base::showpoint;

    static event_func_t& get_event_initializer()
    {
        static event_func_t _instance = []() {
            auto env = get_env<string_t>("TIMEMORY_CUPTI_EVENTS", "");
            return delimit(env);
        };
        return _instance;
    }

    static metric_func_t& get_metric_initializer()
    {
        static metric_func_t _instance = []() {
            auto env = get_env<string_t>("TIMEMORY_CUPTI_METRICS", "");
            return delimit(env);
        };
        return _instance;
    }

    static device_func_t& get_device_initializer()
    {
        static device_func_t _instance = []() {
            auto     env = get_env<string_t>("TIMEMORY_CUPTI_DEVICES", "");
            intvec_t devices;
            int      device_count = cuda::device_count();
            if(env.length() > 0)
            {
                auto _devices = tim::delimit(env);
                for(auto itr : _devices)
                {
                    auto dev = atoi(itr.c_str());
                    if(dev < device_count)
                        devices.push_back(dev);
                }
            }
            if(devices.size() == 0)
            {
                devices.resize(device_count, 0);
                std::iota(devices.begin(), devices.end(), 0);
            }
            return devices;
        };
        return _instance;
    }

    static initializer_type& get_initializer()
    {
        static auto _lambda_instance = []() -> tuple_type {
            return tuple_type(get_device_initializer()(), get_event_initializer()(),
                              get_metric_initializer()());
        };
        static initializer_type _instance = _lambda_instance;
        return _instance;
    }

    static void invoke_global_init() { init(); }
    static void invoke_global_finalize() { clear(); }

    static const profvec_t& get_profilers() { return _get_profilers(); }
    static const strvec_t&  get_events() { return _get_events(); }
    static const strvec_t&  get_metrics() { return _get_metrics(); }
    static const intvec_t&  get_devices() { return _get_devices(); }
    static const strvec_t&  get_labels() { return _get_labels(); }

    explicit cupti_counters()
    {
        auto& _labels = _get_labels();
        value.resize(_labels.size());
        accum.resize(_labels.size());
        for(size_type i = 0; i < _labels.size(); ++i)
        {
            value[i].name = _labels[i];
            accum[i].name = _labels[i];
        }
    }

    ~cupti_counters() {}

    static int64_t unit() { return 1; }
    // leave these empty
    static string_t label() { return ""; }
    static string_t descript() { return ""; }
    static string_t display_unit() { return ""; }

    static value_type record()
    {
        value_type tmp;
        auto&      _profilers = _get_profilers();
        auto&      _labels    = _get_labels();
        for(size_type i = 0; i < _profilers.size(); ++i)
        {
            _profilers[i]->stop();
            if(tmp.size() == 0)
            {
                tmp = _profilers[i]->get_events_and_metrics(_labels);
            } else if(tmp.size() == _labels.size())
            {
                auto ret = _profilers[i]->get_events_and_metrics(_labels);
                for(size_t j = 0; j < _labels.size(); ++j)
                    tmp[j] += ret[j];
            } else
            {
                fprintf(stderr, "Warning! mis-matched size in cupti_event::%s @ %s:%i\n",
                        __FUNCTION__, __FILE__, __LINE__);
            }
        }
        return tmp;
    }

    string_t get_display() const
    {
        auto _get_display = [&](std::ostream& os, const cupti::result& obj) {
            auto _label = obj.name;
            auto _prec  = base_type::get_precision();
            auto _width = base_type::get_width();
            auto _flags = base_type::get_format_flags();

            std::stringstream ss, ssv, ssi;
            ssv.setf(_flags);
            ssv << std::setw(_width) << std::setprecision(_prec);
            cupti::print(ssv, obj.data);
            if(!_label.empty())
                ssi << " " << _label;
            ss << ssv.str() << ssi.str();
            os << ss.str();
        };

        const auto&       _data = (is_transient) ? accum : value;
        std::stringstream ss;
        for(size_type i = 0; i < _data.size(); ++i)
        {
            _get_display(ss, _data[i]);
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
            values.push_back(cupti::get<double>(itr.data));
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
        value            = record();
        auto& _profilers = _get_profilers();
        for(size_type i = 0; i < _profilers.size(); ++i)
            _profilers[i]->start();
    }

    void stop()
    {
        value_type tmp = record();
        if(accum.size() == 0)
        {
            accum = tmp;
            for(size_type i = 0; i < tmp.size(); ++i)
                accum[i] -= value[i];
        } else
        {
            for(size_type i = 0; i < tmp.size(); ++i)
                accum[i] += (tmp[i] - value[i]);
        }
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        auto& _labels = _get_labels();
        if(accum.empty())
        {
            accum = rhs.accum;
        } else
        {
            for(size_type i = 0; i < _labels.size(); ++i)
                accum[i] += rhs.accum[i];
        }
        if(value.empty())
        {
            value = rhs.value;
        } else
        {
            for(size_type i = 0; i < _labels.size(); ++i)
                value[i] += rhs.value[i];
        }
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        auto& _labels = _get_labels();
        if(!accum.empty())
        {
            for(size_type i = 0; i < _labels.size(); ++i)
                accum[i] -= rhs.accum[i];
        }
        if(!value.empty())
        {
            for(size_type i = 0; i < _labels.size(); ++i)
                value[i] -= rhs.value[i];
        }
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
                values.push_back(cupti::get<double>(itr.data));
            return values;
        };
        array_t<double> _disp  = _get(accum);
        array_t<double> _value = _get(value);
        array_t<double> _accum = _get(accum);
        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("repr_data", _disp),
           serializer::make_nvp("value", _value), serializer::make_nvp("accum", _accum),
           serializer::make_nvp("display", _disp));
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Archive>
    static void invoke_serialize(_Archive& ar, const unsigned int /*version*/)
    {
        auto& _devices = _get_devices();
        auto& _events  = _get_events();
        auto& _metrics = _get_metrics();
        auto& _labels  = _get_labels();

        ar(serializer::make_nvp("devices", _devices),
           serializer::make_nvp("events", _events),
           serializer::make_nvp("metrics", _metrics),
           serializer::make_nvp("labels", _labels));
    }

    //----------------------------------------------------------------------------------//
    //
    cupti_counters(const cupti_counters&) = default;
    cupti_counters(cupti_counters&&)      = default;
    cupti_counters& operator              =(const cupti_counters& rhs)
    {
        if(this != &rhs)
            base_type::operator=(rhs);
        return *this;
    }
    cupti_counters& operator=(cupti_counters&&) = default;

private:
    template <typename _Tp>
    struct writer
    {
        using const_iterator = typename _Tp::const_iterator;
        _Tp& obj;
        writer(_Tp& _obj)
        : obj(_obj)
        {}

        const_iterator begin() const { return obj.begin(); }
        const_iterator end() const { return obj.end(); }

        friend std::ostream& operator<<(std::ostream& os, const writer& obj)
        {
            auto sz = std::distance(obj.begin(), obj.end());
            for(auto itr = obj.begin(); itr != obj.end(); ++itr)
            {
                auto idx = std::distance(obj.begin(), itr);
                os << (*itr);
                if(idx + 1 < sz)
                    os << ", ";
            }
            return os;
        }
    };

    static profvec_t& _get_profilers()
    {
        static profvec_t _instance;
        return _instance;
    }

    static strvec_t& _get_events()
    {
        static strvec_t _instance;
        return _instance;
    }

    static strvec_t& _get_metrics()
    {
        static strvec_t _instance;
        return _instance;
    }

    static intvec_t& _get_devices()
    {
        static intvec_t _instance;
        return _instance;
    }

    static strvec_t& _get_labels()
    {
        static strvec_t _instance;
        return _instance;
    }

    static strvec_t generate_labels()
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

    static strvec_t get_available_events(int devid)
    {
        return cupti::available_events(cupti::get_device(devid));
    }

    static strvec_t get_available_metrics(int devid)
    {
        return cupti::available_metrics(cupti::get_device(devid));
    }

    static device_tuple_type get_available(const tuple_type&, int);

    static void init()
    {
        cupti::init_driver();
        clear();
        auto& _profilers = _get_profilers();
        auto& _events    = _get_events();
        auto& _metrics   = _get_metrics();
        auto& _devices   = _get_devices();
        auto& _labels    = _get_labels();

        auto _init = get_initializer()();

        _devices = std::get<0>(_init);
        _events  = std::get<1>(_init);
        _metrics = std::get<2>(_init);

        using intset_t = std::set<int>;
        using strset_t = std::set<string_t>;

        intset_t _used_devs;
        strset_t _used_evts;
        strset_t _used_mets;

        for(const auto& _device : _devices)
        {
            auto  _dev_init = get_available(_init, _device);
            auto& _dev      = std::get<0>(_dev_init);

            // if < 0, no metrics or events available/specified
            if(_dev < 0)
                continue;

            auto& _evt = std::get<1>(_dev_init);
            auto& _met = std::get<2>(_dev_init);

            _profilers.push_back(profptr_t(new cupti::profiler(_evt, _met, _dev)));

            _used_devs.insert(_dev);
            for(const auto& itr : _evt)
                _used_evts.insert(itr);
            for(const auto& itr : _met)
                _used_mets.insert(itr);
        }

        _labels = generate_labels();
        if(settings::verbose() > 0 || settings::debug())
        {
            std::cout << "Devices : " << writer<intset_t>(_used_devs) << std::endl;
            std::cout << "Event   : " << writer<strset_t>(_used_evts) << std::endl;
            std::cout << "Metrics : " << writer<strset_t>(_used_mets) << std::endl;
            std::cout << "Labels  : " << writer<strvec_t>(_labels) << std::endl;
        }
    }

    static void clear()
    {
        _get_devices().clear();
        _get_metrics().clear();
        _get_events().clear();
        _get_profilers().clear();
    }
};

/*
#else
struct cupti_counters
: public base<cupti_counters, std::vector<cupti::result>, policy::thread_init,
              policy::thread_finalize, policy::serialization>
{
    using value_type = std::vector<cupti::result>;
    using this_type  = cupti_counters;
    using base_type  = base<this_type, value_type, policy::thread_init,
                           policy::thread_finalize, policy::serialization>;

    // reproduce some aliases here
    using strvec_t = std::vector<string_t>;
    using intvec_t = std::vector<int>;
    /// a tuple of the <devices, events, metrics>
    using tuple_type = std::tuple<intvec_t, strvec_t, strvec_t>;
    /// function for setting device, metrics, and events to record
    using event_func_t  = std::function<strvec_t()>;
    using metric_func_t = std::function<strvec_t()>;
    using device_func_t = std::function<intvec_t()>;
    /// function for setting all of device, metrics, and events
    using initializer_type = std::function<tuple_type()>;

    template <typename _Tp>
    using array_t = std::vector<_Tp>;

    static const short                   precision = 0;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    template <typename _Archive>
    static void invoke_serialize(_Archive&, const unsigned int)
    {}

    static void              invoke_thread_init() {}
    static void              invoke_thread_finalize() {}
    static array_t<string_t> label_array() { return array_t<string_t>{}; }
    static array_t<string_t> descript_array() { return array_t<string_t>{}; }
    static array_t<string_t> display_unit_array() { return array_t<string_t>{}; }
    static array_t<int64_t>  unit_array() { return array_t<int64_t>{}; }
    static int64_t           unit() { return 1; }
    static std::string       label() { return "cupti_counters"; }
    static std::string       descript() { return "cupti_counters"; }
    static std::string       display_unit() { return ""; }
    static value_type        record() { return value_type{}; }
    value_type               get_display() const { return value_type{}; }
    value_type               get() const { return value_type{}; }
    void                     start() {}
    void                     stop() {}

    static event_func_t& get_event_initializer()
    {
        static event_func_t _instance = []() { return strvec_t{}; };
        return _instance;
    }

    static metric_func_t& get_metric_initializer()
    {
        static metric_func_t _instance = []() { return strvec_t{}; };
        return _instance;
    }

    static device_func_t& get_device_initializer()
    {
        static device_func_t _instance = []() { return intvec_t{}; };
        return _instance;
    }

    static initializer_type& get_initializer()
    {
        static auto _lambda_instance = []() -> tuple_type {
            return tuple_type(get_device_initializer()(), get_event_initializer()(),
                              get_metric_initializer()());
        };
        static initializer_type _instance = _lambda_instance;
        return _instance;
    }
};
#endif
*/

//--------------------------------------------------------------------------------------//

inline cupti_counters::device_tuple_type
cupti_counters::get_available(const tuple_type& _init, int devid)
{
    if(devid < 0 || devid >= cuda::device_count())
        return device_tuple_type(-1, strvec_t(), strvec_t());

    // handle events
    strvec_t    _events       = std::get<1>(_init);
    const auto& _avail_events = get_available_events(devid);
    auto        _find_event   = [&_avail_events, devid](const string_t& evt) {
        bool nf = (std::find(std::begin(_avail_events), std::end(_avail_events), evt) ==
                   std::end(_avail_events));
        if(nf)
        {
            fprintf(stderr,
                    "[cupti_counters]> Removing unavailable event '%s' on device %i...\n",
                    evt.c_str(), devid);
        }
        return nf;
    };

    // handle metrics
    strvec_t    _metrics      = std::get<2>(_init);
    const auto& _avail_metric = get_available_metrics(devid);
    auto        _find_metric  = [&_avail_metric, devid](const string_t& met) {
        bool nf = (std::find(std::begin(_avail_metric), std::end(_avail_metric), met) ==
                   std::end(_avail_metric));
        if(nf)
        {
            fprintf(stderr,
                    "[cupti_counters]> Removing unavailable metric '%s' on device "
                    "%i...\n",
                    met.c_str(), devid);
        }
        return nf;
    };

    // do the removals
    _events.erase(std::remove_if(std::begin(_events), std::end(_events), _find_event),
                  std::end(_events));

    _metrics.erase(std::remove_if(std::begin(_metrics), std::end(_metrics), _find_metric),
                   std::end(_metrics));

    // determine total
    auto ntot = _events.size() + _metrics.size();
    return device_tuple_type((ntot == 0) ? -1 : devid, _events, _metrics);
}

//--------------------------------------------------------------------------------------//

using cupti_event = cupti_counters;

//--------------------------------------------------------------------------------------//

}  // namespace component

}  // namespace tim
