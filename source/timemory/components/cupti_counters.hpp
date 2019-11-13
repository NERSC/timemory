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
#include "timemory/bits/settings.hpp"
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
//--------------------------------------------------------------------------------------//

namespace stl_overload
{
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
}  // namespace stl_overload

//--------------------------------------------------------------------------------------//

namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

#endif

//--------------------------------------------------------------------------------------//
//
//          CUPTI component
//
//--------------------------------------------------------------------------------------//

//#if defined(TIMEMORY_USE_CUPTI)

struct cupti_counters
: public base<cupti_counters, cupti::profiler::results_t, policy::global_init,
              policy::global_finalize, policy::serialization>
{
    // required aliases
    using value_type = cupti::profiler::results_t;
    using this_type  = cupti_counters;
    using base_type  = base<cupti_counters, value_type, policy::global_init,
                           policy::global_finalize, policy::serialization>;

    // custom aliases
    using size_type        = std::size_t;
    using string_t         = std::string;
    using kernel_data_t    = cupti::result;
    using entry_type       = typename value_type::value_type;
    using results_t        = cupti::profiler::results_t;
    using kernel_results_t = cupti::profiler::kernel_results_t;

    // short-hand for vectors
    using strvec_t  = std::vector<string_t>;
    using profptr_t = std::shared_ptr<cupti::profiler>;
    /// a tuple of the <devices, events, metrics>
    using tuple_type = std::tuple<int, strvec_t, strvec_t>;
    /// function for setting device, metrics, and events to record
    using event_func_t  = std::function<strvec_t()>;
    using metric_func_t = std::function<strvec_t()>;
    using device_func_t = std::function<int()>;
    /// function for setting all of device, metrics, and events
    using get_initializer_t = std::function<tuple_type()>;

    static const short precision = 3;
    static const short width     = 8;

    static event_func_t& get_event_initializer()
    {
        static event_func_t _instance = []() {
            return delimit(settings::cupti_events());
        };
        return _instance;
    }

    static metric_func_t& get_metric_initializer()
    {
        static metric_func_t _instance = []() {
            return delimit(settings::cupti_metrics());
        };
        return _instance;
    }

    static device_func_t& get_device_initializer()
    {
        static device_func_t _instance = []() {
            if(cuda::device_count() < 1)
                return -1;
            return settings::cupti_device();
        };
        return _instance;
    }

    static get_initializer_t& get_initializer()
    {
        static auto _lambda_instance = []() -> tuple_type {
            return tuple_type(get_device_initializer()(), get_event_initializer()(),
                              get_metric_initializer()());
        };
        static get_initializer_t _instance = _lambda_instance;
        return _instance;
    }

    static void configure()
    {
        if(_get_profiler().get() == nullptr)
            init();
    }

    static void configure(int device, const strvec_t& events, const strvec_t& metrics)
    {
        get_initializer() = [=]() -> tuple_type {
            return tuple_type(device, events, metrics);
        };
        if(_get_profiler().get() == nullptr)
            init();
    }

    static void invoke_global_init(storage_type*) { configure(); }
    static void invoke_global_finalize(storage_type*) { clear(); }

    static const profptr_t& get_profiler() { return _get_profiler(); }
    static const strvec_t&  get_events() { return *_get_events(); }
    static const strvec_t&  get_metrics() { return *_get_metrics(); }
    static const int&       get_device() { return *_get_device(); }
    static const strvec_t&  get_labels() { return *_get_labels(); }

    explicit cupti_counters()
    {
        configure();
        auto* _labels = _get_labels();
        if(_labels)
        {
            value.resize(_labels->size());
            accum.resize(_labels->size());
            for(size_type i = 0; i < _labels->size(); ++i)
            {
                value[i].name = (*_labels)[i];
                accum[i].name = (*_labels)[i];
            }
        }
    }

    ~cupti_counters() {}
    cupti_counters(const cupti_counters&) = default;
    cupti_counters(cupti_counters&&)      = default;
    cupti_counters& operator              =(const cupti_counters& rhs)
    {
        if(this != &rhs)
        {
            base_type::operator=(rhs);
            m_kernel_value     = rhs.m_kernel_value;
            m_kernel_accum     = rhs.m_kernel_accum;
        }
        return *this;
    }
    cupti_counters& operator=(cupti_counters&&) = default;

    static int64_t unit() { return 1; }
    // leave these empty
    static string_t label() { return "cupti_counters"; }
    static string_t description() { return "CUpti Callback API for hardware counters"; }
    static string_t display_unit() { return ""; }

    static value_type record()
    {
        configure();
        value_type tmp;
        auto&      _profiler = _get_profiler();
        auto&      _labels   = *_get_labels();
        _profiler->stop();
        if(tmp.size() == 0)
        {
            tmp = _profiler->get_events_and_metrics(_labels);
        }
        else if(tmp.size() == _labels.size())
        {
            auto ret = _profiler->get_events_and_metrics(_labels);
            for(size_t j = 0; j < _labels.size(); ++j)
                tmp[j] += ret[j];
        }
        else
        {
            fprintf(stderr, "Warning! mis-matched size in cupti_event::%s @ %s:%i\n",
                    TIMEMORY_ERROR_FUNCTION_MACRO, __FILE__, __LINE__);
        }

        return tmp;
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        set_started();
        value           = record();
        auto& _profiler = _get_profiler();
        m_kernel_value  = _profiler->get_kernel_events_and_metrics(*_get_labels());
        _profiler->start();
    }

    void stop()
    {
        using namespace stl_overload;

        value_type       tmp       = record();
        auto&            _profiler = _get_profiler();
        kernel_results_t kernel_data =
            _profiler->get_kernel_events_and_metrics(*_get_labels());
        kernel_results_t kernel_tmp = kernel_data;

        if(accum.size() == 0)
        {
            accum = tmp;
            for(size_type i = 0; i < tmp.size(); ++i)
                accum[i] -= value[i];
        }
        else
        {
            for(size_type i = 0; i < tmp.size(); ++i)
                accum[i] += (tmp[i] - value[i]);
        }

        for(size_t i = 0; i < m_kernel_value.size(); ++i)
            kernel_tmp[i].second -= m_kernel_value[i].second;
        for(size_t i = 0; i < kernel_tmp.size(); ++i)
        {
            if(i >= m_kernel_accum.size())
                m_kernel_accum.resize(i + 1, kernel_tmp[i]);
            else
                m_kernel_accum[i].second += kernel_tmp[i].second;
        }

        value          = std::move(tmp);
        m_kernel_value = std::move(kernel_data);
        set_stopped();
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

    using secondary_type = std::unordered_multimap<std::string, value_type>;

    secondary_type get_secondary() const
    {
        secondary_type _data;
        for(const auto& itr : (is_transient) ? m_kernel_accum : m_kernel_value)
            _data.insert({ itr.first, itr.second });
        return _data;
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
        auto profiler = get_profiler();
        for(const auto& itr : profiler->get_event_names())
            insert(itr);
        for(const auto& itr : profiler->get_metric_names())
            insert(itr);
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
        return array_t<string_t>(get_labels().size(), "");
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    static array_t<int64_t> unit_array()
    {
        return array_t<int64_t>(get_labels().size(), 1);
    }

    this_type& operator+=(const this_type& rhs)
    {
        auto _combine = [](value_type& _data, const value_type& _other) {
            auto& _labels = *_get_labels();
            if(_data.empty())
                _data = _other;
            else
            {
                for(size_type i = 0; i < _labels.size(); ++i)
                    _data[i] += _other[i];
            }
        };

        _combine(value, rhs.value);
        _combine(accum, rhs.accum);
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        auto _combine = [](value_type& _data, const value_type& _other) {
            auto& _labels = *_get_labels();
            // set to other
            if(_data.empty())
                _data = _other;
            // subtract other (if data was empty, will contain zero data)
            for(size_type i = 0; i < _labels.size(); ++i)
                _data[i] -= _other[i];
        };

        _combine(value, rhs.value);
        _combine(accum, rhs.accum);
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator+=(const results_t& rhs)
    {
        auto _combine = [](value_type& _data, const value_type& _other) {
            if(_data.empty())
                _data = _other;
            else
            {
                for(size_type i = 0; i < _other.size(); ++i)
                    _data[i] += _other[i];
            }
        };

        _combine(value, rhs);
        _combine(accum, rhs);

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
        auto& _devices = *_get_device();
        auto& _events  = *_get_events();
        auto& _metrics = *_get_metrics();
        auto& _labels  = *_get_labels();

        ar(serializer::make_nvp("devices", _devices),
           serializer::make_nvp("events", _events),
           serializer::make_nvp("metrics", _metrics),
           serializer::make_nvp("labels", _labels));
    }

    //----------------------------------------------------------------------------------//

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

        friend std::ostream& operator<<(std::ostream& os, const writer<_Tp>& _obj)
        {
            auto sz = std::distance(_obj.begin(), _obj.end());
            for(auto itr = _obj.begin(); itr != _obj.end(); ++itr)
            {
                auto idx = std::distance(_obj.begin(), itr);
                os << (*itr);
                if(idx + 1 < sz)
                    os << ", ";
            }
            return os;
        }
    };

    static profptr_t& _get_profiler()
    {
        static profptr_t _instance = profptr_t(nullptr);
        return _instance;
    }

    static strvec_t*& _get_events()
    {
        static strvec_t* _instance = new strvec_t();
        return _instance;
    }

    static strvec_t*& _get_metrics()
    {
        static strvec_t* _instance = new strvec_t();
        return _instance;
    }

    static int*& _get_device()
    {
        static int* _instance = new int(0);
        return _instance;
    }

    static strvec_t*& _get_labels()
    {
        static strvec_t* _instance = new strvec_t();
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
        auto profiler = get_profiler();
        if(profiler)
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

    static tuple_type get_available(const tuple_type&, int);

    static void init()
    {
        cupti::init_driver();
        cuda::device_sync();
        clear();

        auto& _profiler = _get_profiler();
        auto& _events   = *_get_events();
        auto& _metrics  = *_get_metrics();
        auto& _device   = *_get_device();
        auto& _labels   = *_get_labels();

        auto _init = get_initializer()();

        _device  = std::get<0>(_init);
        _events  = std::get<1>(_init);
        _metrics = std::get<2>(_init);

        using intset_t = std::set<int>;
        using strset_t = std::set<string_t>;

        intset_t _used_devs;
        strset_t _used_evts;
        strset_t _used_mets;

        auto  _dev_init = get_available(_init, _device);
        auto& _dev      = std::get<0>(_dev_init);

        // if < 0, no metrics or events available/specified
        if(_dev >= 0)
        {
            if(settings::debug())
                printf("Creating CUPTI hardware profiler for device %i...\n", _device);

            auto& _evt = std::get<1>(_dev_init);
            auto& _met = std::get<2>(_dev_init);

            if(_evt.size() > 0 && _met.size() > 0)
            {
                _profiler.reset(new cupti::profiler(_evt, _met, _dev));
                _used_devs.insert(_dev);
                for(const auto& itr : _evt)
                    _used_evts.insert(itr);
                for(const auto& itr : _met)
                    _used_mets.insert(itr);
                _labels = generate_labels();
            }
        }
        else
        {
            fprintf(stderr, "[cupti_counters]> Warning! No devices available!");
        }

        if(_used_devs.size() > 0)
        {
            if(settings::verbose() > 0 || settings::debug())
            {
                std::cout << "Devices : " << writer<intset_t>(_used_devs) << std::endl;
                std::cout << "Event   : " << writer<strset_t>(_used_evts) << std::endl;
                std::cout << "Metrics : " << writer<strset_t>(_used_mets) << std::endl;
                std::cout << "Labels  : " << writer<strvec_t>(_labels) << std::endl;
            }
        }
    }

    static void clear()
    {
        if(_get_metrics())
            _get_metrics()->clear();
        if(_get_events())
            _get_events()->clear();
        _get_profiler().reset();
    }

public:
    static void cleanup()
    {
        clear();
        delete _get_device();
        delete _get_events();
        delete _get_labels();
        delete _get_metrics();
        _get_device()  = nullptr;
        _get_events()  = nullptr;
        _get_labels()  = nullptr;
        _get_metrics() = nullptr;
    }

private:
    kernel_results_t m_kernel_value;
    kernel_results_t m_kernel_accum;
};

//--------------------------------------------------------------------------------------//

inline cupti_counters::tuple_type
cupti_counters::get_available(const tuple_type& _init, int devid)
{
    if(devid < 0 || devid >= cuda::device_count())
    {
        fprintf(stderr, "[cupti_counters]> Invalid device id: %i...\n", devid);
        return tuple_type(-1, strvec_t(), strvec_t());
    }

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
    return tuple_type(devid, _events, _metrics);
}

//--------------------------------------------------------------------------------------//

using cupti_event = cupti_counters;

//--------------------------------------------------------------------------------------//

}  // namespace component

}  // namespace tim
