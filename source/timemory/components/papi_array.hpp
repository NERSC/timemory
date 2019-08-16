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

#pragma once

#include "timemory/backends/papi.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/storage.hpp"

#include <iostream>

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//                          Array of PAPI counters
//
//--------------------------------------------------------------------------------------//

template <std::size_t MaxNumEvents>
struct papi_array
: public base<papi_array<MaxNumEvents>, std::array<long long, MaxNumEvents>,
              policy::thread_init, policy::thread_finalize>
, public static_counted_object<papi_array<0>>
{
    friend struct policy::wrapper<policy::thread_init, policy::thread_finalize>;

    using size_type   = std::size_t;
    using event_list  = std::vector<int>;
    using value_type  = std::array<long long, MaxNumEvents>;
    using entry_type  = typename value_type::value_type;
    using base_type   = base<papi_array<MaxNumEvents>, value_type, policy::thread_init,
                           policy::thread_finalize>;
    using this_type   = papi_array<MaxNumEvents>;
    using event_count = static_counted_object<papi_array<0>>;
    using get_events_func_t = std::function<event_list()>;

    static const short                   precision = 3;
    static const short                   width     = 12;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::scientific | std::ios_base::dec | std::ios_base::showpoint;

    static bool initialize_papi()
    {
        static thread_local bool _initalized = false;
        static thread_local bool _working    = false;
        if(!_initalized)
        {
            papi::init();
            _initalized = true;
            _working    = papi::working();
            if(!_working)
            {
                std::cerr << "Warning! PAPI failed to initialized!\n";
                std::cerr << "The following PAPI events will not be reported: \n";
                for(const auto& itr : get_events())
                    std::cerr << "    " << papi::get_event_info(itr).short_descr << "\n";
                std::cerr << std::flush;
            }
        }
        return _working;
    }
    static int& event_set()
    {
        static thread_local int _instance = PAPI_NULL;
        return _instance;
    }

    static bool& enable_multiplex()
    {
        static thread_local bool _instance = get_env("TIMEMORY_PAPI_MULTIPLEX", true);
        return _instance;
    }

    static get_events_func_t& get_events_func()
    {
        static get_events_func_t _instance = []() {
            auto events_str = get_env<string_t>("TIMEMORY_PAPI_EVENTS", "");
            std::vector<string_t> events_str_list = delimit(events_str);
            std::vector<int>      events_list;
            for(const auto& itr : events_str_list)
                events_list.push_back(papi::get_event_code(itr));
            return events_list;
        };
        return _instance;
    }

    static event_list get_events() { return get_events_func()(); }

    static void invoke_thread_init()
    {
        if(!initialize_papi())
            return;
        auto events = get_events();
        papi::create_event_set(&event_set());
        papi::add_events(event_set(), events.data(), events.size());
        papi::start(event_set(), enable_multiplex());
    }

    static void invoke_thread_finalize()
    {
        if(!initialize_papi())
            return;
        value_type values;
        auto       events = get_events();
        papi::stop(event_set(), values.data());
        papi::remove_events(event_set(), events.data(), events.size());
        papi::destroy_event_set(event_set());
        event_set() = PAPI_NULL;
    }

    using base_type::accum;
    using base_type::is_running;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;
    using event_count::m_count;

    template <typename _Tp>
    using array_t = std::array<_Tp, MaxNumEvents>;

    explicit papi_array()
    : events(get_events())
    {
        apply<void>::set_value(value, 0);
        apply<void>::set_value(accum, 0);
    }

    ~papi_array() {}

    papi_array(const papi_array& rhs) = default;
    this_type& operator=(const this_type& rhs) = default;
    papi_array(papi_array&& rhs)               = default;
    this_type& operator=(this_type&&) = default;

    // data types
    event_list events;

    static value_type record()
    {
        value_type read_value;
        apply<void>::set_value(read_value, 0);
        if(initialize_papi())
            papi::read(event_set(), read_value.data());
        return read_value;
    }

    template <typename _Tp = double>
    std::vector<_Tp> get() const
    {
        std::vector<_Tp> values;
        auto&            _data = (is_transient) ? accum : value;
        for(auto& itr : _data)
            values.push_back(itr);
        return values;
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp = record();
        for(size_type i = 0; i < events.size(); ++i)
            accum[i] += (tmp[i] - value[i]);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < events.size(); ++i)
            accum[i] += rhs.accum[i];
        for(size_type i = 0; i < events.size(); ++i)
            value[i] += rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < events.size(); ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < events.size(); ++i)
            value[i] -= rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

public:
    //==================================================================================//
    //
    //      data representation
    //
    //==================================================================================//

    static std::string label() { return "papi" + std::to_string(event_set()); }
    static std::string descript() { return ""; }
    static std::string display_unit() { return ""; }
    static int64_t     unit() { return 1; }

    entry_type compute_display(int evt_type) const
    {
        auto val = (is_transient) ? accum[evt_type] : value[evt_type];
        return val;
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        array_t<double> _disp;
        array_t<double> _value;
        array_t<double> _accum;
        for(size_type i = 0; i < events.size(); ++i)
        {
            _disp[i]  = compute_display(i);
            _value[i] = value[i];
            _accum[i] = accum[i];
        }
        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("value", _value),
           serializer::make_nvp("accum", _accum), serializer::make_nvp("display", _disp));
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    array_t<std::string> label_array() const
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).short_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    array_t<std::string> descript_array() const
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).long_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    array_t<std::string> display_unit_array() const
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).units;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    array_t<int64_t> unit_array() const
    {
        array_t<int64_t> arr;
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = 1;
        return arr;
    }

    string_t compute_display() const
    {
        auto val              = (is_transient) ? accum : value;
        auto _compute_display = [&](std::ostream& os, size_type idx) {
            auto     _obj_value = val[idx];
            auto     _evt_type  = events[idx];
            string_t _label     = papi::get_event_info(_evt_type).short_descr;
            string_t _disp      = papi::get_event_info(_evt_type).units;
            auto     _prec      = base_type::get_precision();
            auto     _width     = base_type::get_width();
            auto     _flags     = base_type::get_format_flags();

            std::stringstream ss, ssv, ssi;
            ssv.setf(_flags);
            ssv << std::setw(_width) << std::setprecision(_prec) << _obj_value;
            if(!_disp.empty())
                ssv << " " << _disp;
            if(!_label.empty())
                ssi << " " << _label;
            ss << ssv.str() << ssi.str();
            os << ss.str();
        };

        std::stringstream ss;
        for(size_type i = 0; i < events.size(); ++i)
        {
            _compute_display(ss, i);
            if(i + 1 < events.size())
                ss << ", ";
        }
        return ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        // output the metrics
        auto _value = obj.compute_display();
        auto _label = this_type::get_label();
        auto _disp  = this_type::display_unit();
        auto _prec  = this_type::get_precision();
        auto _width = this_type::get_width();
        auto _flags = this_type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;
        if(!_disp.empty())
            ss_extra << " " << _disp;
        else if(!_label.empty())
            ss_extra << " " << _label;
        os << ss_value.str() << ss_extra.str();

        return os;
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
