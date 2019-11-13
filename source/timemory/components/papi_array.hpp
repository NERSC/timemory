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
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<papi_array<8>, std::array<long long, 8>, policy::thread_init,
                            policy::thread_finalize>;

extern template struct base<papi_array<16>, std::array<long long, 16>,
                            policy::thread_init, policy::thread_finalize>;

extern template struct base<papi_array<32>, std::array<long long, 32>,
                            policy::thread_init, policy::thread_finalize>;

#endif

//--------------------------------------------------------------------------------------//
//
//                          Array of PAPI counters
//
//--------------------------------------------------------------------------------------//

template <size_t MaxNumEvents>
struct papi_array
: public base<papi_array<MaxNumEvents>, std::array<long long, MaxNumEvents>,
              policy::thread_init, policy::thread_finalize>
{
    using size_type  = std::size_t;
    using event_list = std::vector<int>;
    using value_type = std::array<long long, MaxNumEvents>;
    using entry_type = typename value_type::value_type;
    using this_type  = papi_array<MaxNumEvents>;
    using base_type =
        base<this_type, value_type, policy::thread_init, policy::thread_finalize>;
    using storage_type      = typename base_type::storage_type;
    using get_initializer_t = std::function<event_list()>;

    static const short precision = 3;
    static const short width     = 8;

    template <typename _Tp>
    using array_t = std::array<_Tp, MaxNumEvents>;

    //----------------------------------------------------------------------------------//

    static bool initialize_papi()
    {
        static thread_local bool _initalized = false;
        static thread_local bool _working    = false;
        if(!_initalized)
        {
            papi::init();
            papi::register_thread();
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

    //----------------------------------------------------------------------------------//

    static int event_set() { return _event_set(); }

    //----------------------------------------------------------------------------------//

    static get_initializer_t& get_initializer()
    {
        static get_initializer_t _instance = []() {
            papi::init();
            auto events_str = settings::papi_events();

            if(settings::verbose() > 1 || settings::debug())
            {
                static std::atomic<int> _once;
                if(_once++ == 0)
                {
                    printf("[papi_array]> TIMEMORY_PAPI_EVENTS: '%s'...\n",
                           events_str.c_str());
                }
            }

            std::vector<string_t> events_str_list = delimit(events_str);
            std::vector<int>      events_list;
            for(const auto& itr : events_str_list)
            {
                if(itr.length() == 0)
                    continue;

                if(settings::debug())
                    printf("[papi_array]> Getting event code from '%s'...\n",
                           itr.c_str());

                int evt_code = papi::get_event_code(itr);
                if(evt_code == PAPI_NOT_INITED)  // defined as zero
                {
                    std::stringstream ss;
                    ss << "[papi_array] Error creating event with ID: " << itr;
                    if(settings::papi_fail_on_error())
                        throw std::runtime_error(ss.str());
                    else
                        fprintf(stderr, "%s\n", ss.str().c_str());
                }
                else
                {
                    if(settings::debug())
                        printf("[papi_array] Successfully created event '%s' with code "
                               "'%i'...\n",
                               itr.c_str(), evt_code);
                    events_list.push_back(evt_code);
                }
            }
            return events_list;
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static event_list get_events()
    {
        static event_list _instance = get_initializer()();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void invoke_thread_init(storage_type*)
    {
        if(!initialize_papi())
            return;
        auto events = get_events();
        if(events.size() > 0)
        {
            papi::create_event_set(&_event_set(), settings::papi_multiplexing());
            papi::add_events(_event_set(), events.data(), events.size());
            papi::start(_event_set());
        }
    }

    //----------------------------------------------------------------------------------//

    static void invoke_thread_finalize(storage_type*)
    {
        if(!initialize_papi())
            return;
        auto events = get_events();
        if(events.size() > 0 && _event_set() != PAPI_NULL && _event_set() >= 0)
        {
            value_type values;
            papi::stop(_event_set(), values.data());
            // papi::remove_events(_event_set(), events.data(), events.size());
            papi::destroy_event_set(_event_set());
            _event_set() = PAPI_NULL;
        }
        papi::unregister_thread();
    }

    //----------------------------------------------------------------------------------//

    explicit papi_array()
    : events(get_events())
    {
        apply<void>::set_value(value, 0);
        apply<void>::set_value(accum, 0);
    }

    //----------------------------------------------------------------------------------//

    ~papi_array() {}
    papi_array(const papi_array& rhs) = default;
    papi_array(papi_array&& rhs)      = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&&) = default;

    // data types
    event_list events;

    //----------------------------------------------------------------------------------//

    std::size_t size() { return events.size(); }

    //----------------------------------------------------------------------------------//

    static value_type record()
    {
        value_type read_value;
        apply<void>::set_value(read_value, 0);
        if(initialize_papi() && _event_set() != PAPI_NULL)
            papi::read(_event_set(), read_value.data());
        return read_value;
    }

    //----------------------------------------------------------------------------------//

    template <typename _Tp = double>
    std::vector<_Tp> get() const
    {
        std::vector<_Tp> values;
        auto&            _data = (is_transient) ? accum : value;
        for(auto& itr : _data)
            values.push_back(itr);
        values.resize(events.size());
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

    //----------------------------------------------------------------------------------//

    void stop()
    {
        auto tmp = record();
        for(size_type i = 0; i < events.size(); ++i)
            accum[i] += (tmp[i] - value[i]);
        value = std::move(tmp);
        set_stopped();
    }

    //----------------------------------------------------------------------------------//

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

    //----------------------------------------------------------------------------------//

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

    //----------------------------------------------------------------------------------//

protected:
    using base_type::accum;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    friend struct policy::wrapper<policy::thread_init, policy::thread_finalize>;
    friend struct base<this_type, value_type, policy::thread_init,
                       policy::thread_finalize>;

    using base_type::implements_storage_v;
    friend class impl::storage<this_type, implements_storage_v>;

public:
    //==================================================================================//
    //
    //      data representation
    //
    //==================================================================================//

    static std::string label()
    {
        return "papi_array" + std::to_string((_event_set() < 0) ? 0 : _event_set());
    }
    static std::string description() { return "Array of PAPI HW counters"; }

    entry_type get_display(int evt_type) const
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
            _disp[i]  = get_display(i);
            _value[i] = value[i];
            _accum[i] = accum[i];
        }
        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("repr_data", _disp),
           serializer::make_nvp("value", _value), serializer::make_nvp("accum", _accum),
           serializer::make_nvp("display", _disp));
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

    //----------------------------------------------------------------------------------//

    string_t get_display() const
    {
        if(events.size() == 0)
            return "";
        auto val          = (is_transient) ? accum : value;
        auto _get_display = [&](std::ostream& os, size_type idx) {
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
            _get_display(ss, i);
            if(i + 1 < events.size())
                ss << ", ";
        }
        return ss.str();
    }

    //----------------------------------------------------------------------------------//

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        if(obj.events.size() == 0)
            return os;
        // output the metrics
        auto _value = obj.get_display();
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

private:
    //----------------------------------------------------------------------------------//

    static int& _event_set()
    {
        static thread_local int _instance = PAPI_NULL;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
