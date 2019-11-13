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
#include "timemory/utility/storage.hpp"

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//                      Compile-time constant set of PAPI counters
//
//--------------------------------------------------------------------------------------//

template <int... EventTypes>
struct papi_tuple
: public base<papi_tuple<EventTypes...>, std::array<long long, sizeof...(EventTypes)>,
              policy::thread_init, policy::thread_finalize>
{
    using size_type  = std::size_t;
    using value_type = std::array<long long, sizeof...(EventTypes)>;
    using entry_type = typename value_type::value_type;
    using this_type  = papi_tuple<EventTypes...>;
    using base_type =
        base<this_type, value_type, policy::thread_init, policy::thread_finalize>;
    using storage_type = typename base_type::storage_type;

    static const size_type num_events = sizeof...(EventTypes);
    template <typename _Tp>
    using array_t = std::array<_Tp, num_events>;

public:
    //==================================================================================//
    //
    //      static data
    //
    //==================================================================================//

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

    static int& event_set()
    {
        static thread_local int _instance = PAPI_NULL;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static bool& enable_multiplex()
    {
        static thread_local bool _instance = settings::papi_multiplexing();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void configure()
    {
        if(!is_configured())
        {
            if(!initialize_papi())
                return;
            // set overhead to zero
            apply<void>::set_value(get_overhead_values(), 0);
            tim::papi::create_event_set(&event_set(), enable_multiplex());
            tim::papi::add_events(event_set(), get_events().data(), num_events);
            tim::papi::start(event_set());
            is_configured() = true;
        }
    }

    //----------------------------------------------------------------------------------//

    static void invoke_thread_init(storage_type*) { configure(); }

    //----------------------------------------------------------------------------------//

    static void invoke_thread_finalize(storage_type*)
    {
        if(initialize_papi())
        {
            value_type values;
            tim::papi::stop(event_set(), values.data());
            tim::papi::remove_events(event_set(), get_events().data(), num_events);
            tim::papi::destroy_event_set(event_set());
            event_set() = PAPI_NULL;
            papi::unregister_thread();
        }
    }

    //----------------------------------------------------------------------------------//

    static value_type record()
    {
        if(is_configured())
            tim::papi::read(event_set(), get_read_values().data());
        return get_read_values();
    }

private:
    //----------------------------------------------------------------------------------//

    static value_type& get_overhead_values()
    {
        static thread_local value_type _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static value_type& get_read_values()
    {
        static auto _get_read_values = []() {
            value_type values;
            apply<void>::set_value(values, 0);
            return values;
        };
        static thread_local value_type _instance = _get_read_values();
        return _instance;
    }

public:
    static value_type get_overhead() { return get_overhead_values(); }

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
    //      construction
    //
    //==================================================================================//

    papi_tuple()
    {
        apply<void>::set_value(value, 0);
        apply<void>::set_value(accum, 0);
    }

    ~papi_tuple() {}

    papi_tuple(const papi_tuple& rhs) = default;
    this_type& operator=(const this_type& rhs) = default;
    papi_tuple(papi_tuple&& rhs)               = default;
    this_type& operator=(this_type&&) = default;

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        configure();
        set_started();
        value = std::move(record());
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void stop()
    {
        auto tmp      = std::move(record());
        auto overhead = get_overhead_values();
        // account for the overhead of recording (relevant for load/store)
        for(uint64_t i = 0; i < tmp.size(); ++i)
        {
            tmp[i] -= overhead[i];
            value[i] -= overhead[i];
        }
        for(size_type i = 0; i < num_events; ++i)
            accum[i] += (tmp[i] - value[i]);
        value = std::move(tmp);
        set_stopped();
    }

    //----------------------------------------------------------------------------------//
    // operators
    //
    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
            accum[i] += rhs.accum[i];
        for(size_type i = 0; i < num_events; ++i)
            value[i] += rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < num_events; ++i)
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
    static const short precision = 3;
    static const short width     = 12;

    // leave these empty
    static std::string label() { return "papi" + std::to_string(event_set()); }
    static std::string description() { return ""; }
    static std::string display_unit() { return ""; }
    static int64_t     unit() { return 1; }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        array_t<double> _disp;
        array_t<double> _value;
        array_t<double> _accum;
        for(size_type i = 0; i < num_events; ++i)
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

    entry_type get_display(int evt_type) const
    {
        auto val = (is_transient) ? accum[evt_type] : value[evt_type];
        return val;
    }

    string_t get_display() const
    {
        auto val          = (is_transient) ? accum : value;
        auto _get_display = [&](std::ostream& os, size_type idx) {
            auto     _obj_value = val[idx];
            auto     _evt_type  = get_events()[idx];
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
        for(size_type i = 0; i < num_events; ++i)
        {
            _get_display(ss, i);
            if(i + 1 < num_events)
                ss << ", ";
        }
        return ss.str();
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
    // array of descriptions
    //
    static array_t<std::string> label_array()
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = papi::get_event_info(get_events()[i]).short_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    static array_t<std::string> descript_array()
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = papi::get_event_info(get_events()[i]).long_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    static array_t<std::string> display_unit_array()
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = papi::get_event_info(get_events()[i]).units;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    static array_t<int64_t> unit_array()
    {
        array_t<int64_t> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = 1;
        return arr;
    }

private:
    //----------------------------------------------------------------------------------//
    //  array of events
    //
    static std::vector<int>& get_events()
    {
        static std::vector<int> _events = { EventTypes... };
        return _events;
    }

    static bool& is_configured()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
