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

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/macros.hpp"
#include "timemory/rusage.hpp"
#include "timemory/serializer.hpp"
#include "timemory/storage.hpp"
#include "timemory/units.hpp"

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

template <int EventSet, std::size_t NumEvent>
struct papi_array
: public base<papi_array<EventSet, NumEvent>, std::array<long long, NumEvent>>
, public static_counted_object<papi_array<EventSet, 0>>
{
    using size_type   = std::size_t;
    using event_list  = std::array<int, NumEvent>;
    using value_type  = std::array<long long, NumEvent>;
    using entry_type  = typename value_type::value_type;
    using base_type   = base<papi_array<EventSet, NumEvent>, value_type>;
    using this_type   = papi_array<EventSet, NumEvent>;
    using event_count = static_counted_object<papi_array<EventSet, 0>>;

    static const short                   precision = 6;
    static const short                   width     = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::scientific | std::ios_base::dec;

    using base_type::accum;
    using base_type::is_running;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;
    using event_count::m_count;

    template <typename _Tp>
    using array_t = std::array<_Tp, NumEvent>;

    event_list evt_types;
    papi_array(const event_list& _evts)
    : evt_types(_evts)
    {
        if(event_count::is_master())
        {
            // add_event_types();
            start_event_set();
        }
        apply<void>::set_value(value, 0);
        apply<void>::set_value(accum, 0);
    }

    ~papi_array()
    {
        if(event_count::live() < 1 && event_count::is_master())
        {
            stop_event_set();
            // remove_event_types();
        }
    }

    papi_array(const papi_array& rhs) = default;
    this_type& operator=(const this_type& rhs) = default;
    papi_array(papi_array&& rhs)               = default;
    this_type& operator=(this_type&&) = default;

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        array_t<double> _disp;
        array_t<double> _value;
        array_t<double> _accum;
        for(size_type i = 0; i < NumEvent; ++i)
        {
            _disp[i]  = compute_display(i);
            _value[i] = value[i];
            _accum[i] = accum[i];
        }
        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("value", _value),
           serializer::make_nvp("accum", _accum), serializer::make_nvp("display", _disp));
    }

    static PAPI_event_info_t info(int evt_type)
    {
        PAPI_event_info_t evt_info;
#if defined(TIMEMORY_USE_PAPI)
        PAPI_get_event_info(evt_type, &evt_info);
#else
        consume_parameters(std::move(evt_type));
#endif
        return evt_info;
    }

    static int64_t unit() { return 1; }
    // leave these empty
    static std::string label() { return "papi" + std::to_string(EventSet); }
    static std::string descript() { return ""; }
    static std::string display_unit() { return ""; }
    // use these instead
    static std::string label(int evt_type) { return info(evt_type).short_descr; }
    static std::string descript(int evt_type) { return info(evt_type).long_descr; }
    static std::string display_unit(int evt_type) { return info(evt_type).units; }

    static value_type record()
    {
        value_type read_value;
        apply<void>::set_value(read_value, 0);
        if(event_count::is_master())
            tim::papi::read(EventSet, read_value.data());
        return read_value;
    }

    entry_type compute_display(int evt_type) const
    {
        auto val = (is_transient) ? accum[evt_type] : value[evt_type];
        return val;
    }

    string_t compute_display() const
    {
        auto val              = (is_transient) ? accum : value;
        auto _compute_display = [&](std::ostream& os, size_type idx) {
            auto _obj_value = val[idx];
            auto _evt_type  = evt_types[idx];
            auto _label     = label(_evt_type);
            auto _disp      = display_unit(_evt_type);
            auto _prec      = base_type::get_precision();
            auto _width     = base_type::get_width();
            auto _flags     = base_type::get_format_flags();

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
        for(size_type i = 0; i < NumEvent; ++i)
        {
            _compute_display(ss, i);
            if(i + 1 < NumEvent)
                ss << ", ";
        }
        return ss.str();
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    array_t<std::string> label_array() const
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < NumEvent; ++i)
            arr[i] = label(evt_types[i]);
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    array_t<std::string> descript_array() const
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < NumEvent; ++i)
            arr[i] = descript(evt_types[i]);
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    array_t<std::string> display_unit_array() const
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < NumEvent; ++i)
            arr[i] = display_unit(evt_types[i]);
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    array_t<int64_t> unit_array() const
    {
        array_t<int64_t> arr;
        for(size_type i = 0; i < NumEvent; ++i)
            arr[i] = 1;
        return arr;
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
        for(size_type i = 0; i < NumEvent; ++i)
        {
            accum[i] += (tmp[i] - value[i]);
        }
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < NumEvent; ++i)
            accum[i] += rhs.accum[i];
        for(size_type i = 0; i < NumEvent; ++i)
            value[i] += rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < NumEvent; ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < NumEvent; ++i)
            value[i] -= rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    value_type serialization() { return accum; }

private:
    inline bool acquire_claim(std::atomic<bool>& m_check)
    {
        bool is_set = m_check.load(std::memory_order_relaxed);
        if(is_set)
            return false;
        return m_check.compare_exchange_strong(is_set, true, std::memory_order_relaxed);
    }

    inline bool release_claim(std::atomic<bool>& m_check)
    {
        bool is_set = m_check.load(std::memory_order_relaxed);
        if(!is_set)
            return false;
        return m_check.compare_exchange_strong(is_set, false, std::memory_order_relaxed);
    }

    static std::atomic<bool>& event_type_added()
    {
        static std::atomic<bool> instance(false);
        return instance;
    }

    static std::atomic<bool>& event_set_started()
    {
        static std::atomic<bool> instance(false);
        return instance;
    }

    void add_event_types()
    {
        if(acquire_claim(event_type_added()))
        {
            tim::papi::add_events(EventSet, evt_types, NumEvent);
        }
    }

    void remove_event_types()
    {
        if(release_claim(event_type_added()))
        {
            tim::papi::remove_events(EventSet, evt_types, NumEvent);
        }
    }

    void start_event_set()
    {
        if(acquire_claim(event_set_started()))
        {
            tim::papi::start_counters(evt_types, NumEvent);
        }
    }

    void stop_event_set()
    {
        if(release_claim(event_set_started()))
        {
#if defined(_WINDOWS)
            for(std::size_t i = 0; i < NumEvent; ++i)
                evt_types[i] = 0;
#else
            apply<void>::set_value(evt_types, 0);
#endif
            tim::papi::stop_counters(evt_types.data(), NumEvent);
        }
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
