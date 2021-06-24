// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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

#include "timemory/components/base.hpp"
#include "timemory/components/papi/backends.hpp"
#include "timemory/components/papi/papi_common.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
template <size_t MaxNumEvents>
struct papi_array
: public base<papi_array<MaxNumEvents>, std::array<long long, MaxNumEvents>>
, private policy::instance_tracker<papi_array<MaxNumEvents>>
, private papi_common
{
    using size_type         = size_t;
    using event_list        = std::vector<int>;
    using value_type        = std::array<long long, MaxNumEvents>;
    using entry_type        = typename value_type::value_type;
    using this_type         = papi_array<MaxNumEvents>;
    using base_type         = base<this_type, value_type>;
    using storage_type      = typename base_type::storage_type;
    using tracker_type      = policy::instance_tracker<this_type>;
    using get_initializer_t = std::function<event_list()>;
    using common_type       = void;

    static const short precision = 3;
    static const short width     = 8;

    template <typename Tp>
    using array_t = std::array<Tp, MaxNumEvents>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

    //----------------------------------------------------------------------------------//

    static auto& get_initializer() { return papi_common::get_initializer<common_type>(); }
    static void  configure()
    {
        if(!is_configured<common_type>())
            papi_common::initialize<common_type>();
    }
    static void initialize() { configure(); }
    static void thread_finalize()
    {
        papi_common::finalize<common_type>();
        papi_common::finalize_papi();
    }
    static void finalize() { papi_common::finalize<common_type>(); }

    //----------------------------------------------------------------------------------//

    papi_array() { events = get_events<common_type>(); }
    ~papi_array()                     = default;
    papi_array(const papi_array&)     = default;
    papi_array(papi_array&&) noexcept = default;
    papi_array& operator=(const papi_array&) = default;
    papi_array& operator=(papi_array&&) noexcept = default;

    using base_type::load;

    //----------------------------------------------------------------------------------//

    size_t size() { return events.size(); }

    //----------------------------------------------------------------------------------//

    static value_type record()
    {
        value_type read_value{};
        read_value.fill(0);
        if(is_configured<common_type>())
            papi::read(event_set<common_type>(), read_value.data());
        return read_value;
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp = double>
    std::vector<Tp> get() const
    {
        std::vector<Tp> values;
        auto&           _data = load();
        values.reserve(_data.size());
        for(auto& itr : _data)
            values.emplace_back(itr);
        values.resize(events.size());
        return values;
    }

    //----------------------------------------------------------------------------------//
    // sample
    //
    void sample()
    {
        if(tracker_type::get_thread_started() == 0)
            configure();
        if(events.empty())
            events = get_events<common_type>();

        tracker_type::start();
        value = record();
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        if(tracker_type::get_thread_started() == 0 || events.size() == 0)
            configure();

        events = get_events<common_type>();
        tracker_type::start();
        value = record();
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        tracker_type::stop();
        using namespace tim::component::operators;
        value = (record() - value);
        accum += value;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < events.size(); ++i)
            accum[i] += rhs.accum[i];
        for(size_type i = 0; i < events.size(); ++i)
            value[i] += rhs.value[i];
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < events.size(); ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < events.size(); ++i)
            value[i] -= rhs.value[i];
        return *this;
    }

    //----------------------------------------------------------------------------------//

protected:
    using base_type::accum;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;
    using papi_common::events;

    friend struct base<this_type, value_type>;
    friend class impl::storage<this_type,
                               trait::uses_value_storage<this_type, value_type>::value>;

public:
    //==================================================================================//
    //
    //      data representation
    //
    //==================================================================================//

    static std::string label()
    {
        return "papi_array" + std::to_string((event_set<common_type>() < 0)
                                                 ? 0
                                                 : event_set<common_type>());
    }

    static std::string description() { return "Fixed-size array of PAPI HW counters"; }

    TIMEMORY_NODISCARD entry_type get_display(int evt_type) const
    {
        return accum.at(evt_type);
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum), cereal::make_nvp("events", events));
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        array_t<double> _disp;
        for(size_type i = 0; i < events.size(); ++i)
            _disp[i] = get_display(i);
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("repr_data", _disp),
           cereal::make_nvp("value", value), cereal::make_nvp("accum", accum),
           cereal::make_nvp("display", _disp), cereal::make_nvp("events", events));
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    TIMEMORY_NODISCARD std::vector<std::string> label_array() const
    {
        std::vector<std::string> arr(events.size());
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).short_descr;

        for(auto& itr : arr)
        {
            size_t n = std::string::npos;
            while((n = itr.find("L/S")) != std::string::npos)
                itr.replace(n, 3, "Loads_Stores");
        }

        for(auto& itr : arr)
        {
            size_t n = std::string::npos;
            while((n = itr.find('/')) != std::string::npos)
                itr.replace(n, 1, "_per_");
        }

        for(auto& itr : arr)
        {
            size_t n = std::string::npos;
            while((n = itr.find(' ')) != std::string::npos)
                itr.replace(n, 1, "_");

            while((n = itr.find("__")) != std::string::npos)
                itr.replace(n, 2, "_");
        }

        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    TIMEMORY_NODISCARD std::vector<std::string> description_array() const
    {
        std::vector<std::string> arr(events.size());
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).long_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    TIMEMORY_NODISCARD std::vector<std::string> display_unit_array() const
    {
        std::vector<std::string> arr(events.size());
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).units;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    TIMEMORY_NODISCARD std::vector<int64_t> unit_array() const
    {
        std::vector<int64_t> arr(events.size());
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = 1;
        return arr;
    }

    //----------------------------------------------------------------------------------//

    TIMEMORY_NODISCARD string_t get_display() const
    {
        if(events.size() == 0)
            return "";
        auto val          = load();
        auto _get_display = [&](std::ostream& os, size_type idx) {
            auto     _obj_value = val[idx];
            auto     _evt_type  = events[idx];
            string_t _label     = papi::get_event_info(_evt_type).short_descr;
            string_t _disp      = papi::get_event_info(_evt_type).units;
            auto     _prec      = base_type::get_precision();
            auto     _width     = base_type::get_width();
            auto     _flags     = base_type::get_format_flags();

            std::stringstream ss;
            std::stringstream ssv;
            std::stringstream ssi;
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
        {
            ss_extra << " " << _disp;
        }
        else if(!_label.empty())
        {
            ss_extra << " " << _label;
        }
        os << ss_value.str() << ss_extra.str();
        return os;
    }
};
}  // namespace component
}  // namespace tim
