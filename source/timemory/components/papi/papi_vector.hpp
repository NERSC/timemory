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

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                          Array of PAPI counters
//
//--------------------------------------------------------------------------------------//
//
struct papi_vector
: public base<papi_vector, std::vector<long long>>
, private policy::instance_tracker<papi_vector>
, private papi_common
{
    template <typename Tp>
    using vector_t = std::vector<Tp>;

    using size_type         = size_t;
    using event_list        = vector_t<int>;
    using value_type        = vector_t<long long>;
    using entry_type        = typename value_type::value_type;
    using this_type         = papi_vector;
    using base_type         = base<this_type, value_type>;
    using storage_type      = typename base_type::storage_type;
    using get_initializer_t = std::function<event_list()>;
    using tracker_type      = policy::instance_tracker<this_type>;
    using common_type       = void;

    using tracker_type::m_thr;

    static const short precision = 3;
    static const short width     = 8;

    template <typename... T>
    friend struct cpu_roofline;

    template <typename... T>
    friend struct gpu_roofline;

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

    papi_vector()
    {
        events = get_events<common_type>();
        value.resize(events.size(), 0);
        accum.resize(events.size(), 0);
    }

    ~papi_vector()                      = default;
    papi_vector(const papi_vector&)     = default;
    papi_vector(papi_vector&&) noexcept = default;
    papi_vector& operator=(const papi_vector&) = default;
    papi_vector& operator=(papi_vector&&) noexcept = default;

    using base_type::load;

    //----------------------------------------------------------------------------------//

    size_t size() { return events.size(); }

    //----------------------------------------------------------------------------------//

    value_type record()
    {
        value_type read_value(events.size(), 0);
        if(is_configured<common_type>())
            papi::read(event_set<common_type>(), read_value.data());
        return read_value;
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp = double>
    vector_t<Tp> get() const
    {
        std::vector<Tp> values;
        const auto&     _data = load();
        for(const auto& itr : _data)
            values.push_back(itr);
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
        if(tracker_type::get_thread_started() == 0 || events.empty())
        {
            configure();
        }

        events = get_events<common_type>();
        value.resize(events.size(), 0);
        accum.resize(events.size(), 0);
        tracker_type::start();
        value = record();
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        using namespace tim::component::operators;
        tracker_type::stop();
        value = (record() - value);
        accum += value;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        value += rhs.value;
        accum += rhs.accum;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        return *this;
    }

    //----------------------------------------------------------------------------------//

protected:
    // data types
    using papi_common::events;

public:
    //==================================================================================//
    //
    //      data representation
    //
    //==================================================================================//

    static std::string label()
    {
        return "papi_vector" + std::to_string((event_set<common_type>() < 0)
                                                  ? 0
                                                  : event_set<common_type>());
    }

    static std::string description()
    {
        return "Dynamically allocated array of PAPI HW counters";
    }

    TIMEMORY_NODISCARD entry_type get_display(int evt_type) const
    {
        return accum.at(evt_type);
    }

    //----------------------------------------------------------------------------------//
    // load
    //
    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum), cereal::make_nvp("events", events));
    }

    //----------------------------------------------------------------------------------//
    // save
    //
    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        auto             sz = events.size();
        vector_t<double> _disp(sz, 0.0);
        for(size_type i = 0; i < sz; ++i)
        {
            _disp[i] = get_display(i);
        }
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("repr_data", _disp),
           cereal::make_nvp("value", value), cereal::make_nvp("accum", accum),
           cereal::make_nvp("display", _disp), cereal::make_nvp("events", events));
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    TIMEMORY_NODISCARD vector_t<std::string> label_array() const
    {
        vector_t<std::string> arr(events.size(), "");
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
    TIMEMORY_NODISCARD vector_t<std::string> description_array() const
    {
        vector_t<std::string> arr(events.size(), "");
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).long_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    TIMEMORY_NODISCARD vector_t<std::string> display_unit_array() const
    {
        vector_t<std::string> arr(events.size(), "");
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).units;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    TIMEMORY_NODISCARD vector_t<int64_t> unit_array() const
    {
        vector_t<int64_t> arr(events.size(), 0);
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = 1;
        return arr;
    }

    //----------------------------------------------------------------------------------//

    TIMEMORY_NODISCARD string_t get_display() const
    {
        if(events.empty())
            return "";
        auto val          = load();
        auto _get_display = [&](std::ostream& os, size_type idx) {
            auto     _obj_value = val.at(idx);
            auto     _evt_type  = events.at(idx);
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
        if(obj.events.empty())
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
