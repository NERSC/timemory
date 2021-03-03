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
/// \struct tim::component::papi_tuple
/// \tparam EventTypes Compile-time constant list of PAPI event identifiers
///
/// \brief This component is useful for bundling together a fixed set of hardware counter
/// identifiers which require no runtime configuration
///
/// \code{.cpp}
/// // the "Instructions" alias below explicitly collects the total instructions,
/// // the number of load instructions, the number of store instructions
/// using Instructions = papi_tuple<PAPI_TOT_INS, PAPI_LD_INS, PAPI_SR_INS>;
///
/// Instructions inst{};
/// inst.start();
/// ...
/// inst.stop();
/// std::vector<double> data = inst.get();
///
/// \endcode
template <int... EventTypes>
struct papi_tuple
: public base<papi_tuple<EventTypes...>, std::array<long long, sizeof...(EventTypes)>>
, private policy::instance_tracker<papi_tuple<EventTypes...>>
, private papi_common
{
    using size_type    = std::size_t;
    using value_type   = std::array<long long, sizeof...(EventTypes)>;
    using entry_type   = typename value_type::value_type;
    using this_type    = papi_tuple<EventTypes...>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using tracker_type = policy::instance_tracker<papi_tuple<EventTypes...>>;
    using common_type  = this_type;

    static const size_type num_events = sizeof...(EventTypes);
    template <typename Tp>
    using array_t = std::array<Tp, num_events>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

public:
    //----------------------------------------------------------------------------------//

    static void configure()
    {
        if(!is_configured<common_type>())
        {
            papi_common::get_initializer<common_type>() = []() {
                return std::vector<int>({ EventTypes... });
            };
            papi_common::get_events<common_type>() = { EventTypes... };
            papi_common::initialize<common_type>();
        }
    }
    static void thread_init() { this_type::configure(); }
    static void thread_finalize()
    {
        papi_common::finalize<common_type>();
        papi_common::finalize_papi();
    }
    static void initialize() { configure(); }
    static void finalize() { papi_common::finalize<common_type>(); }

    //----------------------------------------------------------------------------------//

    static value_type record()
    {
        if(is_configured<common_type>())
            tim::papi::read(event_set<common_type>(), get_read_values().data());
        return get_read_values();
    }

private:
    static value_type& get_read_values()
    {
        static thread_local value_type _instance = []() {
            value_type values;
            mpl::apply<void>::set_value(values, 0);
            return values;
        }();
        return _instance;
    }

protected:
    using base_type::accum;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    friend struct base<this_type, value_type>;
    friend class impl::storage<this_type,
                               trait::uses_value_storage<this_type, value_type>::value>;

public:
    //==================================================================================//
    //
    //      construction
    //
    //==================================================================================//

    TIMEMORY_DEFAULT_OBJECT(papi_tuple)

    using base_type::load;

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
        {
            configure();
            events = get_events<common_type>();
        }

        tracker_type::start();
        value = record();
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void stop()
    {
        tracker_type::stop();
        using namespace tim::component::operators;
        value = (record() - value);
        accum += value;
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
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < num_events; ++i)
            value[i] -= rhs.value[i];
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
    static std::string label()
    {
        return "papi" + std::to_string(event_set<common_type>());
    }
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
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("repr_data", _disp),
           cereal::make_nvp("value", _value), cereal::make_nvp("accum", _accum),
           cereal::make_nvp("display", _disp));
    }

    entry_type get_display(int evt_type) const { return accum[evt_type]; }

    TIMEMORY_NODISCARD string_t get_display() const
    {
        auto val          = load();
        auto _get_display = [&](std::ostream& os, size_type idx) {
            auto     _obj_value = val[idx];
            auto     _evt_type  = std::vector<int>({ EventTypes... }).at(idx);
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
            {
                ssv << " " << _disp;
            }
            else if(!_label.empty())
            {
                ssi << " " << _label;
            }
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

    template <typename Lhs, typename Rhs, size_t N, size_t... Idx>
    static void convert(std::array<Lhs, N>& lhs, const std::array<Rhs, N>& rhs,
                        std::index_sequence<Idx...>)
    {
        TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(lhs) =
                                     static_cast<Lhs>(std::get<Idx>(rhs)));
    }
    template <typename Tp = double>
    auto get() const
    {
        std::array<Tp, num_events> values;
        auto&                      _data = load();
        convert(values, _data, std::make_index_sequence<num_events>{});
        return values;
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    static array_t<std::string> label_array()
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = papi::get_event_info(get_events<common_type>().at(i)).short_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    static array_t<std::string> description_array()
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = papi::get_event_info(get_events<common_type>().at(i)).long_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    static array_t<std::string> display_unit_array()
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = papi::get_event_info(get_events<common_type>().at(i)).units;
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
};
}  // namespace component
}  // namespace tim
