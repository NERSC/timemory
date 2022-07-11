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

#ifndef TIMEMORY_COMPONENTS_PAPI_PAPI_VECTOR_CPP_
#define TIMEMORY_COMPONENTS_PAPI_PAPI_VECTOR_CPP_ 1

#include "timemory/components/papi/macros.hpp"

#if defined(TIMEMORY_PAPI_SOURCE) && TIMEMORY_PAPI_SOURCE > 0
#    include "timemory/components/papi/papi_vector.hpp"
#elif !defined(TIMEMORY_PAPI_COMPONENT_HEADER_ONLY_MODE) ||                              \
    TIMEMORY_PAPI_COMPONENT_HEADER_ONLY_MODE == 0
#    include "timemory/components/papi/papi_vector.hpp"
#endif

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
papi_vector::configure()
{
    if(!is_configured<common_type>())
        papi_common::initialize<common_type>();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
papi_vector::initialize()
{
    configure();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
papi_vector::thread_finalize()
{
    papi_common::finalize<common_type>();
    papi_common::finalize_papi();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
papi_vector::finalize()
{
    papi_common::finalize<common_type>();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector::papi_vector()
{
    events = get_events<common_type>();
    value.resize(events.size(), 0);
    accum.resize(events.size(), 0);
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
size_t
papi_vector::size()
{
    return events.size();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector::value_type
papi_vector::record()
{
    value_type read_value(events.size(), 0);
    if(is_configured<common_type>())
        papi::read(event_set<common_type>(), read_value.data());
    return read_value;
}

//----------------------------------------------------------------------------------//
// sample
//
TIMEMORY_PAPI_INLINE
void
papi_vector::sample()
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
TIMEMORY_PAPI_INLINE
void
papi_vector::start()
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

TIMEMORY_PAPI_INLINE
void
papi_vector::stop()
{
    using namespace tim::component::operators;
    tracker_type::stop();
    value = (record() - value);
    accum += value;
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector&
papi_vector::operator+=(const papi_vector& rhs)
{
    using namespace tim::component::operators;
    value += rhs.value;
    accum += rhs.accum;
    return *this;
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector&
papi_vector::operator-=(const papi_vector& rhs)
{
    using namespace tim::component::operators;
    value -= rhs.value;
    accum -= rhs.accum;
    return *this;
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::string
papi_vector::label()
{
    auto _event_set = event_set<common_type>();
    if(_event_set > 0)
        return "papi_vector" + std::to_string(_event_set);
    return "papi_vector";
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::string
papi_vector::description()
{
    return "Dynamically allocated array of PAPI HW counters";
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector::entry_type
papi_vector::get_display(int evt_type) const
{
    return accum.at(evt_type);
}

//----------------------------------------------------------------------------------//
// array of descriptions
//
TIMEMORY_PAPI_INLINE
std::vector<std::string>
papi_vector::label_array() const
{
    std::vector<std::string> arr = events;
    for(size_type i = 0; i < events.size(); ++i)
    {
        papi::event_info_t _info = papi::get_event_info(events.at(i));
        if(!_info.modified_short_descr)
            arr.at(i) = _info.short_descr;
    }

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
TIMEMORY_PAPI_INLINE
std::vector<std::string>
papi_vector::description_array() const
{
    std::vector<std::string> arr(events.size(), "");
    for(size_type i = 0; i < events.size(); ++i)
        arr[i] = papi::get_event_info(events[i]).long_descr;
    return arr;
}

//----------------------------------------------------------------------------------//
// array of unit
//
TIMEMORY_PAPI_INLINE
std::vector<std::string>
papi_vector::display_unit_array() const
{
    std::vector<std::string> arr(events.size(), "");
    for(size_type i = 0; i < events.size(); ++i)
        arr[i] = papi::get_event_info(events[i]).units;
    return arr;
}

//----------------------------------------------------------------------------------//
// array of unit values
//
TIMEMORY_PAPI_INLINE
std::vector<int64_t>
papi_vector::unit_array() const
{
    std::vector<int64_t> arr(events.size(), 0);
    for(size_type i = 0; i < events.size(); ++i)
        arr[i] = 1;
    return arr;
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::string
papi_vector::get_display() const
{
    if(events.empty())
        return "";
    auto val          = load();
    auto _get_display = [&](std::ostream& os, size_type idx) {
        auto        _obj_value = val.at(idx);
        auto        _evt_type  = events.at(idx);
        std::string _label     = papi::get_event_info(_evt_type).short_descr;
        std::string _disp      = papi::get_event_info(_evt_type).units;
        auto        _prec      = base_type::get_precision();
        auto        _width     = base_type::get_width();
        auto        _flags     = base_type::get_format_flags();

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

TIMEMORY_PAPI_INLINE
std::ostream&
papi_vector::write(std::ostream& os) const
{
    if(events.empty())
        return os;
    // output the metrics
    auto _value = get_display();
    auto _label = get_label();
    auto _disp  = display_unit();
    auto _prec  = get_precision();
    auto _width = get_width();
    auto _flags = get_format_flags();

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
}  // namespace component
}  // namespace tim

#endif
