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
 * \headerfile cupti.hpp "timemory/cupti.hpp"
 * Provides implementation of CUPTI routines.
 *
 */

#include "timemory/cupti.hpp"

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          CUPTI component
//
//--------------------------------------------------------------------------------------//

struct cupti_event
: public base<cupti_event, std::vector<cupti::impl::kernel_data_t>>
, public static_counted_object<cupti_event>
{
    using size_type     = std::size_t;
    using kernel_data_t = cupti::impl::kernel_data_t;
    using value_type    = std::vector<kernel_data_t>;
    using entry_type    = typename value_type::value_type;
    using base_type     = base<cupti_event, value_type>;
    using this_type     = cupti_event;
    using event_count   = static_counted_object<cupti_event>;
    using strvec_t      = std::vector<std::string>;
    using event_val_t   = kernel_data_t::event_val_t;
    using metric_val_t  = kernel_data_t::metric_val_t;

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

    cupti::profiler* profiler = nullptr;

    size_type size() { return (is_transient) ? accum.size() : value.size(); }

    cupti_event() {}

    cupti_event(const strvec_t& events, const strvec_t& metrics, const int device_num = 0)
    {
        init(events, metrics, device_num);
    }

    void init(const strvec_t& events, const strvec_t& metrics, const int device_num = 0)
    {
        delete profiler;
        profiler = new cupti::profiler(events, metrics, device_num);
    }

    ~cupti_event()
    {
        if(event_count::live() < 1 && event_count::is_master())
        {
        }
    }

    cupti_event(const cupti_event& rhs) = default;
    this_type& operator=(const this_type& rhs) = default;
    cupti_event(cupti_event&& rhs)             = default;
    this_type& operator=(this_type&&) = default;

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
    }

    static int64_t unit() { return 1; }
    // leave these empty
    static std::string label() { return "cupti"; }
    static std::string descript() { return ""; }
    static std::string display_unit() { return ""; }

    static value_type record() { return value_type{}; }

    entry_type compute_display(int evt_type) const
    {
        auto val = (is_transient) ? accum[evt_type] : value[evt_type];
        return val;
    }

    string_t compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        /*
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
        };*/
        std::stringstream ss;
        /*for(size_type i = 0; i < size(); ++i)
        {
            _compute_display(ss, i);
            if(i + 1 < size())
                ss << ", ";
        }*/
        return ss.str();
    }

    template <typename _Tp>
    using array_t = std::vector<_Tp>;

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    array_t<std::string> label_array()
    {
        array_t<std::string> arr;
        if(profiler)
        {
            for(const auto& itr : profiler->get_event_names())
                arr.push_back(itr);
            for(const auto& itr : profiler->get_metric_names())
                arr.push_back(itr);
        }
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    array_t<std::string> descript_array() { return label_array(); }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    array_t<std::string> display_unit_array()
    {
        array_t<std::string> arr;
        /*
        int                  evt_types[] = { EventTypes... };
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = display_unit(evt_types[i]);
        */
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    array_t<int64_t> unit_array()
    {
        array_t<int64_t> arr;
        for(size_type i = 0; i < size(); ++i)
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
        for(size_type i = 0; i < size(); ++i)
        {
            accum[i] += (tmp[i] - value[i]);
        }
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < size(); ++i)
            accum[i] += rhs.accum[i];
        for(size_type i = 0; i < size(); ++i)
            value[i] += rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < size(); ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < size(); ++i)
            value[i] -= rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    value_type serialization() { return accum; }
};

}  // namespace component
}  // namespace tim
