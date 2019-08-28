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
#include "timemory/components/timing.hpp"
#include "timemory/components/types.hpp"
#include "timemory/details/settings.hpp"
#include "timemory/ert/configuration.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/ert/kernels.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <memory>
#include <numeric>
#include <utility>

// default vectorization width
#if !defined(TIMEMORY_VEC)
#    define TIMEMORY_VEC 256
#endif

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
// this computes the numerator of the roofline for a given set of PAPI counters.
// e.g. for FLOPS roofline (floating point operations / second:
//
//  single precision:
//              cpu_roofline<float>
//
//  double precision:
//              cpu_roofline<double>
//
//  generic:
//              cpu_roofline<T, ...>
//
template <typename... _Types>
struct cpu_roofline
: public base<cpu_roofline<_Types...>, std::pair<std::vector<long long>, double>,
              policy::thread_init, policy::thread_finalize, policy::global_finalize,
              policy::serialization>
{
    // static_assert(is_one_of<cuda::fp16_t, std::tuple<_Types...>>::value,
    //              "Error! No CPU roofline support for cuda::fp16_t");

    using size_type  = std::size_t;
    using event_type = std::vector<int>;
    using array_type = std::vector<long long>;
    using data_type  = long long*;
    using value_type = std::pair<array_type, double>;
    using this_type  = cpu_roofline<_Types...>;
    using base_type =
        base<this_type, value_type, policy::thread_init, policy::thread_finalize,
             policy::global_finalize, policy::serialization>;

    using device_t    = device::cpu;
    using clock_type  = real_clock;
    using ratio_t     = typename clock_type::ratio_t;
    using types_tuple = std::tuple<_Types...>;

    using ert_data_t     = ert::exec_data;
    using ert_params_t   = ert::exec_params;
    using ert_data_ptr_t = std::shared_ptr<ert_data_t>;

    // short-hand for variadic expansion
    template <typename _Tp>
    using ert_config_type = ert::configuration<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_counter_type = ert::counter<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_executor_type = ert::executor<device_t, _Tp, ert_data_t, clock_type>;
    template <typename _Tp>
    using ert_callback_type = ert::callback<ert_executor_type<_Tp>>;

    // variadic expansion for ERT types
    using ert_config_t   = std::tuple<ert_config_type<_Types>...>;
    using ert_counter_t  = std::tuple<ert_counter_type<_Types>...>;
    using ert_executor_t = std::tuple<ert_executor_type<_Types>...>;
    using ert_callback_t = std::tuple<ert_callback_type<_Types>...>;

    static_assert(std::tuple_size<ert_config_t>::value ==
                      std::tuple_size<types_tuple>::value,
                  "Error! ert_config_t size does not match types_tuple size!");

    using iterator       = typename array_type::iterator;
    using const_iterator = typename array_type::const_iterator;

    static const short                   precision = 3;
    static const short                   width     = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    //----------------------------------------------------------------------------------//

    // collection mode, AI (arithmetic intensity) is the load/store: PAPI_LST_INS
    enum class MODE
    {
        OP,
        AI
    };

    //----------------------------------------------------------------------------------//

    using strvec_t          = std::vector<std::string>;
    using intvec_t          = std::vector<int>;
    using events_callback_t = std::function<intvec_t(const MODE&)>;

    //----------------------------------------------------------------------------------//

    static events_callback_t& get_events_callback()
    {
        static events_callback_t _instance = [](const MODE&) { return intvec_t{}; };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static int event_set() { return *_event_set_ptr(); }

    //----------------------------------------------------------------------------------//

    static const event_type& events() { return *_events_ptr(); }

    //----------------------------------------------------------------------------------//

    static size_type size() { return events().size(); }

    //----------------------------------------------------------------------------------//

    static MODE& event_mode()
    {
        auto aslc = [](std::string str) {
            for(auto& itr : str)
                itr = tolower(itr);
            return str;
        };

        auto _get = [=]() {
            // check the standard variable
            std::string _env = aslc(settings::cpu_roofline_mode());
            return (_env == "op" || _env == "hw" || _env == "counters")
                       ? MODE::OP
                       : ((_env == "ai" || _env == "ac" || _env == "activity")
                              ? MODE::AI
                              : MODE::OP);
        };

        static MODE _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static ert_config_t& get_finalizer()
    {
        static ert_config_t _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static ert_data_ptr_t& get_ert_data()
    {
        static ert_data_ptr_t _instance = ert_data_ptr_t(new ert_data_t);
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void invoke_thread_init()
    {
        // start PAPI counters
        if(event_mode() == MODE::OP)
        {
            if(is_one_of<float, types_tuple>::value)
                _events_ptr()->push_back(PAPI_SP_OPS);
            if(is_one_of<double, types_tuple>::value)
                _events_ptr()->push_back(PAPI_DP_OPS);
        } else if(event_mode() == MODE::AI)
        {
            _events_ptr()->push_back(PAPI_LST_INS);
        }

        // add in extra events
        auto _extra_events = get_events_callback()(event_mode());
        for(const auto& itr : _extra_events)
            _events_ptr()->push_back(itr);

        papi::create_event_set(_event_set_ptr());
        papi::add_events(event_set(), _events_ptr()->data(), size());
        papi::start(event_set(), settings::papi_multiplexing());
    }

    //----------------------------------------------------------------------------------//

    static void invoke_thread_finalize()
    {
        // store these for later
        _label_array() = label_array();
        // stop PAPI counters
        array_type event_values(events().size(), 0);
        papi::stop(event_set(), event_values.data());
        papi::remove_events(event_set(), _events_ptr()->data(), events().size());
        papi::destroy_event_set(event_set());
        delete _events_ptr();
        delete _event_set_ptr();
        _events_ptr()    = nullptr;
        _event_set_ptr() = nullptr;
    }

    //----------------------------------------------------------------------------------//

    template <typename _Tp, typename _Func>
    static void set_executor_callback(_Func&& f)
    {
        ert_executor_type<_Tp>::get_callback() = std::forward<_Func>(f);
    }

    //----------------------------------------------------------------------------------//

    static void invoke_global_finalize()
    {
        // run roofline peak generation
        auto ert_config = get_finalizer();
        auto ert_data   = get_ert_data();
        apply<void>::access<ert_executor_t>(ert_config, ert_data);
        if(ert_data && (settings::verbose() > 0 || settings::debug()))
            std::cout << *(ert_data) << std::endl;
    }

    //----------------------------------------------------------------------------------//

    template <typename _Archive>
    static void invoke_serialize(_Archive& ar, const unsigned int /*version*/)
    {
        auto& _ert_data = get_ert_data();
        if(!_ert_data.get())  // for input
            _ert_data.reset(new ert_data_t());
        ar(serializer::make_nvp("roofline", *_ert_data.get()));
    }

    //----------------------------------------------------------------------------------//

    static std::string get_mode_string()
    {
        return (event_mode() == MODE::OP) ? "op" : "ai";
    }

    //----------------------------------------------------------------------------------//

    static std::string get_type_string()
    {
        return apply<std::string>::join("_", demangle(typeid(_Types).name())...);
    }

    //----------------------------------------------------------------------------------//

    static int64_t unit() { return 1; }

    //----------------------------------------------------------------------------------//

    static std::string label()
    {
        return "cpu_roofline_" + get_type_string() + "_" + get_mode_string();
    }

    //----------------------------------------------------------------------------------//

    static std::string description()
    {
        return "cpu roofline " + get_type_string() + " " + get_mode_string();
    }

    //----------------------------------------------------------------------------------//

    static std::string display_unit()
    {
        std::stringstream ss;
        ss << "(";
        auto labels = _label_array();

        for(size_type i = 0; i < labels.size(); ++i)
        {
            ss << labels[i];
            if(i + 1 < labels.size())
                ss << " + ";
        }
        ss << ")";
        if(event_mode() == MODE::OP)
            ss << " / " << clock_type::display_unit();
        return ss.str();
    }

    //----------------------------------------------------------------------------------//

    static value_type record()
    {
        array_type read_values(size(), 0);
        papi::read(event_set(), read_values.data());
        auto delta_duration =
            clock_type::record() / static_cast<double>(ratio_t::den) * units::sec;
        return value_type(read_values, delta_duration);
    }

public:
    //----------------------------------------------------------------------------------//

    cpu_roofline()
    {
        value.first.resize(size(), 0);
        accum.first.resize(size(), 0);
        std::tie(value.second, accum.second) = std::make_pair(0, 0);
    }

    //----------------------------------------------------------------------------------//

    ~cpu_roofline() {}
    cpu_roofline(const cpu_roofline& rhs) = default;
    cpu_roofline(cpu_roofline&& rhs)      = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&&) = default;

    //----------------------------------------------------------------------------------//

    double get() const { return get_counted() / get_elapsed(); }

    //----------------------------------------------------------------------------------//

    void start()
    {
        set_started();
        value = record();
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        auto tmp = record();
        resize(std::max<size_type>(tmp.first.size(), value.first.size()));
        for(size_type i = 0; i < accum.first.size(); ++i)
            accum.first[i] += (tmp.first[i] - value.first[i]);
        accum.second += (tmp.second - value.second);
        value = std::move(tmp);
        set_stopped();
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        resize(std::max<size_type>(rhs.value.first.size(), rhs.accum.first.size()));
        for(size_type i = 0; i < accum.first.size(); ++i)
            accum.first[i] += rhs.accum.first[i];
        for(size_type i = 0; i < value.first.size(); ++i)
            value.first[i] += rhs.value.first[i];
        accum.second += rhs.accum.second;
        value.second += rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        resize(std::max<size_type>(rhs.value.first.size(), rhs.accum.first.size()));
        for(size_type i = 0; i < accum.first.size(); ++i)
            accum.first[i] -= rhs.accum.first[i];
        for(size_type i = 0; i < value.first.size(); ++i)
            value.first[i] -= rhs.value.first[i];
        accum.second -= rhs.accum.second;
        value.second -= rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    iterator begin()
    {
        auto& obj = (accum.second > 0) ? accum.first : value.first;
        return obj.begin();
    }

    //----------------------------------------------------------------------------------//

    const_iterator begin() const
    {
        const auto& obj = (accum.second > 0) ? accum.first : value.first;
        return obj.begin();
    }

    //----------------------------------------------------------------------------------//

    iterator end()
    {
        auto& obj = (accum.second > 0) ? accum.first : value.first;
        return obj.end();
    }

    //----------------------------------------------------------------------------------//

    const_iterator end() const
    {
        const auto& obj = (accum.second > 0) ? accum.first : value.first;
        return obj.end();
    }

    //----------------------------------------------------------------------------------//

    double get_elapsed(const int64_t& _unit = clock_type::get_unit()) const
    {
        auto& obj = (accum.second > 0) ? accum : value;
        return static_cast<double>(obj.second) *
               (static_cast<double>(_unit) / units::sec);
    }

    //----------------------------------------------------------------------------------//

    double get_counted() const
    {
        double _sum = 0.0;
        for(auto itr = begin(); itr != end(); ++itr)
            _sum += static_cast<double>(*itr);
        return _sum;
    }

protected:
    using base_type::accum;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    friend struct policy::wrapper<policy::thread_init, policy::thread_finalize,
                                  policy::global_finalize, policy::serialization>;

    friend struct base<this_type, value_type, policy::thread_init,
                       policy::thread_finalize, policy::global_finalize,
                       policy::serialization>;

    friend class storage<this_type>;

public:
    //==================================================================================//
    //
    //      representation as a string
    //
    //==================================================================================//

    double get_display() const
    {
        auto& obj = (accum.second > 0) ? accum : value;
        if(obj.second == 0)
            return 0.0;
        double _sum = 0.0;
        for(auto itr = begin(); itr != end(); ++itr)
            _sum += static_cast<double>(*itr);
        return (event_mode() == MODE::OP) ? (_sum / obj.second) : _sum;
    }

    //----------------------------------------------------------------------------------//

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        // output the time
        auto&             _obj = (obj.accum.second > 0) ? obj.accum : obj.value;
        std::stringstream sst;
        auto              t_value = _obj.second;
        auto              t_label = clock_type::get_label();
        auto              t_disp  = clock_type::get_display_unit();
        auto              t_prec  = clock_type::get_precision();
        auto              t_width = clock_type::get_width();
        auto              t_flags = clock_type::get_format_flags();

        sst.setf(t_flags);
        sst << std::setw(t_width) << std::setprecision(t_prec) << t_value;
        if(!t_disp.empty())
            sst << " " << t_disp;
        if(!t_label.empty())
            sst << " " << t_label;
        sst << ", ";

        // output the roofline metric
        auto _value = obj.get_display();
        auto _label = this_type::get_label();
        auto _disp  = this_type::display_unit();
        auto _prec  = clock_type::get_precision();
        auto _width = this_type::get_width();
        auto _flags = clock_type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;
        if(!_disp.empty())
            ss_extra << " " << _disp;
        else if(!_label.empty())
            ss_extra << " " << _label;
        os << sst.str() << ss_value.str() << ss_extra.str();

        return os;
    }

    //----------------------------------------------------------------------------------//

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto _disp = get_display();

        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("display", _disp),
           serializer::make_nvp("mode", get_mode_string()),
           serializer::make_nvp("type", get_type_string()));

        const auto& labels  = get_labels();
        auto        data    = (is_transient) ? accum.first : value.first;
        auto        runtime = (is_transient) ? accum.second : value.second;
        ar.setNextName("repr_data");
        ar.startNode();
        auto litr = labels.begin();
        auto ditr = data.begin();
        ar(serializer::make_nvp("runtime", runtime));
        for(; litr != labels.end() && ditr != data.end(); ++litr, ++ditr)
            ar(serializer::make_nvp(*litr, double(*ditr)));
        ar(serializer::make_nvp("counted", get_counted()),
           serializer::make_nvp("elapsed", get_elapsed()),
           serializer::make_nvp("fom", get()));
        ar.finishNode();

        ar(serializer::make_nvp("value", value), serializer::make_nvp("accum", accum));
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    static strvec_t label_array()
    {
        strvec_t arr;
        for(const auto& itr : events())
            arr.push_back(papi::get_event_info(itr).short_descr);
        return arr;
    }

    //----------------------------------------------------------------------------------//

    const strvec_t& get_labels() const { return m_label_array; }

private:
    //----------------------------------------------------------------------------------//
    // these are needed after the global label array is destroyed
    //
    size_type m_event_size  = events().size();
    strvec_t  m_label_array = label_array();

    //----------------------------------------------------------------------------------//

    void resize(size_type sz)
    {
        sz           = std::max<size_type>(size(), sz);
        m_event_size = std::max<size_type>(m_event_size, sz);
        value.first.resize(std::max<size_type>(sz, value.first.size()), 0);
        accum.first.resize(std::max<size_type>(sz, accum.first.size()), 0);
    }

private:
    //----------------------------------------------------------------------------------//

    static strvec_t& _label_array()
    {
        static thread_local strvec_t _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static event_type*& _events_ptr()
    {
        static thread_local event_type* _instance = new event_type;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static int*& _event_set_ptr()
    {
        static thread_local int* _instance = new int(PAPI_NULL);
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
// Shorthand aliases for common roofline types
//
using cpu_roofline_sp_flops = cpu_roofline<float>;
using cpu_roofline_dp_flops = cpu_roofline<double>;
using cpu_roofline_flops    = cpu_roofline<float, double>;

//--------------------------------------------------------------------------------------//
}  // namespace component

namespace trait
{
template <>
struct requires_json<component::cpu_roofline_sp_flops> : std::true_type
{};

template <>
struct requires_json<component::cpu_roofline_dp_flops> : std::true_type
{};

#if !defined(TIMEMORY_USE_PAPI)
template <>
struct is_available<component::cpu_roofline_sp_flops> : std::false_type
{};

template <>
struct is_available<component::cpu_roofline_dp_flops> : std::false_type
{};
#endif

}  // namespace trait

}  // namespace tim
