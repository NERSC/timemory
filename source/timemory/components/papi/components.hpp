//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

/**
 * \file timemory/components/papi/components.hpp
 * \brief Implementation of the papi component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/papi/backends.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/components/timing/wall_clock.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                          Common PAPI configuration
//
//--------------------------------------------------------------------------------------//
//
struct papi_common
{
public:
    template <typename Tp>
    using vector_t = std::vector<Tp>;

    template <typename Tp, size_t N>
    using array_t = std::array<Tp, N>;

    using size_type         = size_t;
    using event_list        = vector_t<int>;
    using value_type        = vector_t<long long>;
    using entry_type        = typename value_type::value_type;
    using get_initializer_t = std::function<event_list()>;

    //----------------------------------------------------------------------------------//

    struct common_data
    {
        TIMEMORY_DEFAULT_OBJECT(common_data)

        bool          is_configured = false;
        bool          is_fixed      = false;
        int           event_set     = PAPI_NULL;
        vector_t<int> events        = {};
    };

    //----------------------------------------------------------------------------------//

    template <typename Tp>
    static common_data& data()
    {
        static thread_local common_data _instance{};
        return _instance;
    }

    template <typename Tp>
    static int& event_set()
    {
        return data<Tp>().event_set;
    }

    template <typename Tp>
    static bool& is_configured()
    {
        return data<Tp>().is_configured;
    }

    template <typename Tp>
    static bool& is_fixed()
    {
        return data<Tp>().is_fixed;
    }

    template <typename Tp>
    static vector_t<int>& get_events()
    {
        auto& _ret = data<Tp>().events;
        if(!is_fixed<Tp>() && _ret.empty())
            _ret = get_initializer<Tp>()();
        return _ret;
    }

    //----------------------------------------------------------------------------------//

    static void overflow_handler(int evt_set, void* address, long long overflow_vector,
                                 void* context)
    {
        fprintf(stderr, "[papi_common%i]> Overflow at %p! bit=0x%llx \n", evt_set,
                address, overflow_vector);
        consume_parameters(context);
    }

    //----------------------------------------------------------------------------------//

    static void add_event(int evt)
    {
        auto& pevents = private_events();
        auto  fitr    = std::find(pevents.begin(), pevents.end(), evt);
        if(fitr == pevents.end())
            pevents.push_back(evt);
    }

    //----------------------------------------------------------------------------------//

    static bool initialize_papi()
    {
        static thread_local bool _initalized = false;
        static thread_local bool _working    = false;
        if(!_initalized)
        {
            if(settings::debug() || settings::verbose() > 2)
                PRINT_HERE("%s", "Initializing papi");
            papi::init();
            papi::register_thread();
            _initalized = true;
            _working    = papi::working();
            if(!_working)
            {
                std::cerr << "Warning! PAPI failed to initialized!\n";
                std::cerr << "The following PAPI events will not be reported: \n";
                for(const auto& itr : get_events<void>())
                    std::cerr << "    " << papi::get_event_info(itr).short_descr << "\n";
                std::cerr << std::flush;
                // disable all the papi APIs with concrete instantiations
                tim::trait::apply<tim::trait::runtime_enabled>::set<
                    tpls::papi, papi_array_t, papi_common, papi_vector, papi_array8_t,
                    papi_array16_t, papi_array32_t>(false);
            }
        }
        return _working;
    }

    //----------------------------------------------------------------------------------//

    static bool finalize_papi()
    {
        static thread_local bool _finalized = false;
        static thread_local bool _working   = false;
        if(!_finalized)
        {
            papi::unregister_thread();
            _working = papi::working();
        }
        return _working;
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp>
    static get_initializer_t& get_initializer()
    {
        static get_initializer_t _instance = []() {
            papi::init();
            auto events_str = settings::papi_events();

            if(settings::verbose() > 1 || settings::debug())
            {
                printf("[papi_common]> TIMEMORY_PAPI_EVENTS: '%s'...\n",
                       events_str.c_str());
            }

            vector_t<string_t> events_str_list = delimit(events_str);
            vector_t<int>      events_list;

            auto& pevents = private_events();
            for(auto itr = pevents.begin(); itr != pevents.end(); ++itr)
            {
                auto fitr = std::find(events_list.begin(), events_list.end(), *itr);
                if(fitr == events_list.end())
                    events_list.push_back(*itr);
            }

            for(const auto& itr : events_str_list)
            {
                if(itr.length() == 0)
                    continue;

                if(settings::debug())
                    printf("[papi_common]> Getting event code from '%s'...\n",
                           itr.c_str());

                int evt_code = papi::get_event_code(itr);
                if(evt_code == PAPI_NOT_INITED)  // defined as zero
                {
                    std::stringstream ss;
                    ss << "[papi_common] Error creating event with ID: " << itr;
                    if(settings::papi_fail_on_error())
                        throw std::runtime_error(ss.str());
                    else
                        fprintf(stderr, "%s\n", ss.str().c_str());
                }
                else
                {
                    auto fitr =
                        std::find(events_list.begin(), events_list.end(), evt_code);
                    if(fitr == events_list.end())
                    {
                        if(settings::debug() || settings::verbose() > 1)
                            printf("[papi_common] Successfully created event '%s' with "
                                   "code '%i'...\n",
                                   itr.c_str(), evt_code);
                        events_list.push_back(evt_code);
                    }
                    else
                    {
                        if(settings::debug() || settings::verbose() > 1)
                            printf("[papi_common] Event '%s' with code '%i' already "
                                   "exists...\n",
                                   itr.c_str(), evt_code);
                    }
                }
            }

            return events_list;
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp>
    static void initialize()
    {
        if(!is_configured<Tp>() && initialize_papi())
        {
            if(settings::debug() || settings::verbose() > 1)
                PRINT_HERE("%s", "configuring papi");

            auto& _event_set = event_set<Tp>();
            auto& _events    = get_events<Tp>();
            if(_events.size() > 0)
            {
                papi::create_event_set(&_event_set, settings::papi_multiplexing());
                papi::add_events(_event_set, _events.data(), _events.size());
                if(settings::papi_overflow() > 0)
                {
                    for(auto itr : _events)
                        papi::overflow(_event_set, itr, settings::papi_overflow(), 0,
                                       &overflow_handler);
                }
                if(settings::papi_attach())
                    papi::attach(_event_set, process::get_target_id());
                papi::start(_event_set);
                is_configured<Tp>() = papi::working();
            }
        }
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp>
    static void finalize()
    {
        if(!initialize_papi())
            return;
        auto& _event_set = event_set<Tp>();
        auto& _events    = get_events<Tp>();
        if(_events.size() > 0 && _event_set != PAPI_NULL && _event_set >= 0)
        {
            value_type values(_events.size(), 0);
            papi::stop(_event_set, values.data());
            papi::remove_events(_event_set, _events.data(), _events.size());
            papi::destroy_event_set(_event_set);
            _event_set = PAPI_NULL;
            _events.clear();
        }
    }

    //----------------------------------------------------------------------------------//

public:
    template <typename Tp = vector_t<int>>
    papi_common(Tp&& _events = get_events<void>())
    : events(std::forward<Tp>(_events))
    {}

protected:
    event_list events{};

protected:
    static vector_t<int>& private_events()
    {
        static auto _instance = vector_t<int>{};
        return _instance;
    }
};
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
, public papi_common
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
    static void  configure() { papi_common::initialize<common_type>(); }
    static void  thread_init() { papi_common::initialize<common_type>(); }
    static void  thread_finalize()
    {
        papi_common::finalize<common_type>();
        papi_common::finalize_papi();
    }
    static void initialize() { papi_common::initialize<common_type>(); }
    static void finalize() { papi_common::finalize<common_type>(); }

    //----------------------------------------------------------------------------------//

    papi_vector()
    : papi_common(get_events<common_type>())
    {
        value.resize(events.size(), 0);
        accum.resize(events.size(), 0);
    }

    //----------------------------------------------------------------------------------//

    ~papi_vector()                          = default;
    papi_vector(const papi_vector& rhs)     = default;
    papi_vector(papi_vector&& rhs) noexcept = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&&) noexcept = default;

    //----------------------------------------------------------------------------------//

    size_t size() { return events.size(); }

    //----------------------------------------------------------------------------------//

    value_type record()
    {
        value_type read_value(events.size(), 0);
        if(initialize_papi() && event_set<common_type>() != PAPI_NULL)
            papi::read(event_set<common_type>(), read_value.data());
        return read_value;
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp = double>
    vector_t<Tp> get() const
    {
        std::vector<Tp> values;
        auto&           _data = (is_transient) ? accum : value;
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
        if(tracker_type::get_thread_started() == 0)
        {
            papi_common::initialize<common_type>();
            events = get_events<common_type>();
        }

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
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
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

    entry_type get_display(int evt_type) const
    {
        auto val = (is_transient) ? accum[evt_type] : value[evt_type];
        return val;
    }

    //----------------------------------------------------------------------------------//
    // load
    //
    template <typename Archive>
    void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("is_transient", is_transient), cereal::make_nvp("laps", laps),
           cereal::make_nvp("value", value), cereal::make_nvp("accum", accum),
           cereal::make_nvp("events", events));
    }

    //----------------------------------------------------------------------------------//
    // save
    //
    template <typename Archive>
    void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const unsigned int) const
    {
        auto             sz = events.size();
        vector_t<double> _disp(sz, 0.0);
        for(size_type i = 0; i < sz; ++i)
            _disp[i] = get_display(i);
        ar(cereal::make_nvp("is_transient", is_transient), cereal::make_nvp("laps", laps),
           cereal::make_nvp("repr_data", _disp), cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum), cereal::make_nvp("display", _disp),
           cereal::make_nvp("events", events));
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    vector_t<std::string> label_array() const
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
            while((n = itr.find(" ")) != std::string::npos)
                itr.replace(n, 1, "_");

            while((n = itr.find("__")) != std::string::npos)
                itr.replace(n, 2, "_");
        }

        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    vector_t<std::string> description_array() const
    {
        vector_t<std::string> arr(events.size(), "");
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).long_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    vector_t<std::string> display_unit_array() const
    {
        vector_t<std::string> arr(events.size(), "");
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).units;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    vector_t<int64_t> unit_array() const
    {
        vector_t<int64_t> arr(events.size(), 0);
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
};
//
//--------------------------------------------------------------------------------------//
//
template <size_t MaxNumEvents>
struct papi_array
: public base<papi_array<MaxNumEvents>, std::array<long long, MaxNumEvents>>
, public papi_common
{
    using size_type         = size_t;
    using event_list        = std::vector<int>;
    using value_type        = std::array<long long, MaxNumEvents>;
    using entry_type        = typename value_type::value_type;
    using this_type         = papi_array<MaxNumEvents>;
    using base_type         = base<this_type, value_type>;
    using storage_type      = typename base_type::storage_type;
    using get_initializer_t = std::function<event_list()>;
    using common_type       = void;

    static const short precision = 3;
    static const short width     = 8;

    template <typename Tp>
    using array_t = std::array<Tp, MaxNumEvents>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;

    //----------------------------------------------------------------------------------//

    static auto& get_initializer() { return papi_common::get_initializer<common_type>(); }
    static void  configure() { papi_common::initialize<common_type>(); }
    static void  thread_init() { papi_common::initialize<common_type>(); }
    static void  thread_finalize()
    {
        papi_common::finalize<common_type>();
        papi_common::finalize_papi();
    }
    static void initialize() { papi_common::initialize<common_type>(); }
    static void finalize() { papi_common::finalize<common_type>(); }

    //----------------------------------------------------------------------------------//

    papi_array()
    : papi_common(get_events<common_type>())
    {
        apply<void>::set_value(value, 0);
        apply<void>::set_value(accum, 0);
    }

    //----------------------------------------------------------------------------------//

    ~papi_array() {}
    papi_array(const papi_array& rhs)     = default;
    papi_array(papi_array&& rhs) noexcept = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&&) noexcept = default;

    //----------------------------------------------------------------------------------//

    size_t size() { return events.size(); }

    //----------------------------------------------------------------------------------//

    static value_type record()
    {
        value_type read_value;
        apply<void>::set_value(read_value, 0);
        if(initialize_papi() && event_set<common_type>() != PAPI_NULL)
            papi::read(event_set<common_type>(), read_value.data());
        return read_value;
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp = double>
    std::vector<Tp> get() const
    {
        std::vector<Tp> values;
        auto&           _data = (is_transient) ? accum : value;
        for(auto& itr : _data)
            values.push_back(itr);
        values.resize(events.size());
        return values;
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start() { value = record(); }

    //----------------------------------------------------------------------------------//

    void stop()
    {
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
    using papi_common::events;

    friend struct base<this_type, value_type>;
    friend class impl::storage<this_type,
                               trait::implements_storage<this_type, value_type>::value>;

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

    entry_type get_display(int evt_type) const
    {
        auto val = (is_transient) ? accum[evt_type] : value[evt_type];
        return val;
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("is_transient", is_transient), cereal::make_nvp("laps", laps),
           cereal::make_nvp("value", value), cereal::make_nvp("accum", accum),
           cereal::make_nvp("events", events));
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const unsigned int) const
    {
        array_t<double> _disp;
        for(size_type i = 0; i < events.size(); ++i)
            _disp[i] = get_display(i);
        ar(cereal::make_nvp("is_transient", is_transient), cereal::make_nvp("laps", laps),
           cereal::make_nvp("repr_data", _disp), cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum), cereal::make_nvp("display", _disp),
           cereal::make_nvp("events", events));
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    std::vector<std::string> label_array() const
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
            while((n = itr.find(" ")) != std::string::npos)
                itr.replace(n, 1, "_");

            while((n = itr.find("__")) != std::string::npos)
                itr.replace(n, 2, "_");
        }

        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    std::vector<std::string> description_array() const
    {
        std::vector<std::string> arr(events.size());
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).long_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    std::vector<std::string> display_unit_array() const
    {
        std::vector<std::string> arr(events.size());
        for(size_type i = 0; i < events.size(); ++i)
            arr[i] = papi::get_event_info(events[i]).units;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    std::vector<int64_t> unit_array() const
    {
        std::vector<int64_t> arr(events.size());
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
};
//
//--------------------------------------------------------------------------------------//
//
//                      Compile-time constant set of PAPI counters
//
//--------------------------------------------------------------------------------------//
//
template <int... EventTypes>
struct papi_tuple
: public base<papi_tuple<EventTypes...>, std::array<long long, sizeof...(EventTypes)>>
, public papi_common
{
    using size_type    = std::size_t;
    using value_type   = std::array<long long, sizeof...(EventTypes)>;
    using entry_type   = typename value_type::value_type;
    using this_type    = papi_tuple<EventTypes...>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using common_type  = this_type;

    static const size_type num_events = sizeof...(EventTypes);
    template <typename Tp>
    using array_t = std::array<Tp, num_events>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;

public:
    //----------------------------------------------------------------------------------//

    static void configure()
    {
        papi_common::get_initializer<common_type>() = []() {
            return std::vector<int>({ EventTypes... });
        };
        papi_common::get_events<common_type>() = { EventTypes... };
        papi_common::initialize<common_type>();
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
            apply<void>::set_value(values, 0);
            return values;
        }();
        return _instance;
    }

protected:
    using base_type::accum;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    friend struct base<this_type, value_type>;
    friend class impl::storage<this_type,
                               trait::implements_storage<this_type, value_type>::value>;

public:
    //==================================================================================//
    //
    //      construction
    //
    //==================================================================================//

    papi_tuple()
    : papi_common(get_events<common_type>())
    {
        apply<void>::set_value(value, 0);
        apply<void>::set_value(accum, 0);
    }

    ~papi_tuple() {}

    papi_tuple(const papi_tuple& rhs) = default;
    this_type& operator=(const this_type& rhs) = default;
    papi_tuple(papi_tuple&& rhs) noexcept      = default;
    this_type& operator=(this_type&&) noexcept = default;

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        if(!papi_common::is_configured<common_type>())
        {
            papi_common::initialize<common_type>();
            events = get_events<common_type>();
        }
        value = record();
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void stop()
    {
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
        ar(cereal::make_nvp("is_transient", is_transient), cereal::make_nvp("laps", laps),
           cereal::make_nvp("repr_data", _disp), cereal::make_nvp("value", _value),
           cereal::make_nvp("accum", _accum), cereal::make_nvp("display", _disp));
        // ar(cereal::make_nvp("units", unit_array()),
        //   cereal::make_nvp("display_units", display_unit_array()));
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
            auto     _evt_type  = std::vector<int>({ EventTypes... }).at(idx);
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
        auto&                      _data = (is_transient) ? accum : value;
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
            arr[i] = papi::get_event_info(get_events<common_type>()[i]).short_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    static array_t<std::string> description_array()
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = papi::get_event_info(get_events<common_type>()[i]).long_descr;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    static array_t<std::string> display_unit_array()
    {
        array_t<std::string> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = papi::get_event_info(get_events<common_type>()[i]).units;
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
//
template <int... EventTypes>
struct papi_rate_tuple
: public base<papi_rate_tuple<EventTypes...>,
              std::pair<papi_tuple<EventTypes...>, wall_clock>>
{
    using size_type                   = std::size_t;
    static const size_type num_events = sizeof...(EventTypes);

    using tuple_type   = papi_tuple<EventTypes...>;
    using value_type   = std::pair<tuple_type, wall_clock>;
    using this_type    = papi_rate_tuple<EventTypes...>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using common_type  = this_type;

    template <typename Tp>
    using array_t = std::array<Tp, num_events>;

    friend struct operation::record<common_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;

public:
    static void configure() { tuple_type::configure(); }
    static void thread_init() { tuple_type::thread_init(); }
    static void thread_finalize() { tuple_type::thread_finalize(); }
    static void initialize() { tuple_type::initialize(); }
    static void finalize() { tuple_type::finalize(); }

    static std::string label() { return tuple_type::label() + "_rate"; }
    static std::string description()
    {
        return "Divides the given set of HW counters by the elapsed time of the "
               "measurement";
    }

public:
    void start()
    {
        value.first.start();
        value.second.start();
    }
    void stop()
    {
        value.first.stop();
        value.second.stop();
    }

    this_type& operator+=(const this_type& rhs)
    {
        value.first += rhs.value.first;
        value.second += rhs.value.second;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        value.first -= rhs.value.first;
        value.second -= rhs.value.second;
        return *this;
    }

    auto get() const
    {
        auto _val = value.first.get();
        for(auto& itr : _val)
            itr /= value.second.get();
        return _val;
    }

    static auto label_array()
    {
        auto arr = tuple_type::label_array();
        for(auto& itr : arr)
            itr += " per " + wall_clock::get_display_unit();
        return arr;
    }

    static auto description_array()
    {
        auto arr = tuple_type::description_array();
        for(auto& itr : arr)
            itr += " Rate";
        return arr;
    }

    static auto display_unit_array()
    {
        auto arr = tuple_type::display_unit_array();
        for(auto& itr : arr)
        {
            if(itr.empty())
                itr = "1";
            itr += "/" + wall_clock::display_unit();
        }
        return arr;
    }

    static auto unit_array()
    {
        std::array<double, num_events> arr;
        auto                           _units = tuple_type::unit_array();
        for(size_t i = 0; i < _units.size(); ++i)
            arr.at(i) = _units.at(i);
        for(auto& itr : _units)
            itr /= wall_clock::unit();
        return arr;
    }

protected:
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    friend struct base<this_type, value_type>;
    friend class impl::storage<this_type,
                               trait::implements_storage<this_type, value_type>::value>;
};
}  // namespace component
}  // namespace tim
//
//======================================================================================//
