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
//              cpu_roofline<PAPI_SP_OPS>
//
//  double precision:
//              cpu_roofline<PAPI_DP_OPS>
//
//  generic:
//              cpu_roofline<PAPI_FP_OPS>
//              cpu_roofline<PAPI_SP_OPS, PAPI_DP_OPS>
//
// NOTE: in order to do a roofline, the peak must be calculated with ERT
//      (eventually will be integrated)
//
template <typename _Tp, int... EventTypes>
struct cpu_roofline
: public base<cpu_roofline<_Tp, EventTypes...>,
              std::pair<std::array<long long, sizeof...(EventTypes) + 1>, double>,
              policy::thread_init, policy::thread_finalize, policy::global_finalize,
              policy::serialization>
{
    friend struct policy::wrapper<policy::thread_init, policy::thread_finalize,
                                  policy::global_finalize, policy::serialization>;

    using size_type  = std::size_t;
    using array_type = std::array<long long, sizeof...(EventTypes) + 1>;
    using data_type  = long long*;
    using value_type = std::pair<array_type, double>;
    using this_type  = cpu_roofline<_Tp, EventTypes...>;
    using base_type =
        base<this_type, value_type, policy::thread_init, policy::thread_finalize,
             policy::global_finalize, policy::serialization>;

    using papi_op_type                = papi_tuple<EventTypes...>;
    using papi_ai_type                = papi_tuple<PAPI_LST_INS>;
    using clock_type                  = real_clock;
    using ratio_t                     = typename clock_type::ratio_t;
    using operation_counter_t         = tim::ert::cpu::operation_counter<_Tp, clock_type>;
    using operation_function_t        = std::function<operation_counter_t*()>;
    using operation_counter_ptr_t     = std::shared_ptr<operation_counter_t>;
    using operation_uint64_function_t = std::function<uint64_t()>;

    using iterator       = typename array_type::iterator;
    using const_iterator = typename array_type::const_iterator;

    // total size of data
    static const size_type num_events = sizeof...(EventTypes) + 1;
    // array size
    static const size_type num_op_events = sizeof...(EventTypes);
    static const size_type num_ai_events = 1;
    // array offsets
    static const size_type num_op_offset = 0;
    static const size_type num_ai_offset = sizeof...(EventTypes);

    static const short                   precision = 3;
    static const short                   width     = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    // collection mode, AI (arithmetic intensity) is the load/store: PAPI_LST_INS
    enum class MODE
    {
        OP,
        AI
    };

    static int& op_event_set()
    {
        static thread_local int _instance = PAPI_NULL;
        return _instance;
    }
    static int& ai_event_set()
    {
        static thread_local int _instance = PAPI_NULL;
        return _instance;
    }

    static MODE& event_mode()
    {
        static auto aslower = [](std::string str) {
            for(auto& itr : str)
                itr = tolower(itr);
            return str;
        };
        static std::string _env = get_env<std::string>("TIMEMORY_ROOFLINE_MODE", "op");
        static MODE        _instance = (aslower(_env) == "op")
                                    ? MODE::OP
                                    : ((aslower(_env) == "ai") ? MODE::AI : MODE::OP);
        return _instance;
    }

    static operation_uint64_function_t& get_finalize_threads_function()
    {
        static operation_uint64_function_t _instance = []() -> uint64_t {
            return get_env<uint64_t>("TIMEMORY_ROOFLINE_NUM_THREADS",
                                     std::thread::hardware_concurrency());
        };
        return _instance;
    }

    static operation_uint64_function_t& get_finalize_align_function()
    {
        static operation_uint64_function_t _instance = []() -> uint64_t {
            return std::max<uint64_t>(32, sizeof(_Tp));
        };
        return _instance;
    }

    static operation_function_t& get_finalize_function()
    {
        static operation_function_t _instance = []() {
            // vectorization number of ops
            static constexpr const int SIZE_BITS = sizeof(_Tp) * 8;
            static_assert(SIZE_BITS > 0, "Calculated bits size is not greater than zero");
            static constexpr const int VEC = TIMEMORY_VEC / SIZE_BITS;
            static_assert(VEC > 0, "Calculated vector size is zero");
            // functions
            auto store_func = [](_Tp& a, const _Tp& b) { a = b; };
            auto add_func   = [](_Tp& a, const _Tp& b, const _Tp& c) { a = b + c; };
            auto fma_func   = [](_Tp& a, const _Tp& b, const _Tp& c) { a = a * b + c; };
            // configuration sizes
            auto lm_size    = tim::ert::cache_size::get_max();
            auto num_thread = get_finalize_threads_function()();
            auto align_size = get_finalize_align_function()();
            // execution parameters
            ert::exec_params params(16, 8 * lm_size, num_thread);
            // operation counter instance
            auto op_counter = new operation_counter_t(params, align_size);
            // set bytes per element
            op_counter->bytes_per_element = sizeof(_Tp);
            // set number of memory accesses per element from two functions
            op_counter->memory_accesses_per_element = 2;
            // run the kernels (<4> is ideal for avx, <8> is ideal for KNL)
            tim::ert::cpu_ops_main<1>(*op_counter, add_func, store_func);
            tim::ert::cpu_ops_main<VEC, 2 * VEC>(*op_counter, fma_func, store_func);
            // return the operation count data
            return op_counter;
        };
        return _instance;
    }

    static operation_counter_ptr_t& get_operation_counter()
    {
        static operation_counter_ptr_t _instance;
        return _instance;
    }

    static void invoke_thread_init()
    {
        // start PAPI counters
        int op_events[] = { EventTypes... };
        int ai_events[] = { PAPI_LST_INS };
        tim::papi::create_event_set(&op_event_set());
        tim::papi::create_event_set(&ai_event_set());
        tim::papi::add_events(op_event_set(), op_events, num_op_events);
        tim::papi::add_events(ai_event_set(), ai_events, num_ai_events);
        switch(event_mode())
        {
            case MODE::OP: tim::papi::start(op_event_set(), true); break;
            case MODE::AI: tim::papi::start(ai_event_set(), true); break;
        }
    }

    static void invoke_thread_finalize()
    {
        // stop PAPI counters
        std::array<long long, num_op_events> op_values;
        std::array<long long, num_ai_events> ai_values;
        int                                  op_events[] = { EventTypes... };
        int                                  ai_events[] = { PAPI_LST_INS };
        switch(event_mode())
        {
            case MODE::OP: tim::papi::stop(op_event_set(), op_values.data()); break;
            case MODE::AI: tim::papi::stop(ai_event_set(), ai_values.data()); break;
        }
        tim::papi::remove_events(op_event_set(), op_events, num_op_events);
        tim::papi::remove_events(ai_event_set(), ai_events, num_ai_events);
        tim::papi::destroy_event_set(op_event_set());
        tim::papi::destroy_event_set(ai_event_set());
    }

    static void invoke_global_finalize()
    {
        // run roofline peak generation
        auto  op_counter_func = get_finalize_function();
        auto* op_counter      = op_counter_func();
        get_operation_counter().reset(op_counter);
        if(op_counter && (settings::verbose() > 0 || settings::debug()))
            std::cout << *op_counter << std::endl;
    }

    template <typename _Archive>
    static void invoke_serialize(_Archive& ar, const unsigned int /*version*/)
    {
        auto& op_counter = get_operation_counter();
        ar(serializer::make_nvp("roofline", op_counter));
    }

    static std::string get_mode_string()
    {
        return (event_mode() == MODE::OP) ? "op" : "ai";
    }

    static int64_t     unit() { return 1; }
    static std::string label() { return "cpu_roofline_" + get_mode_string(); }
    static std::string descript() { return "cpu roofline " + get_mode_string(); }
    static std::string display_unit()
    {
        std::stringstream ss;
        ss << "(";
        std::vector<std::string> labels;
        if(event_mode() == MODE::OP)
        {
            auto op_labels = papi_op_type::label_array();
            for(auto itr : op_labels)
                labels.push_back(itr);
        } else
        {
            auto ai_labels = papi_ai_type::label_array();
            for(auto itr : ai_labels)
                labels.push_back(itr);
        }

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

    static value_type record()
    {
        std::array<long long, num_events> read_values;
        apply<void>::set_value(read_values, 0);
        switch(event_mode())
        {
            case MODE::OP: tim::papi::read(op_event_set(), read_values.data()); break;
            case MODE::AI:
                tim::papi::read(ai_event_set(), read_values.data() + num_op_events);
                break;
        }
        auto delta_duration =
            clock_type::record() / static_cast<double>(ratio_t::den) * tim::units::sec;
        return value_type(read_values, delta_duration);
    }

    iterator begin()
    {
        auto& obj = (accum.second > 0) ? accum : value;
        return (event_mode() == MODE::OP) ? obj.first.begin()
                                          : (obj.first.begin() + num_op_events);
    }

    const_iterator begin() const
    {
        const auto& obj = (accum.second > 0) ? accum : value;
        return (event_mode() == MODE::OP) ? obj.first.begin()
                                          : (obj.first.begin() + num_op_events);
    }

    iterator end()
    {
        auto& obj = (accum.second > 0) ? accum : value;
        return (event_mode() == MODE::OP) ? (obj.first.end() - 1) : obj.first.end();
    }

    const_iterator end() const
    {
        const auto& obj = (accum.second > 0) ? accum : value;
        return (event_mode() == MODE::OP) ? (obj.first.end() - 1) : obj.first.end();
    }

    double get_elapsed(const int64_t& _unit = clock_type::get_unit()) const
    {
        auto& obj = (accum.second > 0) ? accum : value;
        return static_cast<double>(obj.second) *
               (static_cast<double>(_unit) / tim::units::sec);
    }

    double get_counted() const
    {
        double _sum = 0.0;
        for(auto itr = begin(); itr != end(); ++itr)
            _sum += static_cast<double>(*itr);
        return _sum;
    }

    double get() const { return get_counted() / get_elapsed(); }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp = record();
        for(size_type i = 0; i < num_events; ++i)
            accum.first[i] += (tmp.first[i] - value.first[i]);
        accum.second += (tmp.second - value.second);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
        {
            accum.first[i] += rhs.accum.first[i];
            value.first[i] += rhs.value.first[i];
        }
        accum.second += rhs.accum.second;
        value.second += rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
        {
            accum.first[i] -= rhs.accum.first[i];
            value.first[i] -= rhs.value.first[i];
        }
        accum.second -= rhs.accum.second;
        value.second -= rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    using base_type::accum;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

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
};

//--------------------------------------------------------------------------------------//
// Shorthand aliases for common roofline types
//
using cpu_roofline_sp_flops = cpu_roofline<float, PAPI_SP_OPS>;
using cpu_roofline_dp_flops = cpu_roofline<double, PAPI_DP_OPS>;

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
