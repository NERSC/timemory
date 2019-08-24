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
#include "timemory/components/cupti_activity.hpp"
#include "timemory/components/cupti_counters.hpp"
#include "timemory/components/timing.hpp"
#include "timemory/components/types.hpp"
#include "timemory/details/cupti.hpp"
#include "timemory/details/settings.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/ert/kernels.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

#include <array>
#include <memory>
#include <numeric>
#include <utility>

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
//              gpu_roofline<float>
//
//  double precision:
//              gpu_roofline<double>
//
//
template <typename _Tp>
struct gpu_roofline
: public base<gpu_roofline<_Tp>,
              std::tuple<typename cupti_activity::value_type,
                         typename cupti_counters::value_type>,
              policy::global_init, policy::global_finalize, policy::thread_init,
              policy::thread_finalize, policy::global_finalize, policy::serialization>
{
    friend struct policy::wrapper<policy::global_init, policy::global_finalize,
                                  policy::thread_init, policy::thread_finalize,
                                  policy::global_finalize, policy::serialization>;

    using value_type = std::tuple<typename cupti_activity::value_type,
                                  typename cupti_counters::value_type>;
    using this_type  = gpu_roofline<_Tp>;
    using base_type =
        base<this_type, value_type, policy::global_init, policy::global_finalize,
             policy::thread_init, policy::thread_finalize, policy::global_finalize,
             policy::serialization>;

    using size_type = std::size_t;

    using counters_type = cupti_counters;
    using activity_type = cupti_activity;
    using clock_type    = real_clock;
    using device_t      = device::gpu;
    using result_type   = std::map<string_t, double>;

    using operation_counter_t  = tim::ert::operation_counter<device_t, _Tp, clock_type>;
    using operation_function_t = std::function<operation_counter_t*()>;
    using operation_counter_ptr_t     = std::shared_ptr<operation_counter_t>;
    using operation_uint64_function_t = std::function<uint64_t()>;

    static const short                   precision = 3;
    static const short                   width     = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    //----------------------------------------------------------------------------------//
    // collection mode, COUNTERS is the HW counting, ACTIVITY in the runtime measurements
    enum class MODE
    {
        COUNTERS,
        ACTIVITY
    };

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
            std::string _std = aslc(get_env<std::string>("TIMEMORY_ROOFLINE_MODE", "hw"));
            // check the specific variable (to override)
            std::string _env =
                aslc(get_env<std::string>("TIMEMORY_GPU_ROOFLINE_MODE", _std));
            return (_env == "op" || _env == "hw" || _env == "counters")
                       ? MODE::COUNTERS
                       : ((_env == "ai" || _env == "ac" || _env == "activity")
                              ? MODE::ACTIVITY
                              : MODE::COUNTERS);
        };

        static MODE _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static std::string get_mode_string()
    {
        return (event_mode() == MODE::COUNTERS) ? "counters" : "activity";
    }

    //----------------------------------------------------------------------------------//

    static operation_uint64_function_t& get_num_threads_finalizer()
    {
        static operation_uint64_function_t _instance = []() -> uint64_t {
            return get_env<uint64_t>("TIMEMORY_ROOFLINE_NUM_THREADS", 1);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static operation_uint64_function_t& get_num_streams_finalizer()
    {
        static operation_uint64_function_t _instance = []() -> uint64_t {
            return get_env<uint64_t>("TIMEMORY_ROOFLINE_NUM_STREAMS", 1);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static operation_uint64_function_t& get_grid_size_finalizer()
    {
        static operation_uint64_function_t _instance = []() -> uint64_t {
            return get_env<uint64_t>("TIMEMORY_ROOFLINE_GRID_SIZE", 0);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static operation_uint64_function_t& get_block_size_finalizer()
    {
        static operation_uint64_function_t _instance = []() -> uint64_t {
            return get_env<uint64_t>("TIMEMORY_ROOFLINE_BLOCK_SIZE", 32);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static operation_uint64_function_t& get_alignment_finalizer()
    {
        static operation_uint64_function_t _instance = []() -> uint64_t {
            return std::max<uint64_t>(32, sizeof(_Tp));
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static operation_function_t& get_finalizer()
    {
        static operation_function_t _instance = []() {
            // vectorization number of ops
            static constexpr const int SIZE_BITS = sizeof(_Tp) * 8;
            static_assert(SIZE_BITS > 0, "Calculated bits size is not greater than zero");
            static constexpr const int VEC = TIMEMORY_VEC / SIZE_BITS;
            static_assert(VEC > 0, "Calculated vector size is zero");
            // functions
            auto store_func = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b) { a = b; };
            auto add_func   = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
                a = b + c;
            };
            auto fma_func = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
                a = a * b + c;
            };
            // configuration sizes
            auto ws_size    = 1000;
            auto lm_size    = 100 * units::megabyte * sizeof(_Tp);
            auto num_thread = get_num_threads_finalizer()();
            auto num_stream = get_num_streams_finalizer()();
            auto grid_size  = get_grid_size_finalizer()();
            auto block_size = get_block_size_finalizer()();
            auto align_size = get_alignment_finalizer()();
            // execution parameters
            ert::exec_params params(ws_size, lm_size, num_thread, num_stream, grid_size,
                                    block_size);
            // operation counter instance
            auto op_counter = new operation_counter_t(params, align_size);
            // set bytes per element
            op_counter->bytes_per_element = sizeof(_Tp);
            // set number of memory accesses per element from two functions
            op_counter->memory_accesses_per_element = 2;
            // sync device
            cuda::device_sync();
            // run the kernels
            tim::ert::ops_main<1>(*op_counter, add_func, store_func);
            // sync device
            cuda::device_sync();
            tim::ert::ops_main<VEC>(*op_counter, fma_func, store_func);
            // sync device
            cuda::device_sync();
            // return the operation count data
            return op_counter;
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static operation_counter_ptr_t& get_operation_counter()
    {
        static operation_counter_ptr_t _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void invoke_global_init()
    {
        static std::atomic<short> _once;
        if(_once++ > 0)
            return;

        using tuple_type = counters_type::tuple_type;
        using strvec_t   = std::vector<string_t>;

        if(event_mode() == MODE::ACTIVITY)
        {
            activity_type::invoke_global_init();
        } else
        {
            strvec_t events  = { "active_warps", "global_load", "global_store" };
            strvec_t metrics = { "ldst_executed", "ldst_issued" };

            if(std::is_same<float, _Tp>::value)
            {
                for(string_t itr : { "flop_count_sp", "flop_count_sp_add",
                                     "flop_count_sp_mul", "flop_count_sp_fma" })
                    metrics.push_back(itr);
            }

            if(std::is_same<double, _Tp>::value)
            {
                for(string_t itr : { "flop_count_dp", "flop_count_dp_add",
                                     "flop_count_dp_mul", "flop_count_dp_fma" })
                    metrics.push_back(itr);
            }

            counters_type::get_initializer() = [=]() {
                return tuple_type(counters_type::get_device_initializer()(), events,
                                  metrics);
            };

            counters_type::invoke_global_init();
        }
    }

    //----------------------------------------------------------------------------------//

    static void invoke_global_finalize()
    {
        if(event_mode() == MODE::ACTIVITY)
        {
            activity_type::invoke_global_finalize();
        } else
        {
            counters_type::invoke_global_finalize();
        }

        // run roofline peak generation
        auto  op_counter_func = get_finalizer();
        auto* op_counter      = op_counter_func();
        get_operation_counter().reset(op_counter);
        if(op_counter && (settings::verbose() > 0 || settings::debug()))
            std::cout << *op_counter << std::endl;
    }

    //----------------------------------------------------------------------------------//

    static void invoke_thread_init() {}
    static void invoke_thread_finalize() {}

    //----------------------------------------------------------------------------------//

    template <typename _Archive>
    static void invoke_serialize(_Archive& ar, const unsigned int)
    {
        auto& op_counter = get_operation_counter();
        ar(serializer::make_nvp("roofline", op_counter));
    }

    //----------------------------------------------------------------------------------//

    static int64_t     unit() { return 1; }
    static std::string label() { return "gpu_roofline_" + get_mode_string(); }
    static std::string descript() { return "gpu roofline " + get_mode_string(); }
    static std::string display_unit()
    {
        std::stringstream ss;
        ss << "(";
        std::vector<std::string> labels;
        switch(event_mode())
        {
            case MODE::ACTIVITY: labels.push_back(activity_type::get_label()); break;
            case MODE::COUNTERS:
            {
                auto counters_labels = counters_type::label_array();
                for(auto itr : counters_labels)
                    labels.push_back(itr);
                break;
            }
            default: break;
        }

        for(size_type i = 0; i < labels.size(); ++i)
        {
            ss << labels[i];
            if(i + 1 < labels.size())
                ss << " + ";
        }
        ss << ")";
        return ss.str();
    }

    //----------------------------------------------------------------------------------//

    static value_type record()
    {
        value_type tmp;
        switch(event_mode())
        {
            case MODE::ACTIVITY: std::get<0>(tmp) = activity_type::record(); break;
            case MODE::COUNTERS: std::get<1>(tmp) = counters_type::record(); break;
            default: break;
        }
        return tmp;
    }

    //----------------------------------------------------------------------------------//

public:

    //----------------------------------------------------------------------------------//

    gpu_roofline()
    {
        if(event_mode() == MODE::COUNTERS)
        {
            const auto& _labels = counters_type::get_labels();
            auto&       _value  = std::get<1>(value);
            auto&       _accum  = std::get<1>(accum);
            _value.resize(_labels.size());
            _accum.resize(_labels.size());
            for(size_type i = 0; i < _labels.size(); ++i)
            {
                _value[i].name = _labels[i];
                _accum[i].name = _labels[i];
            }
        }
    }

    //----------------------------------------------------------------------------------//

    result_type get_elapsed(const int64_t& _unit = clock_type::get_unit()) const
    {
        if(event_mode() == MODE::COUNTERS)
            return result_type{};

        auto&       _obj = (is_transient) ? accum : value;
        auto&       obj  = std::get<0>(_obj);
        result_type _ret;
        _ret["runtime"] =
            static_cast<double>(obj) * (static_cast<double>(_unit) / tim::units::sec);
        return _ret;
    }

    //----------------------------------------------------------------------------------//

    result_type get_counted() const
    {
        if(event_mode() == MODE::ACTIVITY)
            return result_type{};

        const auto& __data  = (is_transient) ? accum : value;
        const auto& _data   = std::get<1>(__data);
        const auto& _labels = counters_type::label_array();
        result_type _ret;
        uint64_t    _ntot = std::min<uint64_t>(_labels.size(), _data.size());

        if(_labels.size() != _data.size())
        {
            fprintf(stderr,
                    "[gpu_roofline]> Warning! Number of labels differ from number of "
                    "data values!\n");
        }

        for(uint64_t i = 0; i < _ntot; ++i)
        {
            if(settings::debug())
            {
                std::cout << "    " << std::setw(28) << _labels[i] << " : "
                          << std::setw(12) << std::setprecision(6) << _data[i]
                          << std::endl;
            }
            _ret[_labels[i]] = cupti::get<double>(_data[i].data);
        }
        return _ret;
    }

    //----------------------------------------------------------------------------------//

    result_type get() const
    {
        switch(event_mode())
        {
            case MODE::ACTIVITY: return get_elapsed(); break;
            case MODE::COUNTERS: return get_counted(); break;
            default: break;
        }
        return result_type{};
    }

    //----------------------------------------------------------------------------------//

    void start()
    {
        set_started();
        switch(event_mode())
        {
            case MODE::ACTIVITY: std::get<0>(value) = activity_type::record(); break;
            case MODE::COUNTERS: std::get<1>(value) = counters_type::record(); break;
            default: break;
        }
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        value_type tmp;
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                std::get<0>(tmp) = activity_type::record();
                std::get<0>(accum) += (std::get<0>(tmp) - std::get<0>(value));
                std::get<0>(value) = std::move(std::get<0>(tmp));
                break;
            }
            case MODE::COUNTERS:
            {
                std::get<1>(tmp) = counters_type::record();
                if(std::get<1>(accum).size() == 0)
                    std::get<1>(accum) = std::get<1>(tmp);
                else
                {
                    for(uint64_t i = 0; i < std::get<1>(tmp).size(); ++i)
                        std::get<1>(accum)[i] +=
                            (std::get<1>(tmp)[i] - std::get<1>(value)[i]);
                }
                std::get<1>(value) = std::move(std::get<1>(tmp));
                break;
            }
        }
        set_stopped();
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                std::get<0>(accum) += std::get<0>(rhs.accum);
                std::get<0>(value) += std::get<0>(rhs.value);
                break;
            }
            case MODE::COUNTERS:
            {
                auto& _accum = std::get<1>(accum);
                auto& _value = std::get<1>(value);
                if(_accum.empty())
                    _accum = std::get<1>(rhs.accum);
                else
                {
                    for(uint64_t i = 0; i < std::get<1>(rhs.accum).size(); ++i)
                        std::get<1>(accum)[i] += std::get<1>(rhs.accum)[i];
                }
                if(_value.empty())
                    _value = std::get<1>(rhs.value);
                else
                {
                    for(uint64_t i = 0; i < std::get<1>(rhs.value).size(); ++i)
                        std::get<1>(value)[i] += std::get<1>(rhs.value)[i];
                }
                break;
            }
        }

        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                std::get<0>(accum) -= std::get<0>(rhs.accum);
                std::get<0>(value) -= std::get<0>(rhs.value);
                break;
            }
            case MODE::COUNTERS:
            {
                for(uint64_t i = 0; i < std::get<1>(accum).size(); ++i)
                    std::get<1>(accum)[i] -= std::get<1>(rhs.accum)[i];
                for(uint64_t i = 0; i < std::get<1>(value).size(); ++i)
                    std::get<1>(value)[i] -= std::get<1>(rhs.value)[i];
                break;
            }
        }

        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    //----------------------------------------------------------------------------------//

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
    string_t get_display() const
    {
        if(event_mode() == MODE::COUNTERS)
        {
            auto _get_display = [&](std::ostream& os, const cupti::result& obj) {
                auto _label = obj.name;
                auto _prec  = base_type::get_precision();
                auto _width = base_type::get_width();
                auto _flags = base_type::get_format_flags();

                std::stringstream ss, ssv, ssi;
                ssv.setf(_flags);
                ssv << std::setw(_width) << std::setprecision(_prec);
                cupti::print(ssv, obj.data);
                if(!_label.empty())
                    ssi << " " << _label;
                ss << ssv.str() << ssi.str();
                os << ss.str();
            };

            const auto&       _data = (is_transient) ? accum : value;
            std::stringstream ss;
            for(size_type i = 0; i < std::get<1>(_data).size(); ++i)
            {
                _get_display(ss, std::get<1>(_data)[i]);
                if(i + 1 < std::get<1>(_data).size())
                    ss << ", ";
            }
            return ss.str();
        }

        auto              _val = (is_transient) ? accum : value;
        auto              val  = std::get<0>(_val);
        std::stringstream ss;
        ss << static_cast<float>(val / static_cast<float>(activity_type::ratio_t::den) *
                                 base_type::get_unit());
        return ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        os << as_string(obj.get_display());
        return os;
    }

private:
    static string_t as_string(const string_t& _value)
    {
        auto _label = this_type::get_label();
        auto _disp  = this_type::get_display_unit();
        auto _prec  = this_type::get_precision();
        auto _width = this_type::get_width();
        auto _flags = this_type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;
        if(!_disp.empty() && !trait::custom_unit_printing<this_type>::value)
            ss_extra << " " << _disp;
        if(!_label.empty() && !trait::custom_label_printing<this_type>::value)
            ss_extra << " " << _label;

        std::stringstream ss;
        ss << ss_value.str() << ss_extra.str();
        return ss.str();
    }

private:
    union cupti_data
    {
        cupti_activity* activity = nullptr;
        cupti_counters* counters;

        cupti_data()
        {
            switch(event_mode())
            {
                case MODE::ACTIVITY: activity = new cupti_activity; break;
                case MODE::COUNTERS: counters = new cupti_counters; break;
                default: break;
            }
        }

        ~cupti_data()
        {
            switch(event_mode())
            {
                case MODE::ACTIVITY: delete activity; break;
                case MODE::COUNTERS: delete counters; break;
                default: break;
            }
        }
    };
    // cupti_data m_data;
};

//--------------------------------------------------------------------------------------//
// Shorthand aliases for common roofline types
//
using gpu_roofline_sp_flops = gpu_roofline<float>;
using gpu_roofline_dp_flops = gpu_roofline<double>;

//--------------------------------------------------------------------------------------//
}  // namespace component

namespace trait
{
template <>
struct requires_json<component::gpu_roofline_sp_flops> : std::true_type
{};

template <>
struct requires_json<component::gpu_roofline_dp_flops> : std::true_type
{};

#if !defined(TIMEMORY_USE_CUPTI)
template <>
struct is_available<component::gpu_roofline_sp_flops> : std::false_type
{};

template <>
struct is_available<component::gpu_roofline_dp_flops> : std::false_type
{};
#endif

}  // namespace trait

}  // namespace tim
