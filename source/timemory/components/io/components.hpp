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
 * \file timemory/components/io/components.hpp
 * \brief Implementation of the io component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"

#include "timemory/components/io/backends.hpp"
#include "timemory/components/io/types.hpp"
#include "timemory/components/timing/backends.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
/// \struct tim::component::read_char
/// \brief I/O counter for chars read. The number of bytes which this task has caused to
/// be read from storage. This is simply the sum of bytes which this process passed to
/// read() and pread(). It includes things like tty IO and it is unaffected by whether or
/// not actual physical disk IO was required (the read might have been satisfied from
/// pagecache)
struct read_char : public base<read_char, std::pair<int64_t, int64_t>>
{
    using this_type         = read_char;
    using value_type        = std::pair<int64_t, int64_t>;
    using base_type         = base<this_type, value_type>;
    using result_type       = std::pair<double, double>;
    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    static std::string label() { return "read_char"; }
    static std::string description()
    {
        return "Number of bytes which this task has caused to be read from storage. Sum "
               "of bytes which this process passed to read() and pread(). Not disk IO.";
    }

    static std::pair<double, double> unit()
    {
        return std::pair<double, double>{
            units::megabyte, static_cast<double>(units::megabyte) / units::sec
        };
    }

    static std::vector<std::string> display_unit_array()
    {
        return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                         std::get<1>(get_display_unit()) };
    }

    static std::vector<std::string> label_array()
    {
        return std::vector<std::string>{ label(), "read_rate" };
    }

    static display_unit_type display_unit()
    {
        return display_unit_type{ "MB", "MB/sec" };
    }

    static std::pair<double, double> unit_array() { return unit(); }

    static std::vector<std::string> description_array()
    {
        return std::vector<std::string>{ "Number of char read", "Rate of char read" };
    }

    static auto get_timestamp() { return tim::get_clock_real_now<int64_t, std::nano>(); }

    static value_type record() { return value_type(get_char_read(), get_timestamp()); }

    static auto get_timing_unit()
    {
        static auto _value = units::sec;
        if(settings::timing_units().length() > 0)
            _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
        return _value;
    }

    TIMEMORY_NODISCARD std::string get_display() const
    {
        std::stringstream ss;
        std::stringstream ssv;
        std::stringstream ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << ", " << ssr.str();
        ss << " rchar";
        return ss.str();
    }

    TIMEMORY_NODISCARD result_type get() const
    {
        auto val = base_type::load();

        double data  = std::get<0>(val);
        double delta = std::get<1>(val);

        delta /= static_cast<double>(std::nano::den);
        delta *= get_timing_unit();

        double rate = 0.0;
        if(delta != 0.0)
            rate = data / delta;

        if(laps > 0)
            rate *= laps;

        data /= std::get<0>(get_unit());
        rate /= std::get<0>(get_unit());

        if(!std::isfinite(rate))
            rate = 0.0;

        return result_type(data, rate);
    }

    void start() { value = record(); }

    void stop()
    {
        using namespace tim::component::operators;
        auto diff         = (record() - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += (value = diff);
    }

    static unit_type get_unit()
    {
        static auto  _instance = this_type::unit();
        static auto& _mem      = std::get<0>(_instance);
        static auto& _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _timing_val =
                std::get<1>(units::get_timing_unit(settings::timing_units()));
            _rate = _mem / (_timing_val);
        }

        static const auto factor = static_cast<double>(std::nano::den);
        unit_type         _tmp   = _instance;
        std::get<1>(_tmp) *= factor;

        return _tmp;
    }

    static display_unit_type get_display_unit()
    {
        static display_unit_type _instance = this_type::display_unit();
        static auto&             _mem      = std::get<0>(_instance);
        static auto&             _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
            _rate      = mpl::apply<std::string>::join('/', _mem, _tval);
        }
        else if(settings::memory_units().length() > 0)
        {
            _rate = mpl::apply<std::string>::join('/', _mem, "sec");
        }

        return _instance;
    }

    /// sample a measurement
    void sample()
    {
        std::get<0>(accum) = std::get<0>(value) =
            std::max<int64_t>(std::get<0>(value), get_char_read());
    }

    /// sample a measurement from cached data
    void sample(const cache_type& _cache)
    {
        std::get<0>(accum) = std::get<0>(value) =
            std::max<int64_t>(std::get<0>(value), _cache.get_char_read());
    }

    /// read the value from cached data
    static value_type record(const cache_type& _cache)
    {
        return value_type{ _cache.get_char_read(), get_timestamp() };
    }

    /// start a measurement using the cached data
    void start(const cache_type& _cache) { value = record(_cache); }

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache)
    {
        using namespace tim::component::operators;
        auto diff         = (record(_cache) - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += (value = diff);
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::written_char
/// \brief I/O counter for chars written. The number of bytes which this task has caused,
/// or shall cause to be written to disk. Similar caveats apply here as with \ref
/// tim::component::read_char (rchar).
struct written_char : public base<written_char, std::array<int64_t, 2>>
{
    using this_type         = written_char;
    using value_type        = std::array<int64_t, 2>;
    using base_type         = base<this_type, value_type>;
    using result_type       = std::array<double, 2>;
    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    static std::string label() { return "written_char"; }
    static std::string description()
    {
        return "Number of bytes which this task has caused, or shall cause to be written "
               "to disk. Similar caveats to read_char.";
    }

    static result_type unit()
    {
        return result_type{ { units::megabyte,
                              static_cast<double>(units::megabyte) / units::sec } };
    }

    static std::vector<std::string> display_unit_array()
    {
        return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                         std::get<1>(get_display_unit()) };
    }

    static std::vector<std::string> label_array()
    {
        return std::vector<std::string>{ label(), "written_rate" };
    }

    static display_unit_type display_unit()
    {
        return display_unit_type{ { "MB", "MB/sec" } };
    }

    static std::array<double, 2> unit_array() { return unit(); }

    static std::vector<std::string> description_array()
    {
        return std::vector<std::string>{ "Number of char written",
                                         "Rate of char written" };
    }

    static auto get_timestamp() { return tim::get_clock_real_now<int64_t, std::nano>(); }

    static value_type record()
    {
        return value_type{ { get_char_written(), get_timestamp() } };
    }

    static auto get_timing_unit()
    {
        static auto _value = units::sec;
        if(settings::timing_units().length() > 0)
            _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
        return _value;
    }

    TIMEMORY_NODISCARD std::string get_display() const
    {
        std::stringstream ss;
        std::stringstream ssv;
        std::stringstream ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << ", " << ssr.str();
        ss << " wchar";
        return ss.str();
    }

    TIMEMORY_NODISCARD result_type get() const
    {
        auto val = base_type::load();

        double data  = std::get<0>(val);
        double delta = std::get<1>(val);

        delta /= static_cast<double>(std::nano::den);
        delta *= get_timing_unit();

        double rate = 0.0;
        if(delta != 0.0)
            rate = data / delta;

        if(laps > 0)
            rate *= laps;

        data /= std::get<0>(get_unit());
        rate /= std::get<0>(get_unit());

        if(!std::isfinite(rate))
            rate = 0.0;

        return result_type{ { data, rate } };
    }

    void start() { value = record(); }

    void stop()
    {
        using namespace tim::component::operators;
        auto diff         = (record() - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += (value = diff);
    }

    static unit_type get_unit()
    {
        static auto  _instance = this_type::unit();
        static auto& _mem      = std::get<0>(_instance);
        static auto& _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _timing_val =
                std::get<1>(units::get_timing_unit(settings::timing_units()));
            _rate = _mem / (_timing_val);
        }

        static const auto factor = static_cast<double>(std::nano::den);
        unit_type         _tmp   = _instance;
        std::get<1>(_tmp) *= factor;

        return _tmp;
    }

    static display_unit_type get_display_unit()
    {
        static display_unit_type _instance = this_type::display_unit();
        static auto&             _mem      = std::get<0>(_instance);
        static auto&             _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
            _rate      = mpl::apply<std::string>::join('/', _mem, _tval);
        }
        else if(settings::memory_units().length() > 0)
        {
            _rate = mpl::apply<std::string>::join('/', _mem, "sec");
        }

        return _instance;
    }

    /// sample a measurement
    void sample()
    {
        std::get<0>(accum) = std::get<0>(value) =
            std::max<int64_t>(std::get<0>(value), get_char_written());
    }

    /// sample a measurement from cached data
    void sample(const cache_type& _cache)
    {
        std::get<0>(accum) = std::get<0>(value) =
            std::max<int64_t>(std::get<0>(value), _cache.get_char_written());
    }

    /// read the value from cached data
    static value_type record(const cache_type& _cache)
    {
        return value_type{ { _cache.get_char_written(), get_timestamp() } };
    }

    /// start a measurement using the cached data
    void start(const cache_type& _cache) { value = record(_cache); }

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache)
    {
        using namespace tim::component::operators;
        auto diff         = (record(_cache) - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += (value = diff);
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::read_bytes
/// \brief I/O counter for bytes read. Attempt to count the number of bytes which this
/// process really did cause to be fetched from the storage layer. Done at the
/// submit_bio() level, so it is accurate for block-backed filesystems.
struct read_bytes : public base<read_bytes, std::pair<int64_t, int64_t>>
{
    using this_type         = read_bytes;
    using value_type        = std::pair<int64_t, int64_t>;
    using base_type         = base<this_type, value_type>;
    using result_type       = std::pair<double, double>;
    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    static std::string label() { return "read_bytes"; }
    static std::string description()
    {
        return "Number of bytes which this process really did cause to be fetched from "
               "the storage layer";
    }

    static std::pair<double, double> unit()
    {
        return std::pair<double, double>{
            units::megabyte, static_cast<double>(units::megabyte) / units::sec
        };
    }

    static std::vector<std::string> display_unit_array()
    {
        return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                         std::get<1>(get_display_unit()) };
    }

    static std::vector<std::string> label_array()
    {
        return std::vector<std::string>{ label(), "read_rate" };
    }

    static display_unit_type display_unit()
    {
        return display_unit_type{ "MB", "MB/sec" };
    }

    static std::pair<double, double> unit_array() { return unit(); }

    static std::vector<std::string> description_array()
    {
        return std::vector<std::string>{ "Number of bytes read", "Rate of bytes read" };
    }

    static auto get_timestamp() { return tim::get_clock_real_now<int64_t, std::nano>(); }

    static value_type record() { return value_type(get_bytes_read(), get_timestamp()); }

    static auto get_timing_unit()
    {
        static auto _value = units::sec;
        if(settings::timing_units().length() > 0)
            _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
        return _value;
    }

    TIMEMORY_NODISCARD std::string get_display() const
    {
        std::stringstream ss;
        std::stringstream ssv;
        std::stringstream ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << ", " << ssr.str();
        ss << " read_bytes";
        return ss.str();
    }

    TIMEMORY_NODISCARD result_type get() const
    {
        auto val = base_type::load();

        double data  = std::get<0>(val);
        double delta = std::get<1>(val);

        delta /= static_cast<double>(std::nano::den);
        delta *= get_timing_unit();

        double rate = 0.0;
        if(delta != 0.0)
            rate = data / delta;

        if(laps > 0)
            rate *= laps;

        data /= std::get<0>(get_unit());
        rate /= std::get<0>(get_unit());

        if(!std::isfinite(rate))
            rate = 0.0;

        return result_type(data, rate);
    }

    void start() { value = record(); }

    void stop()
    {
        using namespace tim::component::operators;
        auto diff         = (record() - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += (value = diff);
    }

    static unit_type get_unit()
    {
        static auto  _instance = this_type::unit();
        static auto& _mem      = std::get<0>(_instance);
        static auto& _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _timing_val =
                std::get<1>(units::get_timing_unit(settings::timing_units()));
            _rate = _mem / (_timing_val);
        }

        static const auto factor = static_cast<double>(std::nano::den);
        unit_type         _tmp   = _instance;
        std::get<1>(_tmp) *= factor;

        return _tmp;
    }

    static display_unit_type get_display_unit()
    {
        static display_unit_type _instance = this_type::display_unit();
        static auto&             _mem      = std::get<0>(_instance);
        static auto&             _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
            _rate      = mpl::apply<std::string>::join('/', _mem, _tval);
        }
        else if(settings::memory_units().length() > 0)
        {
            _rate = mpl::apply<std::string>::join('/', _mem, "sec");
        }

        return _instance;
    }

    /// sample a measurement
    void sample()
    {
        std::get<0>(accum) = std::get<0>(value) =
            std::max<int64_t>(std::get<0>(value), get_bytes_read());
    }

    /// sample a measurement from cached data
    void sample(const cache_type& _cache)
    {
        std::get<0>(accum) = std::get<0>(value) =
            std::max<int64_t>(std::get<0>(value), _cache.get_bytes_read());
    }

    /// read the value from the cache
    static value_type record(const cache_type& _cache)
    {
        return value_type{ _cache.get_bytes_read(), get_timestamp() };
    }

    /// start a measurement using the cached data
    void start(const cache_type& _cache) { value = record(_cache); }

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache)
    {
        using namespace tim::component::operators;
        auto diff         = (record(_cache) - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += (value = diff);
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::written_bytes
/// \brief I/O counter for bytes written. Attempt to count the number of bytes which this
/// process caused to be sent to the storage layer. This is done at page-dirtying time.
struct written_bytes : public base<written_bytes, std::array<int64_t, 2>>
{
    using this_type         = written_bytes;
    using value_type        = std::array<int64_t, 2>;
    using base_type         = base<this_type, value_type>;
    using result_type       = std::array<double, 2>;
    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    static std::string label() { return "written_bytes"; }
    static std::string description()
    {
        return "Number of bytes sent to the storage layer";
    }

    static result_type unit()
    {
        return result_type{ { units::megabyte,
                              static_cast<double>(units::megabyte) / units::sec } };
    }

    static std::vector<std::string> display_unit_array()
    {
        return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                         std::get<1>(get_display_unit()) };
    }

    static std::vector<std::string> label_array()
    {
        return std::vector<std::string>{ label(), "written_rate" };
    }

    static display_unit_type display_unit()
    {
        return display_unit_type{ { "MB", "MB/sec" } };
    }

    static std::array<double, 2> unit_array() { return unit(); }

    static std::vector<std::string> description_array()
    {
        return std::vector<std::string>{ "Number of bytes written",
                                         "Rate of bytes written" };
    }

    static auto get_timestamp() { return tim::get_clock_real_now<int64_t, std::nano>(); }

    static value_type record()
    {
        return value_type{ { get_bytes_written(), get_timestamp() } };
    }

    static auto get_timing_unit()
    {
        static auto _value = units::sec;
        if(settings::timing_units().length() > 0)
            _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
        return _value;
    }

    TIMEMORY_NODISCARD std::string get_display() const
    {
        std::stringstream ss;
        std::stringstream ssv;
        std::stringstream ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << ", " << ssr.str();
        ss << " write_bytes";
        return ss.str();
    }

    TIMEMORY_NODISCARD result_type get() const
    {
        auto val = base_type::load();

        double data  = std::get<0>(val);
        double delta = std::get<1>(val);

        delta /= static_cast<double>(std::nano::den);
        delta *= get_timing_unit();

        double rate = 0.0;
        if(delta != 0.0)
            rate = data / delta;

        if(laps > 0)
            rate *= laps;

        data /= std::get<0>(get_unit());
        rate /= std::get<0>(get_unit());

        if(!std::isfinite(rate))
            rate = 0.0;

        return result_type{ { data, rate } };
    }

    void start() { value = record(); }

    void stop()
    {
        using namespace tim::component::operators;
        auto diff         = (record() - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += (value = diff);
    }

    static unit_type get_unit()
    {
        static auto  _instance = this_type::unit();
        static auto& _mem      = std::get<0>(_instance);
        static auto& _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _timing_val =
                std::get<1>(units::get_timing_unit(settings::timing_units()));
            _rate = _mem / (_timing_val);
        }

        static const auto factor = static_cast<double>(std::nano::den);
        unit_type         _tmp   = _instance;
        std::get<1>(_tmp) *= factor;

        return _tmp;
    }

    static display_unit_type get_display_unit()
    {
        static display_unit_type _instance = this_type::display_unit();
        static auto&             _mem      = std::get<0>(_instance);
        static auto&             _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
            _rate      = mpl::apply<std::string>::join('/', _mem, _tval);
        }
        else if(settings::memory_units().length() > 0)
        {
            _rate = mpl::apply<std::string>::join('/', _mem, "sec");
        }

        return _instance;
    }

    /// sample a measurement
    void sample()
    {
        std::get<0>(accum) = std::get<0>(value) =
            std::max<int64_t>(std::get<0>(value), get_bytes_written());
    }

    /// sample a measurement from cached data
    void sample(const cache_type& _cache)
    {
        std::get<0>(accum) = std::get<0>(value) =
            std::max<int64_t>(std::get<0>(value), _cache.get_bytes_written());
    }

    /// read the value from the cache
    static value_type record(const cache_type& _cache)
    {
        return value_type{ { _cache.get_bytes_written(), get_timestamp() } };
    }

    /// start a measurement using the cached data
    void start(const cache_type& _cache) { value = record(_cache); }

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache)
    {
        using namespace tim::component::operators;
        auto diff         = (record(_cache) - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += (value = diff);
    }
};
//
//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
