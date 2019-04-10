//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file environment.hpp
 * \headerfile environment.hpp "timemory/environment.hpp"
 * Handles TiMemory settings via environment
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include <timemory/macros.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#    define TIMEMORY_DEFAULT_ENABLED true
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace env
{
//--------------------------------------------------------------------------------------//

typedef std::string string_t;

#define DEFINE_STATIC_ACCESSOR_FUNCTION(TYPE, FUNC, INIT)                                \
    inline TYPE& FUNC()                                                                  \
    {                                                                                    \
        static TYPE instance = INIT;                                                     \
        return instance;                                                                 \
    }

//--------------------------------------------------------------------------------------//

DEFINE_STATIC_ACCESSOR_FUNCTION(int, verbose, 0)
DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, env_num_threads, "TIMEMORY_NUM_THREADS")
DEFINE_STATIC_ACCESSOR_FUNCTION(int, num_threads, 0)
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, max_depth, std::numeric_limits<uint16_t>::max())
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, enabled, true)

DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, timing_precision, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, timing_width, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, timing_units, "")
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, timing_scientific, false)

DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, memory_precision, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, memory_width, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, memory_units, "")
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, memory_scientific, false)

DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, output_path, "timemory_output/")
DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, output_prefix, "")

//--------------------------------------------------------------------------------------//

string_t tolower(string_t);
string_t toupper(string_t);
void
parse();
inline string_t
get_output_prefix();
inline string_t
compose_output_filename(const std::string& _tag, std::string _ext);

//--------------------------------------------------------------------------------------//

}  // namespace env

}  // namespace tim

#include "timemory/components.hpp"
#include "timemory/macros.hpp"
#include "timemory/units.hpp"
#include "timemory/utility.hpp"

//======================================================================================//

inline std::string
tim::env::tolower(std::string str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}

//======================================================================================//

inline std::string
tim::env::toupper(std::string str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}

//--------------------------------------------------------------------------------------//

inline void
tim::env::parse()
{
    using namespace tim::component;

    auto get_env_bool = [](const std::string& _env_var, const bool& _default) {
        return (tim::get_env<int>(_env_var, static_cast<int>(_default)) > 0) ? true
                                                                             : false;
    };

    output_path()   = tim::get_env("TIMEMORY_OUTPUT_PATH", output_path());
    output_prefix() = tim::get_env("TIMEMORY_OUTPUT_PREFIX", output_prefix());

    verbose()         = tim::get_env("TIMEMORY_VERBOSE", verbose());
    env_num_threads() = tim::get_env("TIMEMORY_NUM_THREADS_ENV", env_num_threads());
    num_threads()     = tim::get_env(env_num_threads(), num_threads());
    max_depth()       = tim::get_env("TIMEMORY_MAX_DEPTH", max_depth());
    enabled()         = get_env_bool("TIMEMORY_ENABLE", enabled());

    timing_precision()  = tim::get_env("TIMEMORY_TIMING_PRECISION", timing_precision());
    timing_width()      = tim::get_env("TIMEMORY_TIMING_WIDTH", timing_width());
    timing_units()      = tim::get_env("TIMEMORY_TIMING_UNITS", timing_units());
    timing_scientific() = get_env_bool("TIMEMORY_TIMING_SCIENTIFIC", timing_scientific());

    memory_precision()  = tim::get_env("TIMEMORY_MEMORY_PRECISION", memory_precision());
    memory_width()      = tim::get_env("TIMEMORY_MEMORY_WIDTH", memory_width());
    memory_units()      = tim::get_env("TIMEMORY_MEMORY_UNITS", memory_units());
    memory_scientific() = get_env_bool("TIMEMORY_MEMORY_SCIENTIFIC", memory_scientific());

    //------------------------------------------------------------------------//
    //  Helper function for memory units processing
    auto get_memory_unit = [&](string_t _unit) {
        using return_type = std::tuple<std::string, long>;

        if(_unit.length() == 0)
            return return_type("MB", tim::units::megabyte);

        using inner            = std::tuple<string_t, string_t, string_t, intmax_t>;
        using pair_vector_t    = std::vector<inner>;
        pair_vector_t matching = { inner("byte", "B", "Bi", tim::units::byte),
                                   inner("kilobyte", "KB", "KiB", tim::units::kilobyte),
                                   inner("megabyte", "MB", "MiB", tim::units::megabyte),
                                   inner("gigabyte", "GB", "GiB", tim::units::gigabyte),
                                   inner("terabyte", "TB", "TiB", tim::units::terabyte),
                                   inner("petabyte", "PB", "PiB", tim::units::petabyte) };

        _unit = tolower(_unit);
        for(const auto& itr : matching)
            if(_unit == tolower(std::get<0>(itr)) || _unit == tolower(std::get<1>(itr)) ||
               _unit == tolower(std::get<2>(itr)))
                return return_type(std::get<1>(itr), std::get<3>(itr));
        std::cerr << "Warning!! No memory unit matching \"" << _unit
                  << "\". Using default..." << std::endl;
        return return_type("MB", tim::units::megabyte);
    };
    //------------------------------------------------------------------------//
    //  Helper function for timing units processing
    auto get_timing_unit = [&](string_t _unit) {
        using return_type = std::tuple<std::string, long>;

        if(_unit.length() == 0)
            return return_type("sec", tim::units::sec);

        using inner            = std::tuple<string_t, string_t, intmax_t>;
        using pair_vector_t    = std::vector<inner>;
        pair_vector_t matching = { inner("ps", "picosecond", tim::units::psec),
                                   inner("ns", "nanosecond", tim::units::nsec),
                                   inner("us", "microsecond", tim::units::usec),
                                   inner("ms", "millisecond", tim::units::msec),
                                   inner("cs", "centisecond", tim::units::csec),
                                   inner("ds", "decisecond", tim::units::dsec),
                                   inner("s", "second", tim::units::sec) };

        _unit = tolower(_unit);
        for(const auto& itr : matching)
            if(_unit == tolower(std::get<0>(itr)) || _unit == tolower(std::get<1>(itr)) ||
               _unit == (tolower(std::get<0>(itr)) + "ec") ||
               _unit == (tolower(std::get<1>(itr)) + "s"))
                return return_type(std::get<0>(itr), std::get<2>(itr));
        std::cerr << "Warning!! No timing unit matching \"" << _unit
                  << "\". Using default..." << std::endl;
        return return_type("sec", tim::units::sec);
    };

    if(!(memory_width() < 0))
    {
        peak_rss::get_width()                 = memory_width();
        current_rss::get_width()              = memory_width();
        stack_rss::get_width()                = memory_width();
        data_rss::get_width()                 = memory_width();
        num_swap::get_width()                 = memory_width();
        num_io_in::get_width()                = memory_width();
        num_io_out::get_width()               = memory_width();
        num_major_page_faults::get_width()    = memory_width();
        num_minor_page_faults::get_width()    = memory_width();
        num_msg_sent::get_width()             = memory_width();
        num_msg_recv::get_width()             = memory_width();
        num_signals::get_width()              = memory_width();
        voluntary_context_switch::get_width() = memory_width();
        priority_context_switch::get_width()  = memory_width();
    }

    if(!(timing_width() < 0))
    {
        real_clock::get_width()          = timing_width();
        system_clock::get_width()        = timing_width();
        user_clock::get_width()          = timing_width();
        cpu_clock::get_width()           = timing_width();
        monotonic_clock::get_width()     = timing_width();
        monotonic_raw_clock::get_width() = timing_width();
        thread_cpu_clock::get_width()    = timing_width();
        process_cpu_clock::get_width()   = timing_width();
        cpu_util::get_width()            = timing_width();
        thread_cpu_util::get_width()     = timing_width();
        process_cpu_util::get_width()    = timing_width();
    }

    if(memory_scientific())
    {
        peak_rss::get_format_flags()                 = std::ios_base::scientific;
        current_rss::get_format_flags()              = std::ios_base::scientific;
        stack_rss::get_format_flags()                = std::ios_base::scientific;
        data_rss::get_format_flags()                 = std::ios_base::scientific;
        num_swap::get_format_flags()                 = std::ios_base::scientific;
        num_io_in::get_format_flags()                = std::ios_base::scientific;
        num_io_out::get_format_flags()               = std::ios_base::scientific;
        num_major_page_faults::get_format_flags()    = std::ios_base::scientific;
        num_minor_page_faults::get_format_flags()    = std::ios_base::scientific;
        num_msg_sent::get_format_flags()             = std::ios_base::scientific;
        num_msg_recv::get_format_flags()             = std::ios_base::scientific;
        num_signals::get_format_flags()              = std::ios_base::scientific;
        voluntary_context_switch::get_format_flags() = std::ios_base::scientific;
        priority_context_switch::get_format_flags()  = std::ios_base::scientific;
    }

    if(timing_scientific())
    {
        real_clock::get_format_flags()          = std::ios_base::scientific;
        system_clock::get_format_flags()        = std::ios_base::scientific;
        user_clock::get_format_flags()          = std::ios_base::scientific;
        cpu_clock::get_format_flags()           = std::ios_base::scientific;
        monotonic_clock::get_format_flags()     = std::ios_base::scientific;
        monotonic_raw_clock::get_format_flags() = std::ios_base::scientific;
        thread_cpu_clock::get_format_flags()    = std::ios_base::scientific;
        process_cpu_clock::get_format_flags()   = std::ios_base::scientific;
        cpu_util::get_format_flags()            = std::ios_base::scientific;
        thread_cpu_util::get_format_flags()     = std::ios_base::scientific;
        process_cpu_util::get_format_flags()    = std::ios_base::scientific;
    }

    if(!(memory_precision() < 0))
    {
        peak_rss::get_precision()                 = memory_precision();
        current_rss::get_precision()              = memory_precision();
        stack_rss::get_precision()                = memory_precision();
        data_rss::get_precision()                 = memory_precision();
        num_swap::get_precision()                 = memory_precision();
        num_io_in::get_precision()                = memory_precision();
        num_io_out::get_precision()               = memory_precision();
        num_major_page_faults::get_precision()    = memory_precision();
        num_minor_page_faults::get_precision()    = memory_precision();
        num_msg_sent::get_precision()             = memory_precision();
        num_msg_recv::get_precision()             = memory_precision();
        num_signals::get_precision()              = memory_precision();
        voluntary_context_switch::get_precision() = memory_precision();
        priority_context_switch::get_precision()  = memory_precision();
    }

    if(!(timing_precision() < 0))
    {
        real_clock::get_precision()          = timing_precision();
        system_clock::get_precision()        = timing_precision();
        user_clock::get_precision()          = timing_precision();
        cpu_clock::get_precision()           = timing_precision();
        monotonic_clock::get_precision()     = timing_precision();
        monotonic_raw_clock::get_precision() = timing_precision();
        thread_cpu_clock::get_precision()    = timing_precision();
        process_cpu_clock::get_precision()   = timing_precision();
        cpu_util::get_precision()            = timing_precision();
        thread_cpu_util::get_precision()     = timing_precision();
        process_cpu_util::get_precision()    = timing_precision();
    }

    if(memory_units().length() > 0)
    {
        auto _memory_unit = get_memory_unit(memory_units());

        peak_rss::get_display_unit()    = std::get<0>(_memory_unit);
        current_rss::get_display_unit() = std::get<0>(_memory_unit);
        stack_rss::get_display_unit()   = std::get<0>(_memory_unit);
        data_rss::get_display_unit()    = std::get<0>(_memory_unit);

        peak_rss::get_unit()    = std::get<1>(_memory_unit);
        current_rss::get_unit() = std::get<1>(_memory_unit);
        stack_rss::get_unit()   = std::get<1>(_memory_unit);
        data_rss::get_unit()    = std::get<1>(_memory_unit);
    }

    if(timing_units().length() > 0)
    {
        auto _timing_unit = get_timing_unit(timing_units());

        real_clock::get_display_unit()          = std::get<0>(_timing_unit);
        system_clock::get_display_unit()        = std::get<0>(_timing_unit);
        user_clock::get_display_unit()          = std::get<0>(_timing_unit);
        cpu_clock::get_display_unit()           = std::get<0>(_timing_unit);
        monotonic_clock::get_display_unit()     = std::get<0>(_timing_unit);
        monotonic_raw_clock::get_display_unit() = std::get<0>(_timing_unit);
        thread_cpu_clock::get_display_unit()    = std::get<0>(_timing_unit);
        process_cpu_clock::get_display_unit()   = std::get<0>(_timing_unit);

        real_clock::get_unit()          = std::get<1>(_timing_unit);
        system_clock::get_unit()        = std::get<1>(_timing_unit);
        user_clock::get_unit()          = std::get<1>(_timing_unit);
        cpu_clock::get_unit()           = std::get<1>(_timing_unit);
        monotonic_clock::get_unit()     = std::get<1>(_timing_unit);
        monotonic_raw_clock::get_unit() = std::get<1>(_timing_unit);
        thread_cpu_clock::get_unit()    = std::get<1>(_timing_unit);
        process_cpu_clock::get_unit()   = std::get<1>(_timing_unit);
    }
}

//--------------------------------------------------------------------------------------//

#include <timemory/mpi.hpp>
#include <timemory/utility.hpp>

//--------------------------------------------------------------------------------------//

inline tim::env::string_t
tim::env::get_output_prefix()
{
    auto dir = output_path();
    auto ret = makedir(dir);
    return (ret == 0) ? path_t(dir + string_t("/") + output_prefix())
                      : path_t(string_t("./") + output_prefix());
}

//--------------------------------------------------------------------------------------//

inline tim::env::string_t
tim::env::compose_output_filename(const std::string& _tag, std::string _ext)
{
    auto _prefix      = get_output_prefix();
    auto _rank_suffix = (!mpi_is_initialized())
                            ? std::string("")
                            : (std::string("_") + std::to_string(mpi_rank()));
    if(_ext.find(".") != 0)
        _ext = std::string(".") + _ext;
    auto plast = _prefix.length() - 1;
    if(_prefix.length() > 0 && _prefix[plast] != '/' && isalnum(_prefix[plast]))
        _prefix += "_";
    auto fpath = path_t(_prefix + _tag + _rank_suffix + _ext);
    while(fpath.find("//") != std::string::npos)
        fpath.replace(fpath.find("//"), 2, "/");
    return std::move(fpath);
}

//--------------------------------------------------------------------------------------//
