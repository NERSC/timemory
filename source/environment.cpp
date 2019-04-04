//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
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

#include "timemory/environment.hpp"
#include "timemory/components.hpp"
#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/units.hpp"
#include "timemory/utility.hpp"

using namespace tim::component;

template <typename Type>
using CType = typename Type::value_type;

//======================================================================================//

bool
get_env_bool(const tim::string& _env_var, const bool& _default)
{
    return (tim::get_env<int>(_env_var, static_cast<int>(_default)) > 0) ? true : false;
}

//======================================================================================//

namespace tim
{
namespace env
{
//======================================================================================//

int      verbose              = 0;
bool     disable_timer_memory = false;
string_t env_num_threads      = "TIMEMORY_NUM_THREADS";
int      num_threads          = 0;
int      max_depth            = std::numeric_limits<uint16_t>::max();
bool     enabled              = TIMEMORY_DEFAULT_ENABLED;

int16_t  timing_precision  = -1;
int16_t  timing_width      = -1;
string_t timing_units      = "";
bool     timing_scientific = false;

int16_t  memory_precision  = -1;
int16_t  memory_width      = -1;
string_t memory_units      = "";
bool     memory_scientific = false;

//======================================================================================//

string_t
tolower(string_t str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}

//======================================================================================//

string_t
toupper(string_t str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}

//======================================================================================//

void
parse()
{
    verbose         = tim::get_env("TIMEMORY_VERBOSE", verbose);
    env_num_threads = tim::get_env("TIMEMORY_NUM_THREADS_ENV", env_num_threads);
    num_threads     = tim::get_env(env_num_threads, num_threads);
    max_depth       = tim::get_env("TIMEMORY_MAX_DEPTH", max_depth);
    enabled         = get_env_bool("TIMEMORY_ENABLE", enabled);

    timing_precision  = tim::get_env("TIMEMORY_TIMING_PRECISION", timing_precision);
    timing_width      = tim::get_env("TIMEMORY_TIMING_WIDTH", timing_width);
    timing_units      = tim::get_env("TIMEMORY_TIMING_UNITS", timing_units);
    timing_scientific = get_env_bool("TIMEMORY_TIMING_SCIENTIFIC", timing_scientific);

    memory_precision  = tim::get_env("TIMEMORY_MEMORY_PRECISION", memory_precision);
    memory_width      = tim::get_env("TIMEMORY_MEMORY_WIDTH", memory_width);
    memory_units      = tim::get_env("TIMEMORY_MEMORY_UNITS", memory_units);
    memory_scientific = get_env_bool("TIMEMORY_MEMORY_SCIENTIFIC", memory_scientific);

    //------------------------------------------------------------------------//
    //  Helper function for timing units processing
    auto get_timing_unit = [&](string_t _unit) {
        using return_type = std::tuple<std::string, long>;

        if(_unit.length() == 0)
            return return_type("", 0L);

        using inner            = std::tuple<string_t, string_t, intmax_t>;
        using pair_vector_t    = std::vector<inner>;
        pair_vector_t matching = { inner("psec", "picosecond", tim::units::psec),
                                   inner("nsec", "nanosecond", tim::units::nsec),
                                   inner("usec", "microsecond", tim::units::usec),
                                   inner("msec", "millisecond", tim::units::msec),
                                   inner("csec", "centisecond", tim::units::csec),
                                   inner("dsec", "decisecond", tim::units::dsec),
                                   inner("sec", "second", tim::units::sec) };

        _unit = tolower(_unit);
        for(const auto& itr : matching)
            if(_unit == tolower(std::get<0>(itr)) || _unit == tolower(std::get<1>(itr)) ||
               _unit == tolower(std::get<1>(itr)) + "s")
                return return_type(std::get<0>(itr), std::get<2>(itr));
        return return_type("", 0L);
    };
    //------------------------------------------------------------------------//
    //  Helper function for memory units processing
    auto get_memory_unit = [&](string_t _unit) {
        using return_type = std::tuple<std::string, long>;

        if(_unit.length() == 0)
            return return_type("", 0L);

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
        return return_type("", 0L);
    };

    if(!(memory_width < 0))
    {
        base<peak_rss>::get_width()              = memory_width;
        base<current_rss>::get_width()           = memory_width;
        base<stack_rss>::get_width()             = memory_width;
        base<data_rss>::get_width()              = memory_width;
        base<num_swap>::get_width()              = memory_width;
        base<num_io_in>::get_width()             = memory_width;
        base<num_io_out>::get_width()            = memory_width;
        base<num_major_page_faults>::get_width() = memory_width;
        base<num_minor_page_faults>::get_width() = memory_width;
    }

    if(!(timing_width < 0))
    {
        base<real_clock>::get_width()                                = timing_width;
        base<system_clock>::get_width()                              = timing_width;
        base<user_clock>::get_width()                                = timing_width;
        base<cpu_clock>::get_width()                                 = timing_width;
        base<monotonic_clock>::get_width()                           = timing_width;
        base<monotonic_raw_clock>::get_width()                       = timing_width;
        base<thread_cpu_clock>::get_width()                          = timing_width;
        base<process_cpu_clock>::get_width()                         = timing_width;
        base<cpu_util, CType<cpu_util>>::get_width()                 = timing_width;
        base<thread_cpu_util, CType<thread_cpu_util>>::get_width()   = timing_width;
        base<process_cpu_util, CType<process_cpu_util>>::get_width() = timing_width;
    }

    if(memory_scientific)
    {
        base<peak_rss>::get_format_flags()              = std::ios_base::scientific;
        base<current_rss>::get_format_flags()           = std::ios_base::scientific;
        base<stack_rss>::get_format_flags()             = std::ios_base::scientific;
        base<data_rss>::get_format_flags()              = std::ios_base::scientific;
        base<num_swap>::get_format_flags()              = std::ios_base::scientific;
        base<num_io_in>::get_format_flags()             = std::ios_base::scientific;
        base<num_io_out>::get_format_flags()            = std::ios_base::scientific;
        base<num_major_page_faults>::get_format_flags() = std::ios_base::scientific;
        base<num_minor_page_faults>::get_format_flags() = std::ios_base::scientific;
    }

    if(!(memory_precision < 0))
    {
        base<peak_rss>::get_precision()              = memory_precision;
        base<current_rss>::get_precision()           = memory_precision;
        base<stack_rss>::get_precision()             = memory_precision;
        base<data_rss>::get_precision()              = memory_precision;
        base<num_swap>::get_precision()              = memory_precision;
        base<num_io_in>::get_precision()             = memory_precision;
        base<num_io_out>::get_precision()            = memory_precision;
        base<num_major_page_faults>::get_precision() = memory_precision;
        base<num_minor_page_faults>::get_precision() = memory_precision;
    }

    if(!(timing_precision < 0))
    {
        base<real_clock>::get_precision()                              = timing_precision;
        base<system_clock>::get_precision()                            = timing_precision;
        base<user_clock>::get_precision()                              = timing_precision;
        base<cpu_clock>::get_precision()                               = timing_precision;
        base<monotonic_clock>::get_precision()                         = timing_precision;
        base<monotonic_raw_clock>::get_precision()                     = timing_precision;
        base<thread_cpu_clock>::get_precision()                        = timing_precision;
        base<process_cpu_clock>::get_precision()                       = timing_precision;
        base<cpu_util, CType<cpu_util>>::get_precision()               = timing_precision;
        base<thread_cpu_util, CType<thread_cpu_util>>::get_precision() = timing_precision;
        base<process_cpu_util, CType<process_cpu_util>>::get_precision() =
            timing_precision;
    }

    if(memory_units.length() > 0)
    {
        auto _memory_unit = get_memory_unit(memory_units);

        base<peak_rss>::get_display_unit()    = std::get<0>(_memory_unit);
        base<current_rss>::get_display_unit() = std::get<0>(_memory_unit);
        base<stack_rss>::get_display_unit()   = std::get<0>(_memory_unit);
        base<data_rss>::get_display_unit()    = std::get<0>(_memory_unit);

        base<peak_rss>::get_unit()    = std::get<1>(_memory_unit);
        base<current_rss>::get_unit() = std::get<1>(_memory_unit);
        base<stack_rss>::get_unit()   = std::get<1>(_memory_unit);
        base<data_rss>::get_unit()    = std::get<1>(_memory_unit);
    }

    if(timing_units.length() > 0)
    {
        auto _timing_unit = get_timing_unit(timing_units);

        base<real_clock>::get_display_unit()          = std::get<0>(_timing_unit);
        base<system_clock>::get_display_unit()        = std::get<0>(_timing_unit);
        base<user_clock>::get_display_unit()          = std::get<0>(_timing_unit);
        base<cpu_clock>::get_display_unit()           = std::get<0>(_timing_unit);
        base<monotonic_clock>::get_display_unit()     = std::get<0>(_timing_unit);
        base<monotonic_raw_clock>::get_display_unit() = std::get<0>(_timing_unit);
        base<thread_cpu_clock>::get_display_unit()    = std::get<0>(_timing_unit);
        base<process_cpu_clock>::get_display_unit()   = std::get<0>(_timing_unit);

        base<real_clock>::get_unit()          = std::get<1>(_timing_unit);
        base<system_clock>::get_unit()        = std::get<1>(_timing_unit);
        base<user_clock>::get_unit()          = std::get<1>(_timing_unit);
        base<cpu_clock>::get_unit()           = std::get<1>(_timing_unit);
        base<monotonic_clock>::get_unit()     = std::get<1>(_timing_unit);
        base<monotonic_raw_clock>::get_unit() = std::get<1>(_timing_unit);
        base<thread_cpu_clock>::get_unit()    = std::get<1>(_timing_unit);
        base<process_cpu_clock>::get_unit()   = std::get<1>(_timing_unit);
    }

    tim::manager::enable(enabled);
    tim::manager::max_depth(max_depth);
}

//======================================================================================//

}  // namespace env

}  // namespace tim

//======================================================================================//
