//  MIT License
//  
//  Copyright (c) 2018, The Regents of the University of California, 
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

#include "timemory/macros.hpp"
#include "timemory/environment.hpp"
#include "timemory/utility.hpp"
#include "timemory/formatters.hpp"
#include "timemory/manager.hpp"

//============================================================================//

bool get_env_bool(const std::string& _env_var, const bool& _default)
{
    return (tim::get_env<int>(_env_var, static_cast<int>(_default)) > 0)
            ? true : false;
}

//============================================================================//

namespace tim
{

namespace env
{

//============================================================================//

int         verbose                     = 0;
bool        disable_timer_memory        = false;
//bool        output_total                = false;
string_t    env_num_threads             = "TIMEMORY_NUM_THREADS";
int         num_threads                 = 0;
int         max_depth                   = std::numeric_limits<uint16_t>::max();
bool        enabled                     = TIMEMORY_DEFAULT_ENABLED;

string_t    timing_format               = "";
int16_t     timing_precision            = -1;
int16_t     timing_width                = -1;
string_t    timing_units                = "";
bool        timing_scientific           = false;

string_t    memory_format               = "";
int16_t     memory_precision            = -1;
int16_t     memory_width                = -1;
string_t    memory_units                = "";
bool        memory_scientific           = false;

string_t    timing_memory_format        = "";
int16_t     timing_memory_precision     = -1;
int16_t     timing_memory_width         = -1;
string_t    timing_memory_units         = "";
bool        timing_memory_scientific    = false;

//============================================================================//

string_t tolower(string_t str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}

//============================================================================//

string_t toupper(string_t str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}

//============================================================================//

void parse()
{
    typedef tim::format::core_formatter         core_format_t;
    typedef std::function<int64_t(string_t)>    unit_func_t;

    verbose                 = tim::get_env<int>     ("TIMEMORY_VERBOSE",                    verbose);
    disable_timer_memory    = get_env_bool          ("TIMEMORY_DISABLE_TIMER_MEMORY",       disable_timer_memory);
    //output_total            = get_env_bool          ("TIMEMORY_OUTPUT_TOTAL",               output_total);
    env_num_threads         = tim::get_env<string_t>("TIMEMORY_NUM_THREADS_ENV",            env_num_threads);
    num_threads             = tim::get_env<int>     (env_num_threads,                       num_threads);
    max_depth               = tim::get_env<int>     ("TIMEMORY_MAX_DEPTH",                  max_depth);
    enabled                 = get_env_bool          ("TIMEMORY_ENABLE",                     enabled);

    timing_format           = tim::get_env<string_t>("TIMEMORY_TIMING_FORMAT",              timing_format);
    timing_precision        = tim::get_env<int16_t> ("TIMEMORY_TIMING_PRECISION",           timing_precision);
    timing_width            = tim::get_env<int16_t> ("TIMEMORY_TIMING_WIDTH",               timing_width);
    timing_units            = tim::get_env<string_t>("TIMEMORY_TIMING_UNITS",               timing_units);
    timing_scientific       = get_env_bool          ("TIMEMORY_TIMING_SCIENTIFIC",          timing_scientific);

    memory_format           = tim::get_env<string_t>("TIMEMORY_MEMORY_FORMAT",              memory_format);
    memory_precision        = tim::get_env<int16_t> ("TIMEMORY_MEMORY_PRECISION",           memory_precision);
    memory_width            = tim::get_env<int16_t> ("TIMEMORY_MEMORY_WIDTH",               memory_width);
    memory_units            = tim::get_env<string_t>("TIMEMORY_MEMORY_UNITS",               memory_units);
    memory_scientific       = get_env_bool          ("TIMEMORY_MEMORY_SCIENTIFIC",          memory_scientific);

    timing_memory_format    = tim::get_env<string_t>("TIMEMORY_TIMING_MEMORY_FORMAT",       timing_memory_format);
    timing_memory_precision = tim::get_env<int16_t> ("TIMEMORY_TIMING_MEMORY_PRECISION",    timing_memory_precision);
    timing_memory_width     = tim::get_env<int16_t> ("TIMEMORY_TIMING_MEMORY_WIDTH",        timing_memory_width);
    timing_memory_units     = tim::get_env<string_t>("TIMEMORY_TIMING_MEMORY_UNITS",        timing_memory_units);
    timing_memory_scientific= get_env_bool          ("TIMEMORY_TIMING_MEMORY_SCIENTIFIC",   timing_memory_scientific);

    tim::format::timer::push();
    tim::format::rss::push();

    //------------------------------------------------------------------------//
    //  Helper function for timing units processing
    auto get_timing_unit = [&] (string_t _unit)
    {
        if(_unit.length() == 0)
            return (int64_t) 0;

        using inner = std::tuple<string_t, string_t, int64_t>;
        using pair_vector_t = std::vector<inner>;
        pair_vector_t matching =
        {
            inner( "psec",   "picosecond",   tim::units::psec    ),
            inner( "nsec",   "nanosecond",   tim::units::nsec    ),
            inner( "usec",   "microsecond",  tim::units::usec    ),
            inner( "msec",   "millisecond",  tim::units::msec    ),
            inner( "csec",   "centisecond",  tim::units::csec    ),
            inner( "dsec",   "decisecond",   tim::units::dsec    ),
            inner( "sec",    "second",       tim::units::sec     )
        };

        _unit = tolower(_unit);
        for(const auto& itr : matching)
            if(_unit == tolower(std::get<0>(itr)) ||
               _unit == tolower(std::get<1>(itr)) ||
               _unit == tolower(std::get<1>(itr)) + "s")
                return std::get<2>(itr);
        return (int64_t) 0;
    };
    //------------------------------------------------------------------------//
    //  Helper function for memory units processing
    auto get_memory_unit = [&] (string_t _unit)
    {
        if(_unit.length() == 0)
            return (int64_t) 0;

        using inner = std::tuple<string_t, string_t, string_t, int64_t>;
        using pair_vector_t = std::vector<inner>;
        pair_vector_t matching =
        {
            inner( "byte",       "B",    "Bi",   tim::units::byte        ),
            inner( "kilobyte",   "KB",   "KiB",  tim::units::kilobyte    ),
            inner( "megabyte",   "MB",   "MiB",  tim::units::megabyte    ),
            inner( "gigabyte",   "GB",   "GiB",  tim::units::gigabyte    ),
            inner( "terabyte",   "TB",   "TiB",  tim::units::terabyte    ),
            inner( "petabyte",   "PB",   "PiB",  tim::units::petabyte    )
        };

        _unit = tolower(_unit);
        for(const auto& itr : matching)
            if(_unit == tolower(std::get<0>(itr)) ||
               _unit == tolower(std::get<1>(itr)) ||
               _unit == tolower(std::get<2>(itr)))
                return std::get<3>(itr);
        return (int64_t) 0;
    };
    //------------------------------------------------------------------------//
    //  Helper lambda for modifying the core_formatter components
    //------------------------------------------------------------------------//
    auto set_core = [&] (core_format_t*  _core,
                         const string_t& _format,
                         const int16_t&  _prec,
                         const int16_t&  _width,
                         const bool&     _scientific,
                         const string_t& _unit_string,
                         unit_func_t     _unit_func)
    {
        int64_t _unit = _unit_func(_unit_string);

        if(_format.length() > 0)
            _core->format(_format);
        if(_prec > 0)
            _core->precision(_prec);
        if(_width > 0)
            _core->width(_width);
        if(_scientific)
            _core->fixed(false);
        if(_unit > 0)
            _core->unit(_unit);
    };
    //------------------------------------------------------------------------//

    tim::format::timer  _timing         = tim::format::timer::get_default();
    tim::format::rss&   _timing_memory  = _timing.rss_format();
    tim::format::rss    _memory         = tim::format::rss::get_default();

    set_core(&_timing,
             timing_format,
             timing_precision,
             timing_width,
             timing_scientific,
             timing_units,
             get_timing_unit);

    set_core(&_memory,
             memory_format,
             memory_precision,
             memory_width,
             memory_scientific,
             memory_units,
             get_memory_unit);

    set_core(&_timing_memory,
             timing_memory_format,
             timing_memory_precision,
             timing_memory_width,
             timing_memory_scientific,
             timing_memory_units,
             get_memory_unit);

    // set default timing format -- will be identical if no env set
    //  - memory format is included because _timing_memory is a reference
    tim::format::timer::set_default(_timing);
    // set default memory format -- will be identical if no env set
    tim::format::rss::set_default(_memory);

    tim::manager::enable(enabled);
    tim::manager::max_depth(max_depth);
}

//============================================================================//

} // namespace env

} // namespace tim

//============================================================================//
