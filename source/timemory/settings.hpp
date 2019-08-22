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

/** \file settings.hpp
 * \headerfile settings.hpp "timemory/settings.hpp"
 * Handles TiMemory settings, parses environment, provides initializer function
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/details/settings.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

//--------------------------------------------------------------------------------------//

namespace tim
{
// initialization (creates manager and configures output path)
void
timemory_init(int argc, char** argv, const std::string& _prefix = "timemory-",
              const std::string& _suffix = "-output");
// initialization (creates manager and configures output path)
void
timemory_init(const std::string& exe_name, const std::string& _prefix = "timemory-",
              const std::string& _suffix = "-output");
// finalization (optional)
void
timemory_finalize();

namespace settings
{
//--------------------------------------------------------------------------------------//

inline string_t tolower(string_t);
inline string_t toupper(string_t);
inline void
parse();
inline string_t
get_output_prefix();
inline string_t
compose_output_filename(const std::string& _tag, std::string _ext);

//--------------------------------------------------------------------------------------//

}  // namespace settings

}  // namespace tim

//======================================================================================//

inline std::string
tim::settings::tolower(std::string str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}

//======================================================================================//

inline std::string
tim::settings::toupper(std::string str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}

//======================================================================================//

#include "timemory/backends/mpi.hpp"
#include "timemory/components.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/utility.hpp"

namespace tim
{
using complete_tuple_t = std::tuple<
    component::caliper, component::cpu_clock, component::cpu_roofline_dp_flops,
    component::cpu_roofline_sp_flops, component::cpu_util, component::cuda_event,
    component::cupti_activity, component::cupti_counters, component::current_rss,
    component::data_rss, component::monotonic_clock, component::monotonic_raw_clock,
    component::num_io_in, component::num_io_out, component::num_major_page_faults,
    component::num_minor_page_faults, component::num_msg_recv, component::num_msg_sent,
    component::num_signals, component::num_swap, component::nvtx_marker,
    component::papi_array_t, component::peak_rss, component::priority_context_switch,
    component::process_cpu_clock, component::process_cpu_util, component::read_bytes,
    component::real_clock, component::stack_rss, component::system_clock,
    component::thread_cpu_clock, component::thread_cpu_util, component::trip_count,
    component::user_clock, component::voluntary_context_switch, component::written_bytes>;

namespace settings
{
template <typename _Tuple = tim::complete_tuple_t>
void
process();
}
}

//--------------------------------------------------------------------------------------//
// function to parse the environment for settings
//
inline void
tim::settings::parse()
{
    if(suppress_parsing())
    {
        process();
        return;
    }

    // logic
    enabled()     = tim::get_env("TIMEMORY_ENABLE", enabled());
    auto_output() = tim::get_env("TIMEMORY_AUTO_OUTPUT", auto_output());
    file_output() = tim::get_env("TIMEMORY_FILE_OUTPUT", file_output());
    text_output() = tim::get_env("TIMEMORY_TEXT_OUTPUT", text_output());
    json_output() = tim::get_env("TIMEMORY_JSON_OUTPUT", json_output());
    cout_output() = tim::get_env("TIMEMORY_COUT_OUTPUT", cout_output());

    // settings
    verbose()   = tim::get_env("TIMEMORY_VERBOSE", verbose());
    debug()     = tim::get_env("TIMEMORY_DEBUG", debug());
    max_depth() = tim::get_env("TIMEMORY_MAX_DEPTH", max_depth());

    // general formatting
    width()      = tim::get_env("TIMEMORY_WIDTH", width());
    precision()  = tim::get_env("TIMEMORY_PRECISION", precision());
    scientific() = tim::get_env("TIMEMORY_SCIENTIFIC", scientific());

    // timing formatting
    timing_precision()  = tim::get_env("TIMEMORY_TIMING_PRECISION", timing_precision());
    timing_width()      = tim::get_env("TIMEMORY_TIMING_WIDTH", timing_width());
    timing_units()      = tim::get_env("TIMEMORY_TIMING_UNITS", timing_units());
    timing_scientific() = tim::get_env("TIMEMORY_TIMING_SCIENTIFIC", timing_scientific());

    // memory formatting
    memory_precision()  = tim::get_env("TIMEMORY_MEMORY_PRECISION", memory_precision());
    memory_width()      = tim::get_env("TIMEMORY_MEMORY_WIDTH", memory_width());
    memory_units()      = tim::get_env("TIMEMORY_MEMORY_UNITS", memory_units());
    memory_scientific() = tim::get_env("TIMEMORY_MEMORY_SCIENTIFIC", memory_scientific());

    // file settings
    output_path()   = tim::get_env("TIMEMORY_OUTPUT_PATH", output_path());
    output_prefix() = tim::get_env("TIMEMORY_OUTPUT_PREFIX", output_prefix());

    process();
}

//--------------------------------------------------------------------------------------//
// function to process the settings -- always called even when environment processesing
// is suppressed
//
template <typename _Tuple>
inline void
tim::settings::process()
{
    using namespace tim::component;
    using category_timing  = impl::filter_false<trait::is_timing_category, _Tuple>;
    using has_timing_units = impl::filter_false<trait::uses_timing_units, _Tuple>;
    using category_memory  = impl::filter_false<trait::is_memory_category, _Tuple>;
    using has_memory_units = impl::filter_false<trait::uses_memory_units, _Tuple>;

    //------------------------------------------------------------------------//
    //  Helper function for memory units processing
    auto get_memory_unit = [&](string_t _unit) {
        using return_type = std::tuple<std::string, long>;

        if(_unit.length() == 0)
            return return_type("MB", tim::units::megabyte);

        using inner            = std::tuple<string_t, string_t, int64_t>;
        using pair_vector_t    = std::vector<inner>;
        pair_vector_t matching = { inner("byte", "B", tim::units::byte),
                                   inner("kilobyte", "KB", tim::units::kilobyte),
                                   inner("megabyte", "MB", tim::units::megabyte),
                                   inner("gigabyte", "GB", tim::units::gigabyte),
                                   inner("terabyte", "TB", tim::units::terabyte),
                                   inner("petabyte", "PB", tim::units::petabyte),
                                   inner("kibibyte", "KiB", tim::units::KiB),
                                   inner("mebibyte", "MiB", tim::units::MiB),
                                   inner("gibibyte", "GiB", tim::units::GiB),
                                   inner("tebibyte", "TiB", tim::units::TiB),
                                   inner("pebibyte", "PiB", tim::units::PiB) };

        _unit = tolower(_unit);
        for(const auto& itr : matching)
            if(_unit == tolower(std::get<0>(itr)) || _unit == tolower(std::get<1>(itr)))
                return return_type(std::get<1>(itr), std::get<2>(itr));
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

        using inner            = std::tuple<string_t, string_t, int64_t>;
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
            if(_unit == std::get<0>(itr) || _unit == std::get<1>(itr) ||
               _unit == (std::get<0>(itr) + "ec") || _unit == (std::get<1>(itr) + "s"))
            {
                return return_type(std::get<0>(itr) + "ec", std::get<2>(itr));
            }
        std::cerr << "Warning!! No timing unit matching \"" << _unit
                  << "\". Using default..." << std::endl;
        return return_type("sec", tim::units::sec);
    };

    if(precision() > 0)
    {
        apply<void>::type_access<operation::set_precision, _Tuple>(precision());
    }

    if(width() > 0)
    {
        apply<void>::type_access<operation::set_width, _Tuple>(width());
    }

    if(scientific())
    {
        apply<void>::type_access<operation::set_format_flags, _Tuple>(
            std::ios_base::scientific);
    }

    if(!(memory_width() < 0))
    {
        apply<void>::type_access<operation::set_width, category_memory>(memory_width());
    }

    if(!(timing_width() < 0))
    {
        apply<void>::type_access<operation::set_width, category_timing>(timing_width());
    }

    if(memory_scientific())
    {
        apply<void>::type_access<operation::set_format_flags, category_memory>(
            std::ios_base::scientific);
    }

    if(timing_scientific())
    {
        apply<void>::type_access<operation::set_format_flags, category_timing>(
            std::ios_base::scientific);
    }

    if(!(memory_precision() < 0))
    {
        apply<void>::type_access<operation::set_precision, category_memory>(
            memory_precision());
    }

    if(!(timing_precision() < 0))
    {
        apply<void>::type_access<operation::set_precision, category_timing>(
            timing_precision());
    }

    if(memory_units().length() > 0)
    {
        auto _memory_units = get_memory_unit(memory_units());
        apply<void>::type_access<operation::set_units, has_memory_units>(_memory_units);
    }

    if(timing_units().length() > 0)
    {
        auto _timing_units = get_timing_unit(timing_units());
        apply<void>::type_access<operation::set_units, has_timing_units>(_timing_units);
    }
}

//--------------------------------------------------------------------------------------//

inline tim::settings::string_t
tim::settings::get_output_prefix()
{
    auto dir = output_path();
    auto ret = makedir(dir);
    return (ret == 0) ? path_t(dir + string_t("/") + output_prefix())
                      : path_t(string_t("./") + output_prefix());
}

//--------------------------------------------------------------------------------------//

inline tim::settings::string_t
tim::settings::compose_output_filename(const std::string& _tag, std::string _ext)
{
    auto _prefix      = get_output_prefix();
    auto _rank_suffix = (!mpi::is_initialized())
                            ? std::string("")
                            : (std::string("_") + std::to_string(mpi::rank()));
    if(_ext.find('.') != 0)
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

inline void
tim::timemory_init(int argc, char** argv, const std::string& _prefix,
                   const std::string& _suffix)
{
    consume_parameters(argc);
    std::string exe_name = argv[0];

    while(exe_name.find("\\") != std::string::npos)
        exe_name = exe_name.substr(exe_name.find_last_of('\\') + 1);

    while(exe_name.find("/") != std::string::npos)
        exe_name = exe_name.substr(exe_name.find_last_of('/') + 1);

    static const std::vector<std::string> _exe_suffixes = { ".py", ".exe" };
    for(const auto& ext : _exe_suffixes)
    {
        if(exe_name.find(ext) != std::string::npos)
            exe_name.erase(exe_name.find(ext), ext.length() + 1);
    }

    gperf::profiler_start(exe_name);

    exe_name = _prefix + exe_name + _suffix;
    for(auto& itr : exe_name)
    {
        if(itr == '_')
            itr = '-';
    }

    tim::settings::output_path() = exe_name;
    // allow environment overrides
    tim::settings::parse();
}

//--------------------------------------------------------------------------------------//

inline void
tim::timemory_init(const std::string& exe_name, const std::string& _prefix,
                   const std::string& _suffix)
{
    auto cstr = const_cast<char*>(exe_name.c_str());
    tim::timemory_init(1, &cstr, _prefix, _suffix);
}

//--------------------------------------------------------------------------------------//

inline void
tim::timemory_finalize()
{
#if defined(__INTEL_COMPILER) || defined(__PGI__)
    tim::settings::auto_output() = false;
#endif
#if defined(__INTEL_COMPILER)
    apply<void>::type_access<operation::print_storage, complete_tuple_t>();
#endif
}
//--------------------------------------------------------------------------------------//

#include "timemory/details/storage.hpp"

//--------------------------------------------------------------------------------------//
