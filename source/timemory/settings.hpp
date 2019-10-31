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

#include "timemory/bits/settings.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace settings
{
inline void
parse();
}  // namespace settings
}  // namespace tim

//======================================================================================//

#include "timemory/backends/mpi.hpp"
#include "timemory/bits/timemory.hpp"
#include "timemory/components.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/utility.hpp"

namespace tim
{
namespace settings
{
/// initialize the storage of the specified types
template <typename _Tuple = available_tuple<tim::complete_tuple_t>>
void
initialize_storage();
}  // namespace settings

/// initialization (creates manager and configures output path)
void
timemory_init(int argc, char** argv, const std::string& _prefix = "timemory-",
              const std::string& _suffix = "-output");
/// initialization (creates manager and configures output path)
void
timemory_init(const std::string& exe_name, const std::string& _prefix = "timemory-",
              const std::string& _suffix = "-output");
/// initialization (creates manager, configures output path, mpi_init)
void
timemory_init(int* argc, char*** argv, const std::string& _prefix = "timemory-",
              const std::string& _suffix = "-output");
/// finalization of the specified types
void
timemory_finalize();

}  // namespace tim

//--------------------------------------------------------------------------------------//
// function to parse the environment for settings
//
// Nearly all variables will parse env when first access but this allows provides a
// way to reparse the environment so that default settings (possibly from previous
// invocation) can be overwritten
//
inline void
tim::settings::parse()
{
    if(suppress_parsing())
        return;

    // logic
    enabled()     = tim::get_env("TIMEMORY_ENABLED", enabled());
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

    exe_name = _prefix + exe_name + _suffix;
    for(auto& itr : exe_name)
    {
        if(itr == '_')
            itr = '-';
    }

    tim::settings::output_path() = exe_name;
    // allow environment overrides
    tim::settings::parse();

    if(tim::settings::enable_signal_handler())
    {
        auto default_signals = tim::signal_settings::get_default();
        for(auto& itr : default_signals)
            tim::signal_settings::enable(itr);
        // should return default and any modifications from environment
        auto enabled_signals = tim::signal_settings::get_enabled();
        tim::enable_signal_detection(enabled_signals);
    }
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
tim::timemory_init(int* argc, char*** argv, const std::string& _prefix,
                   const std::string& _suffix)
{
#if defined(TIMEMORY_USE_MPI)
    tim::mpi::initialize(argc, argv);
#endif
    timemory_init(*argc, *argv, _prefix, _suffix);
}

//--------------------------------------------------------------------------------------//

#include "timemory/manager.hpp"

//--------------------------------------------------------------------------------------//

inline void
tim::timemory_finalize()
{
    tim::manager::instance()->finalize();
    tim::disable_signal_detection();
}

//--------------------------------------------------------------------------------------//

#include "timemory/utility/bits/storage.hpp"

//--------------------------------------------------------------------------------------//

#define TIMEMORY_INIT(...) ::tim::timemory_init(__VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define TIMEMORY_FINALIZE() ::tim::timemory_finalize()

//--------------------------------------------------------------------------------------//

#if !defined(__library_ctor__)
#    if !defined(_WIN32) && !defined(_WIN64)
#        define __library_ctor__ __attribute__((constructor))
#    else
#        define __library_ctor__
#    endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(__library_dtor__)
#    if !defined(_WIN32) && !defined(_WIN64)
#        define __library_dtor__ __attribute__((destructor))
#    else
#        define __library_dtor__
#    endif
#endif

//--------------------------------------------------------------------------------------//
