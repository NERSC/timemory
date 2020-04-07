// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * \file timemory/config/definition.hpp
 * \brief The definitions for the types in config
 */

#pragma once

#include "timemory/config/declaration.hpp"
#include "timemory/config/macros.hpp"
#include "timemory/config/types.hpp"
//
#if defined(TIMEMORY_CONFIG_SOURCE) ||                                                   \
    (!defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_CONFIG_EXTERN))
//
#    include "timemory/backends/process.hpp"
#    include "timemory/manager/declaration.hpp"
#    include "timemory/mpl/filters.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/utility/signals.hpp"
#    include "timemory/utility/utility.hpp"
//
#    include <string>
//
#endif

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              config
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_CONFIG_SOURCE) ||                                                   \
    (!defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_CONFIG_EXTERN))
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(int argc, char** argv, const std::string& _prefix,
              const std::string& _suffix)
{
    if(settings::debug() || settings::verbose() > 3)
        PRINT_HERE("%s", "");

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

    settings::output_path() = exe_name;
    // allow environment overrides
    settings::parse();

    if(settings::enable_signal_handler())
    {
        auto default_signals = signal_settings::get_default();
        for(auto& itr : default_signals)
            signal_settings::enable(itr);
        // should return default and any modifications from environment
        auto enabled_signals = signal_settings::get_enabled();
        enable_signal_detection(enabled_signals);
    }

    settings::store_command_line(argc, argv);

    auto _manager = manager::instance();
    consume_parameters(_manager);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(const std::string& exe_name, const std::string& _prefix,
              const std::string& _suffix)
{
    auto cstr  = const_cast<char*>(exe_name.c_str());
    auto _argc = 1;
    auto _argv = &cstr;
    timemory_init(_argc, _argv, _prefix, _suffix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(int* argc, char*** argv, const std::string& _prefix,
              const std::string& _suffix)
{
    if(settings::mpi_init())
    {
        if(settings::debug())
            PRINT_HERE("%s", "initializing mpi");

        mpi::initialize(argc, argv);
    }

    if(settings::upcxx_init())
    {
        if(settings::debug())
            PRINT_HERE("%s", "initializing upcxx");

        upc::initialize();
    }

    timemory_init(*argc, *argv, _prefix, _suffix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_finalize()
{
    if(settings::debug() || settings::verbose() > 3)
        PRINT_HERE("%s", "");

    if(settings::enable_signal_handler())
    {
        if(settings::debug())
            PRINT_HERE("%s", "disabling signal detection");
        disable_signal_detection();
    }

    if(settings::debug())
        PRINT_HERE("%s", "finalizing manager");

    auto _manager = manager::instance();
    if(_manager)
        _manager->finalize();

    if(settings::upcxx_finalize())
    {
        if(settings::debug())
            PRINT_HERE("%s", "finalizing upcxx");

        upc::finalize();
    }

    if(settings::mpi_finalize())
    {
        if(settings::debug())
            PRINT_HERE("%s", "finalizing mpi");
        mpi::finalize();
    }

    if(settings::debug())
        PRINT_HERE("%s", "done");
}
//
//--------------------------------------------------------------------------------------//
//
#endif  // SOURCE or (NOT USE_EXTERN AND NOT USE_CONFIG_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
