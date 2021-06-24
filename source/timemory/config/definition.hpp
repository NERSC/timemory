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
#include "timemory/utility/argparse.hpp"

#if defined(TIMEMORY_CONFIG_SOURCE) || !defined(TIMEMORY_USE_CONFIG_EXTERN)
//
#    include "timemory/backends/process.hpp"
#    include "timemory/manager/declaration.hpp"
#    include "timemory/mpl/filters.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/utility/signals.hpp"
#    include "timemory/utility/utility.hpp"

#    include <fstream>
#    include <string>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              config
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(int argc, char** argv, const std::string& _prefix,
              const std::string& _suffix)
{
    static auto _settings = settings::shared_instance();
    if(_settings)
    {
        if(_settings->get_debug() || _settings->get_verbose() > 3)
            PRINT_HERE("%s", "");

        if(_settings->get_enable_signal_handler())
        {
            auto default_signals = signal_settings::get_default();
            for(const auto& itr : default_signals)
                signal_settings::enable(itr);
            // should return default and any modifications from environment
            auto enabled_signals = signal_settings::get_enabled();
            enable_signal_detection(enabled_signals);
        }

        auto _ncfg = _settings->get_suppress_config();
        if(!_ncfg)
        {
            // default configuration files
            std::set<std::string> _dcfg = {
                get_env<string_t>("HOME") + std::string("/.timemory.cfg"),
                get_env<string_t>("HOME") + std::string("/.timemory.json"),
                get_env<string_t>("HOME") + std::string("/.config/timemory.cfg"),
                get_env<string_t>("HOME") + std::string("/.config/timemory.json")
            };

            auto _cfg   = _settings->get_config_file();
            auto _files = tim::delimit(_cfg, ",;");
            for(const auto& citr : _files)
            {
                std::ifstream ifs(citr);
                if(ifs)
                {
                    _settings->read(ifs, citr);
                }
                else if(_dcfg.find(citr) == _dcfg.end())
                {
                    TIMEMORY_EXCEPTION(std::string("Error reading configuration file: ") +
                                       citr);
                }
            }
        }
    }

    std::string exe_name = (argc > 0) ? argv[0] : "";

    while(exe_name.find('\\') != std::string::npos)
        exe_name = exe_name.substr(exe_name.find_last_of('\\') + 1);

    while(exe_name.find('/') != std::string::npos)
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

    size_t pos = std::string::npos;
    while((pos = exe_name.find("--")) != std::string::npos)
        exe_name.erase(pos, 1);

    if(exe_name.empty())
        exe_name = "timemory-output";

    if(_settings)
    {
        _settings->get_output_path() = exe_name;

        // allow environment overrides
        settings::parse(_settings);

        if(_settings->get_enable_signal_handler())
        {
            auto _exit_action = [](int nsig) {
                auto _manager = manager::master_instance();
                if(_manager)
                {
                    std::cout << "Finalizing after signal: " << nsig << " :: "
                              << signal_settings::str(static_cast<sys_signal>(nsig))
                              << std::endl;
                    _manager->finalize();
                }
            };
            signal_settings::set_exit_action(_exit_action);
        }

        settings::store_command_line(argc, argv);
    }

    static auto _manager = manager::instance();
    if(_manager)
    {
        _manager->update_metadata_prefix();
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(const std::string& exe_name, const std::string& _prefix,
              const std::string& _suffix)
{
    auto* cstr  = const_cast<char*>(exe_name.c_str());
    auto  _argc = 1;
    auto* _argv = &cstr;
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
    timemory_argparse(argc, argv);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(int* argc, char*** argv, argparse::argument_parser& parser,
              const std::string& _prefix, const std::string& _suffix)
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
    timemory_argparse(argc, argv, &parser);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_init(std::vector<std::string>& args, argparse::argument_parser& parser,
              const std::string& _prefix, const std::string& _suffix)
{
    int    argc = args.size();
    char** argv = new char*[argc];
    for(int i = 0; i < argc; ++i)
    {
        auto len = args.at(i).length();
        argv[i]  = new char[len + 1];
        strcpy(argv[i], args.at(i).c_str());
        argv[i][len] = '\0';
    }

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

    timemory_init(argc, argv, _prefix, _suffix);
    timemory_argparse(args, &parser);

    for(int i = 0; i < argc; ++i)
        delete[] argv[i];
    delete[] argv;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_argparse(int* argc, char*** argv, argparse::argument_parser* parser,
                  tim::settings* _settings)
{
    // if pointers are null, return
    if(!argc || !argv)
        return;

    // if only one argument, return
    if(*argc < 2)
        return;

    using parser_t     = argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    // print help
    auto help_action = [](parser_t& p) {
        if(dmp::rank() == 0)
            p.print_help("-- <NON_TIMEMORY_ARGS>");
        exit(EXIT_SUCCESS);
    };

    auto err_action = [=](const parser_err_t& _err) {
        if(dmp::rank() == 0 && _err)
            std::cerr << "[timemory_argparse]> Error! " << _err << std::endl;
    };

    // if argument parser was not provided
    bool _cleanup_parser = parser == nullptr;
    if(_cleanup_parser)
        parser = new parser_t{ (*argv)[0] };

    // if settings instance was not provided
    bool        _cleanup_settings = _settings == nullptr;
    static auto _shared_settings  = tim::settings::shared_instance();
    if(_cleanup_settings && !_shared_settings)
        return;
    if(_cleanup_settings)
        _settings = _shared_settings.get();

    // enable help
    parser->enable_help();
    parser->on_error([=](parser_t& p, const parser_err_t& _err) {
        err_action(_err);
        if(dmp::rank() == 0 && _settings->get_verbose() > 0)
            p.print_help("-- <NON_TIMEMORY_ARGS>");
    });

    // add the arguments in the order they were appended to the settings
    for(const auto& itr : _settings->ordering())
    {
        auto sitr = _settings->find(itr);
        if(sitr != _settings->end() && sitr->second)
        {
            sitr->second->add_argument(*parser);
        }
    }

    parser->add_argument()
        .names({ "--timemory-args" })
        .description("A generic option for any setting. Each argument MUST be passed in "
                     "form: 'NAME=VALUE'. E.g. --timemory-args "
                     "\"papi_events=PAPI_TOT_INS,PAPI_TOT_CYC\" text_output=off")
        .action([&](parser_t& p) {
            // get the options
            auto vopt = p.get<std::vector<std::string>>("timemory-args");
            for(auto& str : vopt)
            {
                // get the args
                auto vec = tim::delimit(str, " \t;:");
                for(const auto& itr : vec)
                {
                    DEBUG_PRINT_HERE("Processing: %s", itr.c_str());
                    auto _pos = itr.find('=');
                    auto _key = itr.substr(0, _pos);
                    auto _val = (_pos == std::string::npos) ? "" : itr.substr(_pos + 1);
                    if(!_settings->update(_key, _val, false))
                    {
                        std::cerr << "[timemory_argparse]> Warning! For "
                                     "--timemory-args, key \""
                                  << _key << "\" is not a recognized setting. \"" << _val
                                  << "\" was not applied." << std::endl;
                    }
                }
            }
        });

    err_action(parser->parse_known_args(argc, argv, "--", settings::verbose()));
    if(parser->exists("help"))
        help_action(*parser);

    // cleanup if argparse was not provided
    if(_cleanup_parser)
        delete parser;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)
timemory_argparse(std::vector<std::string>& args, argparse::argument_parser* parser,
                  tim::settings* _settings)
{
    int    argc = args.size();
    char** argv = new char*[argc];
    for(int i = 0; i < argc; ++i)
    {
        // intentional memory leak
        auto len = args.at(i).length();
        argv[i]  = new char[len + 1];
        strcpy(argv[i], args.at(i).c_str());
        argv[i][len] = '\0';
    }

    timemory_argparse(&argc, &argv, parser, _settings);

    // updates args
    args.clear();
    for(int i = 0; i < argc; ++i)
        args.emplace_back(argv[i]);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CONFIG_LINKAGE(void)  // NOLINT
timemory_finalize()
{
    auto* _settings = settings::instance();
    auto  _manager  = manager::instance();

    if(_manager)
    {
        if(_settings && _settings->get_debug())
            PRINT_HERE("%s", "finalizing manager");
        _manager->finalize();
    }

    if(_settings)
    {
        if(_settings->get_upcxx_finalize())
        {
            if(_settings->get_debug())
                PRINT_HERE("%s", "finalizing upcxx");

            upc::finalize();
        }

        if(_settings->get_mpi_finalize())
        {
            if(_settings->get_debug())
                PRINT_HERE("%s", "finalizing mpi");
            mpi::finalize();
        }

        if(_settings->get_debug() || _settings->get_verbose() > 3)
            PRINT_HERE("%s", "");

        if(_settings->get_enable_signal_handler())
        {
            if(_settings->get_debug())
                PRINT_HERE("%s", "disabling signal detection");
            disable_signal_detection();
        }

        if(_settings->get_debug())
            PRINT_HERE("%s", "done");
    }
}
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#endif  // defined(TIMEMORY_CONFIG_SOURCE) || !defined(TIMEMORY_USE_CONFIG_EXTERN)
