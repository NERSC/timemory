// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include "timemory/timemory.hpp"

// C includes
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(TIMEMORY_USE_LIBEXPLAIN)
#    include <libexplain/execvp.h>
#endif

#if defined(_UNIX)
#    include <unistd.h>
#endif

// C++ includes
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

//--------------------------------------------------------------------------------------//

#if defined(__GNUC__) || defined(__clang__)
#    define declare_attribute(attr) __attribute__((attr))
#elif defined(_WIN32)
#    define declare_attribute(attr) __declspec(attr)
#endif

#if !defined(PAPI_NUM_COUNTERS)
#    define PAPI_NUM_COUNTERS 32
#endif

template <typename _Tp>
using vector_t = std::vector<_Tp>;
using string_t = std::string;

using namespace tim::component;

//--------------------------------------------------------------------------------------//
// papi event set 0 with PAPI_NUM_COUNTERS (4) counters
//
using papi_array_t = papi_array<PAPI_NUM_COUNTERS>;

//--------------------------------------------------------------------------------------//
//
//
using comp_tuple_t =
    tim::details::custom_component_tuple<real_clock, user_clock, system_clock, cpu_clock,
                                         cpu_util, peak_rss, num_minor_page_faults,
                                         num_major_page_faults, voluntary_context_switch,
                                         priority_context_switch>;

//--------------------------------------------------------------------------------------//

bool&
papi_enabled()
{
    static bool _instance = tim::get_env("TIMEM_PAPI", false);
    return _instance;
}

//--------------------------------------------------------------------------------------//

std::string&
command()
{
    static std::string _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

declare_attribute(noreturn) void failed_fork()
{
    printf("failure forking, error occured\n");
    exit(EXIT_FAILURE);
}

//--------------------------------------------------------------------------------------//

declare_attribute(noreturn) void parent_process(pid_t pid)
{
    // a positive number is returned for the pid of parent process
    // getppid() returns process id of parent of calling process

    // the parent process calls waitpid() on the child
    // waitpid() system call suspends execution of
    // calling process until a child specified by pid
    // argument has changed state
    // see wait() man page for all the flags or options
    // used here

    int status;
    int ret                = 0;
    tim::get_rusage_type() = RUSAGE_CHILDREN;
    comp_tuple_t measure("total execution time", tim::language(command().c_str()));
    if(getpid() != getppid() + 1)
    {
        measure.start();
    }

    papi_array_t* _papi_array = nullptr;
    if(papi_enabled())
    {
        tim::papi::init();
        papi_array_t::get_events_func() = [&]() {
            auto events_str = tim::get_env<string_t>("TIMEM_PAPI_EVENTS", "PAPI_LST_INS");
            vector_t<string_t> events_str_list = tim::delimit(events_str);
            vector_t<int>      events_list;
            for(const auto& itr : events_str_list)
                events_list.push_back(tim::papi::get_event_code(itr));
            return events_list;
        };
        papi_array_t::enable_multiplex() = tim::get_env("TIMEM_PAPI_MULTIPLEX", false);
        _papi_array                      = new papi_array_t();
        _papi_array->start();
    }

    if(waitpid(pid, &status, 0) > 0)
    {
        if(WIFEXITED(status) && !WEXITSTATUS(status))
        {
            ret = 0;
        }
        else if(WIFEXITED(status) && WEXITSTATUS(status))
        {
            ret = WEXITSTATUS(status);
            if(ret == 127)
            {
                printf("execv failed\n");
            }
            else
            {
                printf("program terminated with a non-zero status\n");
            }
        }
        else
        {
            printf("program terminated annormally\n");
            ret = -1;
        }
    }
    else
    {
        printf("waitpid() failed\n");
    }

    if(getpid() != getppid() + 1)
    {
        measure.stop();
        std::stringstream _oss;
        _oss << "\n" << measure << std::flush;

        if(_papi_array)
        {
            papi_array_t::get_label() = "";
            _papi_array->stop();
            _oss << "\n"
                 << tim::language(command().c_str())
                 << " hardware counters : " << (*_papi_array) << std::flush;
            delete _papi_array;
        }

        if(tim::settings::file_output())
        {
            std::string label = "timem";
            if(tim::settings::text_output())
            {
                auto fname = tim::settings::compose_output_filename(label, ".txt");
                std::ofstream ofs(fname.c_str());
                if(ofs)
                {
                    printf("[timem]> Outputting '%s'...\n", fname.c_str());
                    ofs << _oss.str();
                    ofs.close();
                }
                else
                {
                    std::cout << "[timem]>  opening output file '" << fname << "'...\n";
                    std::cout << _oss.str();
                }
            }

            if(tim::settings::json_output())
            {
                auto jname = tim::settings::compose_output_filename(label, ".json");
                printf("[timem]> Outputting '%s'...\n", jname.c_str());
                serialize_storage(jname, measure);
            }
        }
        else
        {
            std::cout << _oss.str();
        }
    }

    exit(ret);
}

//--------------------------------------------------------------------------------------//

char*
getcharptr(const std::string& str)
{
    return const_cast<char*>(str.c_str());
}

//--------------------------------------------------------------------------------------//

void
explain(int ret, const char* pathname, char** argv)
{
#if defined(TIMEMORY_USE_LIBEXPLAIN)
    if(ret < 0)
        fprintf(stderr, "%s\n", explain_execvp(pathname, argv));
#else
    tim::consume_parameters(ret, pathname, argv);
#endif
}
//--------------------------------------------------------------------------------------//

declare_attribute(noreturn) void child_process(uint64_t argc, char** argv)
{
    if(argc < 2)
        exit(0);

    // the argv list first argument should point to filename associated
    // with file being executed the array pointer must be terminated by
    // NULL pointer

    char** argv_list = static_cast<char**>(malloc(sizeof(char*) * argc));
    for(uint64_t i = 0; i < argc - 1; i++)
        argv_list[i] = argv[i + 1];
    argv_list[argc - 1] = nullptr;

    // launch the child
    int ret = execvp(argv_list[0], argv_list);
    if(ret < 0)
    {
        uint64_t argc_shell    = argc + 2;
        char** argv_shell_list = static_cast<char**>(malloc(sizeof(char*) * argc_shell));
        char*  _shell          = getusershell();
        if(_shell)
        {
            argv_shell_list[0] = _shell;
            argv_shell_list[1] = getcharptr("-c");
            for(uint64_t i = 0; i < argc - 1; ++i)
                argv_shell_list[i + 2] = argv_list[i];
            argv_shell_list[argc_shell - 1] = nullptr;
            ret                             = execvp(argv_shell_list[0], argv_shell_list);
            explain(ret, argv_shell_list[0], argv_shell_list);
        }
        else
        {
            fprintf(stderr, "getusershell failed!\n");
        }
    }

    explain(ret, argv_list[0], argv_list);

    exit(0);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    // set some defaults
    tim::settings::file_output() = false;
    tim::settings::scientific()  = false;
    tim::settings::width()       = 12;
    tim::settings::precision()   = 6;

    // parse for settings configurations
    if(argc > 1)
        tim::timemory_init(argc, argv);

    // override a some settings
    tim::settings::suppress_parsing() = true;
    tim::settings::auto_output()      = false;
    tim::settings::output_prefix()    = "";

    // update values to reflect modifications
    tim::settings::process();

    if(argc > 1)
    {
        command() = "[" + std::string(const_cast<const char*>(argv[1])) + "]";
    }
    else
    {
        command() = "[" + std::string(const_cast<const char*>(argv[0])) + "]";
        tim::get_rusage_type() = RUSAGE_CHILDREN;
        comp_tuple_t measure("total execution time", tim::language(command().c_str()));
        measure.start();
        measure.stop();
        std::cout << "\n" << measure << std::flush;
        exit(EXIT_SUCCESS);
    }

    pid_t pid = fork();

    uint64_t nargs = static_cast<uint64_t>(argc);
    if(pid == -1)  // pid == -1 means error occured
    {
        failed_fork();
    }
    else if(pid == 0)  // pid == 0 means child process created
    {
        child_process(nargs, argv);
    }
    else
    {
        parent_process(pid);  // means parent process
    }
}
