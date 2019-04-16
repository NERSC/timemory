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

#include "timemory/auto_tuple.hpp"
#include "timemory/environment.hpp"
#include "timemory/macros.hpp"
#include "timemory/manager.hpp"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(_UNIX)
#    include <unistd.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using vector_t = std::vector<uintmax_t>;
using namespace tim::component;
using comp_tuple_t = tim::details::custom_component_tuple<
    real_clock, system_clock, cpu_clock, cpu_util, peak_rss, data_rss, stack_rss,
    num_minor_page_faults, num_major_page_faults, voluntary_context_switch,
    priority_context_switch>;

#if defined(__GNUC__) || defined(__clang__)
#    define declare_attribute(attr) __attribute__((attr))
#elif defined(_WIN32)
#    define declare_attribute(attr) __declspec(attr)
#endif

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
    comp_tuple_t measure("total execution time", command());
    if(getpid() != getppid() + 1)
        measure.start();

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
        _oss << "\n" << measure << std::endl;

        if(tim::env::file_output())
        {
            std::string label = "timem";
            if(tim::env::text_output())
            {
                auto          fname = tim::env::compose_output_filename(label, ".txt");
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

            if(tim::env::json_output())
            {
                auto jname = tim::env::compose_output_filename(label, ".json");
                printf("[timem]> Outputting '%s'...\n", jname.c_str());
                serialize_storage(jname, label, measure);
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

declare_attribute(noreturn) void child_process(uintmax_t argc, char** argv)
{
    if(argc < 2)
        exit(0);

    // the argv list first argument should point to filename associated
    // with file being executed the array pointer must be terminated by
    // NULL pointer

    char** argv_list = static_cast<char**>(malloc(sizeof(char*) * argc));
    for(uintmax_t i = 0; i < argc - 1; i++)
        argv_list[i] = argv[i + 1];
    argv_list[argc - 1] = nullptr;

    // launch the child
    int ret = execvp(argv_list[0], argv_list);
    if(ret < 0)
    {
        uintmax_t argc_shell   = argc + 2;
        char** argv_shell_list = static_cast<char**>(malloc(sizeof(char*) * argc_shell));
        char*  _shell          = getusershell();
        if(_shell)
        {
            argv_shell_list[0] = _shell;
            argv_shell_list[1] = getcharptr("-c");
            for(uintmax_t i = 0; i < argc - 1; ++i)
                argv_shell_list[i + 2] = argv_list[i];
            argv_shell_list[argc_shell - 1] = nullptr;
            ret                             = execvp(argv_shell_list[0], argv_shell_list);
        }
    }

    exit(0);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    // set some defaults
    tim::env::file_output() = false;
    tim::env::scientific()  = true;
    tim::env::width()       = 12;
    tim::env::precision()   = 3;

    // parse for settings configurations
    tim::env::parse();

    // override a some settings
    tim::env::suppress_parsing() = true;
    tim::env::auto_output()      = false;
    tim::env::output_prefix()    = "";

    // update values to reflect modifications
    tim::env::process();

    if(argc > 1)
    {
        command() = std::string(const_cast<const char*>(argv[1]));
    }
    else
    {
        command()              = std::string(const_cast<const char*>(argv[0]));
        tim::get_rusage_type() = RUSAGE_CHILDREN;
        comp_tuple_t measure("total execution time", command());
        measure.start();
        measure.stop();
        std::cout << "\n" << measure << std::endl;
        exit(EXIT_SUCCESS);
    }

    pid_t pid = fork();

    uintmax_t nargs = static_cast<uintmax_t>(argc);
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
