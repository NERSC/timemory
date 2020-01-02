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
extern "C"
{
    extern char** environ;
}
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

template <typename _Tp>
using vector_t = std::vector<_Tp>;
using string_t = std::string;

template <typename _Tp>
using vector_t = std::vector<_Tp>;
using string_t = std::string;

using namespace tim::component;

//--------------------------------------------------------------------------------------//
// create a custom component tuple printer
//
namespace tim
{
//--------------------------------------------------------------------------------------//
//
template <typename _Tp>
struct custom_print
{
    using value_type = typename _Tp::value_type;
    using base_type  = component::base<_Tp, value_type>;

    custom_print(std::size_t _N, std::size_t /*_Ntot*/, base_type& obj, std::ostream& os,
                 bool /*endline*/)
    {
        std::stringstream ss;
        if(_N == 0)
            ss << std::endl;
        ss << "    " << obj << std::endl;
        os << ss.str();
    }
};

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
class custom_component_tuple : public component_tuple<Types...>
{
    using apply_stop_t  = modifiers<operation::stop, Types...>;
    using apply_print_t = modifiers<custom_print, Types...>;

    static std::string label();

public:
    explicit custom_component_tuple(const string_t& key)
    : component_tuple<Types...>(key, true, true)
    {}

    //----------------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&                           os,
                                    const custom_component_tuple<Types...>& obj)
    {
        std::stringstream ssp;
        std::stringstream ssd;
        auto&&            data  = obj.m_data;
        auto&&            key   = obj.key();
        auto&&            width = obj.output_width();

        apply<void>::access<apply_stop_t>(data);
        apply<void>::access_with_indices<apply_print_t>(data, std::ref(ssd), false);

        ssp << std::setw(width) << std::left << key;
        os << ssp.str() << ssd.str();

        return os;
    }
};

}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//
using comp_tuple_t = tim::custom_component_tuple<
    real_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss, num_io_in,
    num_io_out, num_minor_page_faults, num_major_page_faults, num_signals,
    voluntary_context_switch, priority_context_switch, read_bytes, written_bytes>;

//--------------------------------------------------------------------------------------//

comp_tuple_t*&
get_measure()
{
    static comp_tuple_t* _instance = nullptr;
    return _instance;
}

//--------------------------------------------------------------------------------------//

bool
use_shell()
{
    static bool _instance = tim::get_env("TIMEM_USE_SHELL", true);
    return _instance;
}

//--------------------------------------------------------------------------------------//

bool
debug()
{
    static bool _instance = tim::get_env("TIMEM_DEBUG", false);
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

template <typename... Types>
std::string
tim::custom_component_tuple<Types...>::label()
{
    return command();
}

//--------------------------------------------------------------------------------------//

declare_attribute(noreturn) void failed_fork()
{
    printf("failure forking, error occured\n");
    exit(EXIT_FAILURE);
}

//--------------------------------------------------------------------------------------//

void
parent_process(pid_t pid)
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
    int ret = 0;

    if(waitpid(pid, &status, 0) > 0)
    {
        get_measure()->stop();

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
                printf("program terminated with a non-zero status: %i\n", ret);
            }
        }
        else
        {
            printf("program terminated abnormally.\n");
            ret = -1;
        }
    }
    else
    {
        printf("waitpid() failed\n");
    }

    std::stringstream _oss;
    _oss << "\n" << *get_measure() << std::flush;

    if(tim::settings::file_output())
    {
        std::string label = "timem";
        if(tim::settings::text_output())
        {
            auto          fname = tim::settings::compose_output_filename(label, ".txt");
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
            tim::generic_serialization(jname, *get_measure());
        }
    }
    else
    {
        std::cout << _oss.str() << std::endl;
    }

    exit(ret);
}

//--------------------------------------------------------------------------------------//

void
explain(int ret, const char* pathname, char** argv)
{
    if(ret < 0)
    {
#if defined(TIMEMORY_USE_LIBEXPLAIN)
        fprintf(stderr, "%s\n", explain_execvp(pathname, argv));
#else
        fprintf(stderr, "Return code: %i : %s\n", ret, pathname);
        int n = 0;
        std::cerr << "Command: ";
        while(argv[n] != nullptr)
            std::cerr << argv[n++] << " ";
        std::cerr << std::endl;
#endif
    }
    else if(debug())
    {
        int n = 0;
        std::cerr << "Command: ";
        while(argv[n] != nullptr)
            std::cerr << argv[n++] << " ";
        std::cerr << std::endl;
    }
}

//--------------------------------------------------------------------------------------//

declare_attribute(noreturn) void child_process(uint64_t argc, char** argv)
{
    if(argc < 2)
        exit(0);

    // the argv list first argument should point to filename associated
    // with file being executed the array pointer must be terminated by
    // NULL pointer

    std::stringstream shell_cmd;
    char**            argv_list = (char**) malloc(sizeof(char**) * (argc));
    for(uint64_t i = 0; i < argc - 1; i++)
    {
        argv_list[i] = strdup(argv[i + 1]);
        shell_cmd << argv[i + 1] << " ";
    }
    argv_list[argc - 1] = nullptr;

    // launches the command with the shell, this is the default because it
    // enables aliases
    auto launch_using_shell = [&]() {
        int         ret             = -1;
        uint64_t    argc_shell      = 5;
        char**      argv_shell_list = (char**) malloc(sizeof(char**) * (argc_shell));
        std::string _shell          = tim::get_env<std::string>("SHELL", getusershell());

        if(debug())
            printf("using shell: %s\n", _shell.c_str());

        argv_shell_list[argc_shell - 1] = nullptr;
        if(_shell.length() > 0)
        {
            if(debug())
                PRINT_HERE("%s", "");

            std::string _interactive = "-i";
            std::string _command     = "-c";
            argv_shell_list[0]       = strdup(_shell.c_str());
            argv_shell_list[1]       = strdup(_interactive.c_str());
            argv_shell_list[2]       = strdup(_command.c_str());
            argv_shell_list[3]       = strdup(shell_cmd.str().c_str());
            argv_shell_list[4]       = nullptr;

            if(debug())
                PRINT_HERE("%s", "");

            explain(0, argv_shell_list[0], argv_shell_list);
            ret = execvp(argv_shell_list[0], argv_shell_list);
            explain(ret, argv_shell_list[0], argv_shell_list);

            if(ret != 0)
            {
                PRINT_HERE("return code: %i", ret);
                explain(ret, argv_shell_list[0], argv_shell_list);
                ret = execv(argv_shell_list[0], argv_shell_list);
            }

            if(ret != 0)
            {
                PRINT_HERE("return code: %i", ret);
                explain(ret, argv_shell_list[0], argv_shell_list);
                ret = execve(argv_shell_list[0], argv_shell_list, environ);
            }

            if(debug())
                PRINT_HERE("return code: %i", ret);
        }
        else
        {
            fprintf(stderr, "getusershell failed!\n");
        }
        if(debug())
            PRINT_HERE("%s", "");
        // delete[] argv_shell_list;
        return ret;
    };

    // this will launch the process and inherit the environment but aliases will not
    // be available
    auto launch_without_shell = [&]() {
        int ret = execvp(argv_list[0], argv_list);
        // explain error if enabled
        explain(ret, argv_list[0], argv_list);
        return ret;
    };

    // default return code
    int ret = -1;

    // determine if the shell should be tested first
    bool try_shell = use_shell();

    if(try_shell)
    {
        // launch the command with shell. If that fails, launch without shell
        ret = launch_using_shell();
        if(ret < 0)
        {
            if(debug())
                puts("Error launching with shell! Trying without shell...");
            ret = launch_without_shell();
        }
    }
    else
    {
        // launch the command without shell. If that fails, launch with shell
        ret = launch_without_shell();
        if(ret < 0)
        {
            if(debug())
                puts("Error launching without shell! Trying with shell...");
            ret = launch_using_shell();
        }
    }

    explain(ret, argv_list[0], argv_list);

    exit(ret);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    // disable banner if not specified
    setenv("TIMEMORY_BANNER", "OFF", 0);

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

    auto compose_prefix = [&]() {
        std::stringstream ss;
        ss << "[" << command().c_str() << "] measurement totals:";
        return ss.str();
    };

    if(argc > 1)
    {
        command() = std::string(const_cast<const char*>(argv[1]));
    }
    else
    {
        command()              = std::string(const_cast<const char*>(argv[0]));
        tim::get_rusage_type() = RUSAGE_CHILDREN;
        get_measure()          = new comp_tuple_t(compose_prefix());
        get_measure()->start();
        get_measure()->stop();
        std::cout << "\n" << *get_measure() << std::flush;
        exit(EXIT_SUCCESS);
    }

    tim::get_rusage_type() = RUSAGE_CHILDREN;
    get_measure()          = new comp_tuple_t(compose_prefix());

    get_measure()->start();

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
    else  // means parent process
    {
        parent_process(pid);
    }

    delete get_measure();
}
