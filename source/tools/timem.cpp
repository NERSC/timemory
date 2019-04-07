// C program to illustrate  use of fork() &
// exec() system call for process creation

#include "timemory/macros.hpp"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(_UNIX)
#    include <unistd.h>
#endif

#include "timemory/auto_tuple.hpp"
#include "timemory/manager.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include <cstdio>
#include <cstring>

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
    static std::string _instance = "";
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
    comp_tuple_t measure("total execution time", command().c_str());
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
                printf("execv failed\n");
            else
                printf("program terminated with a non-zero status\n");
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
        std::cout << "\n" << measure << std::endl;
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
    if(argc > 1)
        command() = std::string(const_cast<const char*>(argv[1]));
    else
    {
        command()              = std::string(const_cast<const char*>(argv[0]));
        tim::get_rusage_type() = RUSAGE_CHILDREN;
        comp_tuple_t measure("total execution time", command().c_str());
        measure.start();
        measure.stop();
        std::cout << "\n" << measure << std::endl;
        exit(EXIT_SUCCESS);
    }

    pid_t pid = fork();

    uintmax_t nargs = static_cast<uintmax_t>(argc);
    if(pid == -1)  // pid == -1 means error occured
        failed_fork();
    else if(pid == 0)  // pid == 0 means child process created
        child_process(nargs, argv);
    else
        parent_process(pid);  // means parent process
}
