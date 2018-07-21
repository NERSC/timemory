// C program to illustrate  use of fork() &
// exec() system call for process creation

#include "timemory/macros.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/types.h>

#if defined(_UNIX)
#   include <unistd.h>
#endif

#include "timemory/manager.hpp"
#include "timemory/rss.hpp"
#include "timemory/timer.hpp"

#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <thread>

#include <cstdio>
#include <cstring>

typedef std::vector<uint64_t> vector_t;
typedef tim::rss::usage rss_usage_t;

//----------------------------------------------------------------------------//

rss_usage_t& rss_init()
{
    static std::shared_ptr<rss_usage_t> _instance(nullptr);
    if(!_instance.get())
    {
        _instance.reset(new rss_usage_t);
    }
    return *(_instance.get());
}

//----------------------------------------------------------------------------//

std::string& tim_format()
{
    static std::string _instance
            = ": %w wall, %u user + %s system = %t cpu (%p%) [%T], %M peak rss [%A]";
    return _instance;
}

//----------------------------------------------------------------------------//

std::string& command()
{
    static std::string _instance = "";
    return _instance;
}

//----------------------------------------------------------------------------//

void failed_fork()
{
    printf("failure forking, error occured\n");
    exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------//

void report()
{
    tim::manager::instance()->stop_total_timer();
    std::stringstream _ss_report;

    (*tim::manager::instance()) -= rss_init();
    _ss_report << (*tim::manager::instance());
    std::string _report = _ss_report.str();
    if(command().length() > 0)
        _report.replace(_report.find("[exe]")+1, 3, command().c_str());

    std::cout << "\n" << _report << std::endl;

    exit(0);
}

//----------------------------------------------------------------------------//

void parent_process(pid_t pid)
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

    if (waitpid(pid, &status, 0) > 0)
    {
        if (WIFEXITED(status) && !WEXITSTATUS(status))
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
        report();

    exit(ret);
}

//----------------------------------------------------------------------------//

void print_command(int argc, char** argv)
{
    for(int i = 0; i < argc; ++i)
        printf("%s ", argv[i]);
    printf("\n");
}

//----------------------------------------------------------------------------//

char* getcharptr(const std::string& str)
{
    return const_cast<char*>(str.c_str());
}

//----------------------------------------------------------------------------//

void child_process(int argc, char** argv)
{
    if(argc < 2)
        exit(0);

    // the argv list first argument should point to filename associated
    // with file being executed the array pointer must be terminated by
    // NULL pointer

    char** argv_list = (char**) malloc(sizeof(char*) * argc);
    for(int i = 0; i < argc - 1; i++)
        argv_list[i] = argv[i+1];
    argv_list[argc-1] = NULL;

    // launch the child
    int ret = execvp(argv_list[0], argv_list);
    if(ret < 0)
    {
        int argc_shell = argc + 2;
        char** argv_shell_list = (char**) malloc(sizeof(char*) * argc_shell);
        char* _shell = getusershell();
        if(_shell)
        {
            argv_shell_list[0] = _shell;
            argv_shell_list[1] = getcharptr("-lc");
            for(int i = 0; i < argc-1; ++i)
                argv_shell_list[i+2] = argv_list[i];
            argv_shell_list[argc_shell-1] = NULL;
            ret = execvp(argv_shell_list[0], argv_shell_list);
        }
    }

    exit(0);
}

//----------------------------------------------------------------------------//

int main(int argc, char** argv)
{
    tim::format::timer::set_default_format(tim_format());
    tim::manager::instance()->update_total_timer_format();
    rss_init().record();

    if(argc > 1)
        command() = std::string(const_cast<const char*>(argv[1]));
    else
    {
        tim::manager::instance()->reset_total_timer();
        report();
    }

    pid_t pid = fork();
    tim::manager::instance()->reset_total_timer();

    if(pid == -1) // pid == -1 means error occured
        failed_fork();
    else if(pid == 0) // pid == 0 means child process created
        child_process(argc, argv);
    else
        parent_process(pid); // means parent process

    return 0;
}
