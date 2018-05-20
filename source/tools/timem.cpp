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

typedef std::vector<uint64_t> vector_t;

//----------------------------------------------------------------------------//

tim::rss::usage rss_init;
std::string tim_format =
        ": %w wall, %u user + %s system = %t cpu (%p%) [%T], %M peak rss [%A]";
std::string command = "";

//----------------------------------------------------------------------------//

void failed_fork()
{
    printf("can't fork, error occured\n");
    exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------//

void report()
{
    tim::manager::instance()->stop_total_timer();
    (*tim::manager::instance()) -= rss_init;

    std::stringstream _ss_report;
    _ss_report << (*tim::manager::instance());
    std::string _report = _ss_report.str();
    if(command.length() > 0)
        _report.replace(_report.find("[exe]")+1, 3, command.c_str());

    std::cout << "\n" << _report << std::endl;

    exit(0);
}

//----------------------------------------------------------------------------//

void parent_process(pid_t pid)
{
    int status;

    // a positive number is returned for the pid of parent process
    // getppid() returns process id of parent of calling process

    // the parent process calls waitpid() on the child
    // waitpid() system call suspends execution of
    // calling process until a child specified by pid
    // argument has changed state
    // see wait() man page for all the flags or options
    // used here

    if (waitpid(pid, &status, 0) > 0)
    {
        if (WIFEXITED(status) && !WEXITSTATUS(status))
        {
            if(getpid() != getppid() + 1)
                report();
        }
        else if (WIFEXITED(status) && WEXITSTATUS(status))
        {
            #if defined(DEBUG)
            if (WEXITSTATUS(status) == 127)
                printf("execv failed\n");
            else
                printf("program terminated normally,"
                       " but returned a non-zero status\n");
            #endif
            exit(WEXITSTATUS(status));
        }
        else
        {
            #if defined(DEBUG)
            printf("program didn't terminate normally\n");
            #endif
            exit(-1);
        }
    }
    else
        printf("waitpid() failed\n");

    exit(0);
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
    execvp(argv_list[0], argv_list);

    exit(0);
}

//----------------------------------------------------------------------------//

int main(int argc, char** argv)
{
    rss_init.record();
    tim::format::timer::set_default_format(tim_format);
    tim::manager::instance()->update_total_timer_format();

    if(argc > 1)
        command = std::string(const_cast<const char*>(argv[1]));
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
