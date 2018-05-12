// C program to illustrate  use of fork() &
// exec() system call for process creation

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/wait.h>
#include <string.h>

#include "timemory/manager.hpp"
#include "timemory/macros.hpp"
#include "timemory/rss.hpp"
#include "timemory/timer.hpp"

#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <thread>

typedef std::vector<uint64_t> vector_t;

//----------------------------------------------------------------------------//

void consume_memory()
{
    uint64_t n = 10000000;
    vector_t v;
    for(uint64_t i = 0; i < n; ++i)
        v.push_back(i);
}

//----------------------------------------------------------------------------//

void failed_fork()
{
    printf("can't fork, error occured\n");
    exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------//

void parent_process(pid_t pid)
{
    int status;

    // a positive number is returned for the pid of
    // parent process
    // getppid() returns process id of parent of
    // calling process

    //printf("parent process, pid = %u, ppid = %u\n", getpid(), getppid());

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
            {
                std::cout << (*tim::manager::instance()) << std::endl;
                tim::manager::instance()->write_json("timem.json");
            }
        }
        else if (WIFEXITED(status) && WEXITSTATUS(status))
        {
            if (WEXITSTATUS(status) == 127)
                printf("execv failed\n");
            else
                printf("program terminated normally,"
                       " but returned a non-zero status\n");
        }
        else
            printf("program didn't terminate normally\n");
    }
    else
        printf("waitpid() failed\n");

    exit(0);
}

//----------------------------------------------------------------------------//

void child_process(int argc, char** argv)
{
    // the argv list first argument should point to
    // filename associated with file being executed
    // the array pointer must be terminated by NULL
    // pointer
    char** argv_list = new char*[argc];
    memcpy(argv_list, &(argv[1]), argc*sizeof(char*));
    argv_list[argc-1] = NULL;

    // the execv() only return if error occured.
    // The return value is -1
    execvp(argv[0], argv_list);

    delete [] argv_list;
    exit(0);
}

//----------------------------------------------------------------------------//

int main(int argc, char** argv)
{
    pid_t pid = fork();

    //printf("main, pid = %u\n", pid);

    if(pid == -1) // pid == -1 means error occured
        failed_fork();
    else if(pid == 0) // pid == 0 means child process created
        child_process(argc, argv);
    else
        parent_process(pid); // means parent process

    return 0;
}
