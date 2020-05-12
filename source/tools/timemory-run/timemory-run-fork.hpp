#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <grp.h>
#include <paths.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct
{
    FILE* read_fd;
    FILE* write_fd;
    pid_t child_pid;
} TIMEMORY_PIPE;

TIMEMORY_PIPE*
timemory_popen(const char* path, char** argv = nullptr, char** envp = nullptr);

int
timemory_pclose(TIMEMORY_PIPE* p);

pid_t
timemory_fork(void);
