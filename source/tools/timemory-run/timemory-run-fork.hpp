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
//

#pragma once

#include "timemory/utility/utility.hpp"

#include <sstream>
#include <string>
#include <vector>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <grp.h>
#include <paths.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

//
//--------------------------------------------------------------------------------------//
//

using string_t = std::string;
using strvec_t = std::vector<string_t>;

//
//--------------------------------------------------------------------------------------//
//

typedef struct TIMEMORY_PIPE
{
    FILE* read_fd;
    FILE* write_fd;
    pid_t child_pid;
} TIMEMORY_PIPE;

//
//--------------------------------------------------------------------------------------//
//

TIMEMORY_PIPE*
timemory_popen(const char* path, char** argv = nullptr, char** envp = nullptr);

//
//--------------------------------------------------------------------------------------//
//

int
timemory_pclose(TIMEMORY_PIPE* p);

//
//--------------------------------------------------------------------------------------//
//

pid_t
timemory_fork(void);

//
//--------------------------------------------------------------------------------------//
//

inline strvec_t
read_timemory_fork(TIMEMORY_PIPE* ldd)
{
    int      counter = 0;
    strvec_t linked_libraries;

    while(ldd)
    {
        char buffer[4096];
        auto ret = fgets(buffer, 4096, ldd->read_fd);
        if(ret == nullptr || strlen(buffer) == 0)
        {
            if(counter++ > 50)
                break;
            continue;
        }
        auto line = string_t(buffer);
        auto loc  = string_t::npos;
        while((loc = line.find_first_of("\n\t")) != string_t::npos)
            line.erase(loc, 1);
        auto delim = tim::delimit(line, " \n\t=>");
        for(auto itr : delim)
        {
            if(itr.find('/') == 0)
                linked_libraries.push_back(itr);
        }
    }

    return linked_libraries;
}

//
//--------------------------------------------------------------------------------------//
//
