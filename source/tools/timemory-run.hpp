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

#include "timemory/environment.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/macros.hpp"

#include "BPatch.h"
#include "BPatch_Vector.h"
#include "BPatch_addressSpace.h"
#include "BPatch_basicBlockLoop.h"
#include "BPatch_function.h"
#include "BPatch_point.h"
#include "BPatch_process.h"
#include "BPatch_snippet.h"
#include "BPatch_statement.h"

#include <cstring>
#include <limits>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <vector>
//
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#define MUTNAMELEN 64
#define FUNCNAMELEN 32 * 1024
#define NO_ERROR -1
#define TIMEMORY_BIN_DIR "bin"

#if !defined(PATH_MAX)
#    define PATH_MAX std::numeric_limits<int>::max();
#endif

using string_t = std::string;

struct function_signature;

//======================================================================================//

// control debug printf statements
#define dprintf(...)                                                                     \
    if(debugPrint)                                                                       \
        fprintf(stderr, __VA_ARGS__);

//======================================================================================//

void
check_cost(BPatch_snippet snippet);

//======================================================================================//

template <typename... T>
void
consume_parameters(T&&...)
{}

//======================================================================================//

extern "C"
{
    bool check_project_source_file_for_instrumentation(const string_t& fname);

    bool are_file_include_exclude_lists_empty(void);

    bool process_file_for_instrumentation(const string_t& file_name);

    bool instrument_entity(const string_t& function_name);

    function_signature get_func_file_line_info(BPatch_image*    mutateeAddressSpace,
                                               BPatch_function* f);

    function_signature get_loop_file_line_info(BPatch_image*          mutateeImage,
                                               BPatch_flowGraph*      cfGraph,
                                               BPatch_basicBlockLoop* loopToInstrument,
                                               BPatch_function*       f);

    extern bool match_name(const string_t& str1, const string_t& str2);
}

//======================================================================================//

inline string_t
get_absolute_path(const char* fname)
{
    char  path_save[PATH_MAX];
    char  abs_exe_path[PATH_MAX];
    char* p = nullptr;

    if(!(p = strrchr((char*) fname, '/')))
    {
        auto ret = getcwd(abs_exe_path, sizeof(abs_exe_path));
        consume_parameters(ret);
    }
    else
    {
        auto rets = getcwd(path_save, sizeof(path_save));
        auto retf = chdir(fname);
        auto reta = getcwd(abs_exe_path, sizeof(abs_exe_path));
        auto retp = chdir(path_save);
        consume_parameters(rets, retf, reta, retp);
    }
    return string_t(abs_exe_path);
}

//======================================================================================//

inline void
prefix_collection_path(const string_t& p, std::vector<string_t>& paths)
{
    if(!p.empty())
    {
        auto delim = tim::delimit(p, " ;:,");
        auto old   = paths;
        paths      = delim;
        for(auto&& itr : old)
            paths.push_back(itr);
    }
}

//======================================================================================//

inline string_t
to_lower(string_t s)
{
    for(auto& itr : s)
        itr = tolower(itr);
    return s;
}
//
//======================================================================================//
//
struct function_signature
{
    using location_t = std::pair<unsigned long, unsigned long>;

    bool       m_loop      = false;
    bool       m_info_beg  = false;
    bool       m_info_end  = false;
    location_t m_row       = { 0, 0 };
    location_t m_col       = { 0, 0 };
    string_t   m_return    = "void";
    string_t   m_name      = "";
    string_t   m_file      = "";
    string_t   m_signature = "";

    function_signature(string_t _ret, string_t _name, string_t _file,
                       location_t _row = { 0, 0 }, location_t _col = { 0, 0 },
                       bool _loop = false, bool _info_beg = false, bool _info_end = false)
    : m_loop(_loop)
    , m_info_beg(_info_beg)
    , m_info_end(_info_end)
    , m_row(_row)
    , m_col(_col)
    , m_return(_ret)
    , m_name(_name)
    , m_file(_file)
    {
        if(m_file.find('/') != std::string::npos)
            m_file = m_file.substr(m_file.find_last_of('/') + 1);
    }

    operator const char*() { return get().c_str(); }

    static const char* get(function_signature& sig) { return sig.get().c_str(); }

    std::string get()
    {
        std::stringstream ss;
        if(m_loop && m_info_beg)
        {
            if(m_info_end)
            {
                ss << m_return << " " << m_name << "()/" << m_file << ":"
                   << "[{" << m_row.first << "," << m_col.first << "}-{" << m_row.second
                   << "," << m_col.second << "}]";
            }
            else
            {
                ss << m_return << " " << m_name << "()/" << m_file << ":"
                   << "[{" << m_row.first << "," << m_col.first << "}]";
            }
        }
        else if(m_file.length() > 0)
        {
            ss << m_return << " " << m_name << "()/" << m_file << ":" << m_row.first;
        }
        else
        {
            ss << m_return << " " << m_name << "()";
        }
        m_signature = ss.str();
        return m_signature;
    }
};
