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

#include "timemory/backends/process.hpp"
#include "timemory/environment.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/macros.hpp"

#include "BPatch.h"
#include "BPatch_Vector.h"
#include "BPatch_addressSpace.h"
#include "BPatch_basicBlockLoop.h"
#include "BPatch_callbacks.h"
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

using exec_callback_t = BPatchExecCallback;
using exit_callback_t = BPatchExitCallback;

//======================================================================================//
//
//                                  Global Variables
//
//======================================================================================//

static int         expectError        = NO_ERROR;
static int         debugPrint         = 0;
static int         binaryRewrite      = 0;  /* by default, it is turned off */
static int         errorPrint         = 0;  // external "dyninst" tracing
static int         verboseLevel       = tim::get_env<int>("TIMEMORY_RUN_VERBOSE", 0);
static bool        loop_level_instr   = false;
static bool        werror             = false;
static bool        stl_func_instr     = false;
static bool        use_mpi            = false;
static bool        use_mpip           = false;
static bool        use_ompt           = false;
static std::string main_fname         = "main";
static std::string argv0              = "";
static std::string default_components = "wall_clock";
static std::string instr_push_func    = "timemory_push_trace";
static std::string instr_pop_func     = "timemory_pop_trace";

using snippet_t     = BPatch_snippet;
using snippet_vec_t = BPatch_Vector<snippet_t*>;

static BPatch*              bpatch          = nullptr;
static BPatch_funcCallExpr* initialize_expr = nullptr;
static BPatch_funcCallExpr* terminate_expr  = nullptr;
static snippet_vec_t        init_names;
static snippet_vec_t        fini_names;

static std::vector<std::regex>  regex_include;
static std::vector<std::regex>  regex_exclude;
static std::set<std::string>    collection_includes;
static std::set<std::string>    collection_excludes;
static std::vector<std::string> collection_paths = { "collections",
                                                     "timemory/collections",
                                                     "../share/timemory/collections" };

//
//======================================================================================//

// control debug printf statements
#define dprintf(...)                                                                     \
    if(debugPrint || verboseLevel > 0)                                                   \
        fprintf(stderr, __VA_ARGS__);

// control verbose printf statements
#define verbprintf(LEVEL, ...)                                                           \
    if(verboseLevel >= LEVEL)                                                            \
        fprintf(stderr, __VA_ARGS__);

//======================================================================================//

template <typename... T>
void
consume_parameters(T&&...)
{}

//======================================================================================//

extern "C"
{
    bool are_file_include_exclude_lists_empty();
    void read_collection(const std::string& fname, std::set<std::string>& collection_set);
    bool process_file_for_instrumentation(const string_t& file_name);
    bool instrument_entity(const string_t& function_name);
    int  module_constraint(char* fname);
    int  routine_constraint(const char* fname);
    bool check_if_timemory_source_file(const string_t& fname);
}

//======================================================================================//

function_signature
get_func_file_line_info(BPatch_image* mutateeAddressSpace, BPatch_function* f);

function_signature
get_loop_file_line_info(BPatch_image* mutateeImage, BPatch_flowGraph* cfGraph,
                        BPatch_basicBlockLoop* loopToInstrument, BPatch_function* f);

void
insert_trace(BPatch_function* funcToInstr, BPatch_addressSpace* mutatee,
             BPatch_function* traceEntryFunc, BPatch_function* traceExitFunc,
             BPatch_flowGraph* cfGraph, BPatch_basicBlockLoop* loopToInstrument,
             function_signature& name);

void
insert_trace(BPatch_function* funcToInstr, BPatch_addressSpace* mutatee,
             BPatch_function* traceEntryFunc, BPatch_function* traceExitFunc,
             function_signature& name);

void
errorFunc(BPatchErrorLevel level, int num, const char** params);

BPatch_function*
find_function(BPatch_image* appImage, const char* functionName);

BPatchSnippetHandle*
invoke_routine_in_func(BPatch_process* appThread, BPatch_image* appImage,
                       BPatch_function* function, BPatch_procedureLocation loc,
                       BPatch_function*               callee,
                       BPatch_Vector<BPatch_snippet*> callee_args);

BPatchSnippetHandle*
invoke_routine_in_func(BPatch_process* appThread, BPatch_image* appImage,
                       BPatch_Vector<BPatch_point*> points, BPatch_function* callee,
                       BPatch_Vector<BPatch_snippet*> callee_args);

void
initialize(BPatch_process* appThread, BPatch_image* appImage,
           BPatch_Vector<BPatch_snippet*>& initArgs);

void
check_cost(BPatch_snippet snippet);

void
error_func_real(BPatchErrorLevel level, int num, const char* const* params);

void
error_func_fake(BPatchErrorLevel level, int num, const char* const* params);

bool
find_func_or_calls(std::vector<const char*> names, BPatch_Vector<BPatch_point*>& points,
                   BPatch_image*            appImage,
                   BPatch_procedureLocation loc = BPatch_locEntry);

bool
find_func_or_calls(const char* name, BPatch_Vector<BPatch_point*>& points,
                   BPatch_image* image, BPatch_procedureLocation loc = BPatch_locEntry);

function_signature
get_func_file_line_info(BPatch_image* mutatee_addr_space, BPatch_function* f);

bool
load_dependent_libraries(BPatch_binaryEdit* bedit, char* bindings);

int
timemory_rewrite_binary(BPatch* bpatch, const char* mutateeName, char* outfile,
                        char* sharedlibname, char* staticlibname, char* bindings);

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

    bool             m_loop      = false;
    bool             m_info_beg  = false;
    bool             m_info_end  = false;
    location_t       m_row       = { 0, 0 };
    location_t       m_col       = { 0, 0 };
    string_t         m_return    = "void";
    string_t         m_name      = "";
    string_t         m_file      = "";
    mutable string_t m_signature = "";

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

    static const char* get(function_signature& sig) { return sig.get().c_str(); }

    std::string get() const
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

//======================================================================================//

static void
timemory_thread_exit(BPatch_thread* proc, BPatch_exitType exit_type)
{
    if(proc && terminate_expr)
    {
        switch(exit_type)
        {
            case ExitedNormally:
            {
                static bool _once = false;
                if(!_once)
                {
                    _once = true;
                    fprintf(stderr, "[timemory-run]> Thread exited normally\n");
                    terminate_expr = nullptr;
                    proc->oneTimeCode(*terminate_expr);
                }
                break;
            }
            case ExitedViaSignal:
            {
                fprintf(stderr, "[timemory-run]> Thread terminated unexpectedly\n");
                break;
            }
            case NoExit:
            default:
            {
                fprintf(stderr, "[timemory-run]> %s invoked with NoExit\n", __FUNCTION__);
                break;
            }
        }
    }
    proc->getProcess()->continueExecution();
}

//======================================================================================//
