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
using fork_callback_t = BPatchForkCallback;

void
timemory_prefork_callback(BPatch_thread* parent, BPatch_thread* child);

//======================================================================================//
//
//                                  Global Variables
//
//======================================================================================//

static int         expect_error       = NO_ERROR;
static int         debug_print        = 0;
static int         binary_rewrite     = 0;  /* by default, it is turned off */
static int         error_print        = 0;  // external "dyninst" tracing
static int         verbose_level      = tim::get_env<int>("TIMEMORY_RUN_VERBOSE", 0);
static bool        loop_level_instr   = false;
static bool        werror             = false;
static bool        stl_func_instr     = false;
static bool        use_mpi            = false;
static bool        use_mpip           = false;
static bool        use_ompt           = false;
static bool        is_static_exe      = false;
static std::string main_fname         = "main";
static std::string argv0              = "";
static std::string cmdv0              = "";
static std::string default_components = "wall_clock";
static std::string instr_push_func    = "timemory_push_trace";
static std::string instr_pop_func     = "timemory_pop_trace";
static std::string prefer_library     = "";

using snippet_t             = BPatch_snippet;
using snippet_vec_t         = BPatch_Vector<snippet_t*>;
using snippet_pointer_t     = std::shared_ptr<snippet_t>;
using snippet_pointer_vec_t = std::vector<snippet_pointer_t>;
using procedure_t           = BPatch_function;
using procedure_vec_t       = BPatch_Vector<procedure_t*>;
using call_expr_t           = BPatch_funcCallExpr;
using call_expr_pointer_t   = std::shared_ptr<BPatch_funcCallExpr>;
using address_space_t       = BPatch_addressSpace;
using basic_loop_t          = BPatch_basicBlockLoop;
using basic_loop_vec_t      = BPatch_Vector<basic_loop_t*>;
using flow_graph_t          = BPatch_flowGraph;
using patch_t               = BPatch;

static patch_t*      bpatch          = nullptr;
static call_expr_t*  initialize_expr = nullptr;
static call_expr_t*  terminate_expr  = nullptr;
static snippet_vec_t init_names;
static snippet_vec_t fini_names;

static std::vector<std::regex>  func_include;
static std::vector<std::regex>  func_exclude;
static std::vector<std::regex>  file_include;
static std::vector<std::regex>  file_exclude;
static std::set<std::string>    collection_includes;
static std::set<std::string>    collection_excludes;
static std::vector<std::string> collection_paths = { "collections",
                                                     "timemory/collections",
                                                     "../share/timemory/collections" };

static std::set<std::string> available_modules;
static std::set<std::string> available_procedures;
static std::set<std::string> instrumented_modules;
static std::set<std::string> instrumented_procedures;

//
//======================================================================================//

// control debug printf statements
#define dprintf(...)                                                                     \
    if(debug_print || verbose_level > 0)                                                 \
        fprintf(stderr, __VA_ARGS__);

// control verbose printf statements
#define verbprintf(LEVEL, ...)                                                           \
    if(verbose_level >= LEVEL)                                                           \
        fprintf(stderr, __VA_ARGS__);

//======================================================================================//

template <typename... T>
void
consume_parameters(T&&...)
{}

//======================================================================================//

static inline void
dump_info(const std::string& _oname, const std::set<std::string> _data, int level)
{
    if(!debug_print && verbose_level > level)
        return;

    std::ofstream ofs(_oname);
    if(ofs)
    {
        verbprintf(level, "Dumping '%s'...\n", _oname.c_str());
        for(const auto& itr : _data)
            ofs << itr << '\n';
    }
    ofs.close();
}

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
get_func_file_line_info(BPatch_image* mutateeImage, BPatch_function* f);

function_signature
get_loop_file_line_info(BPatch_image* mutateeImage, BPatch_function* f,
                        BPatch_flowGraph*      cfGraph,
                        BPatch_basicBlockLoop* loopToInstrument);

void
insert_instr(address_space_t* mutatee, BPatch_function* funcToInstr,
             BPatch_function* traceFunc, BPatch_procedureLocation traceLoc,
             function_signature& name, BPatch_flowGraph* cfGraph = nullptr,
             BPatch_basicBlockLoop* loopToInstrument = nullptr);

void
errorFunc(BPatchErrorLevel level, int num, const char** params);

BPatch_function*
find_function(BPatch_image* appImage, const char* functionName);

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

bool
load_dependent_libraries(address_space_t* bedit, char* bindings);

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
//
//======================================================================================//
//
template <typename Tp>
snippet_pointer_t
get_snippet(Tp arg)
{
    return snippet_pointer_t(new BPatch_constExpr(arg));
}
//
//======================================================================================//
//
inline snippet_pointer_t
get_snippet(std::string arg)
{
    return snippet_pointer_t(new BPatch_constExpr(arg.c_str()));
}
//
//======================================================================================//
//
template <typename... Args>
snippet_pointer_vec_t
get_snippets(Args&&... args)
{
    snippet_pointer_vec_t _tmp;
    TIMEMORY_FOLD_EXPRESSION(_tmp.push_back(get_snippet(std::forward<Args>(args))));
    return _tmp;
}
//
//======================================================================================//
//
struct timemory_call_expr
{
    using snippet_pointer_t = std::shared_ptr<BPatch_snippet>;

    template <typename... Args>
    timemory_call_expr(Args&&... args)
    : m_params(get_snippets(std::forward<Args>(args)...))
    {}

    snippet_vec_t get_params()
    {
        snippet_vec_t _ret;
        for(auto& itr : m_params)
            _ret.push_back(itr.get());
        return _ret;
    }

    inline call_expr_pointer_t get(procedure_t* func)
    {
        return call_expr_pointer_t((func) ? new call_expr_t(*func, get_params())
                                          : nullptr);
    }

private:
    snippet_pointer_vec_t m_params;
};
//
//======================================================================================//
//
static inline address_space_t*
timemory_get_address_space(BPatch* bpatch, int _cmdc, char** _cmdv, bool _rewrite,
                           int _pid = -1, std::string _name = "")
{
    address_space_t* mutatee = nullptr;

    if(_rewrite)
    {
        mutatee = bpatch->openBinary(_name.c_str(), false);
        if(!mutatee)
        {
            fprintf(stderr, "[timemory-run]> Failed to open binary '%s'\n",
                    _name.c_str());
            throw std::runtime_error("Failed to open binary");
        }
    }
    else if(_pid >= 0)
    {
        verbprintf(0, "Before processAttach\n");
        char* _cmdv0 = (_cmdc > 0) ? _cmdv[0] : nullptr;
        mutatee      = bpatch->processAttach(_cmdv0, _pid);
        if(!mutatee)
        {
            fprintf(stderr, "[timemory-run]> Failed to connect to process %i\n",
                    (int) _pid);
            throw std::runtime_error("Failed to attach to process");
        }
    }
    else
    {
        verbprintf(0, "Before processCreate\n");
        mutatee = bpatch->processCreate(_cmdv[0], (const char**) _cmdv, nullptr);
        if(!mutatee)
        {
            std::stringstream ss;
            for(int i = 0; i < _cmdc; ++i)
            {
                if(!_cmdv[i])
                    continue;
                ss << _cmdv[i] << " ";
            }
            fprintf(stderr, "[timemory-run]> Failed to create process: '%s'\n",
                    ss.str().c_str());
            throw std::runtime_error("Failed to create process");
        }
    }

    return mutatee;
}
//
//======================================================================================//
//
static void
timemory_thread_exit(BPatch_thread* thread, BPatch_exitType exit_type)
{
    if(!thread)
        return;

    BPatch_process* app = thread->getProcess();

    if(!terminate_expr)
    {
        app->continueExecution();
        return;
    }

    switch(exit_type)
    {
        case ExitedNormally:
        {
            fprintf(stderr, "[timemory-run]> Thread exited normally\n");
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

    // terminate_expr = nullptr;
    thread->oneTimeCode(*terminate_expr);

    app->continueExecution();
}
//
//======================================================================================//
//
static void
timemory_fork_callback(BPatch_thread* parent, BPatch_thread* child)
{
    if(child)
    {
        auto* app = child->getProcess();
        if(app)
        {
            verbprintf(4, "Stopping execution and detaching child fork...\n");
            app->stopExecution();
            app->detach(true);
            // app->terminateExecution();
            // app->continueExecution();
        }
    }

    if(parent)
    {
        auto app = parent->getProcess();
        if(app)
        {
            verbprintf(4, "Continuing execution on parent after fork callback...\n");
            app->continueExecution();
        }
    }
}
//
//======================================================================================//
//
