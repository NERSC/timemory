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

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/environment.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/popen.hpp"
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

#define MUTNAMELEN 1024
#define FUNCNAMELEN 32 * 1024
#define NO_ERROR -1
#define TIMEMORY_BIN_DIR "bin"

#if !defined(PATH_MAX)
#    define PATH_MAX std::numeric_limits<int>::max();
#endif

struct function_signature;
struct module_function;

template <typename Tp>
using bpvector_t = BPatch_Vector<Tp>;

using string_t              = std::string;
using stringstream_t        = std::stringstream;
using strvec_t              = std::vector<string_t>;
using strset_t              = std::set<string_t>;
using regexvec_t            = std::vector<std::regex>;
using fmodset_t             = std::set<module_function>;
using exec_callback_t       = BPatchExecCallback;
using exit_callback_t       = BPatchExitCallback;
using fork_callback_t       = BPatchForkCallback;
using patch_t               = BPatch;
using process_t             = BPatch_process;
using thread_t              = BPatch_thread;
using binary_edit_t         = BPatch_binaryEdit;
using image_t               = BPatch_image;
using module_t              = BPatch_module;
using procedure_t           = BPatch_function;
using snippet_t             = BPatch_snippet;
using call_expr_t           = BPatch_funcCallExpr;
using address_space_t       = BPatch_addressSpace;
using flow_graph_t          = BPatch_flowGraph;
using basic_loop_t          = BPatch_basicBlockLoop;
using procedure_loc_t       = BPatch_procedureLocation;
using point_t               = BPatch_point;
using local_var_t           = BPatch_localVar;
using const_expr_t          = BPatch_constExpr;
using error_level_t         = BPatchErrorLevel;
using patch_pointer_t       = std::shared_ptr<patch_t>;
using snippet_pointer_t     = std::shared_ptr<snippet_t>;
using call_expr_pointer_t   = std::shared_ptr<call_expr_t>;
using snippet_vec_t         = bpvector_t<snippet_t*>;
using procedure_vec_t       = bpvector_t<procedure_t*>;
using basic_loop_vec_t      = bpvector_t<basic_loop_t*>;
using snippet_pointer_vec_t = std::vector<snippet_pointer_t>;

void
timemory_prefork_callback(thread_t* parent, thread_t* child);

//======================================================================================//
//
//                                  Global Variables
//
//======================================================================================//
//
//  boolean settings
//
static bool binary_rewrite   = 0;
static bool loop_level_instr = false;
static bool werror           = false;
static bool stl_func_instr   = false;
static bool cstd_func_instr  = false;
static bool use_mpi          = false;
static bool is_static_exe    = false;
static bool use_return_info  = false;
static bool use_args_info    = false;
static bool use_file_info    = false;
static bool use_line_info    = false;
//
//  integral settings
//
static bool debug_print   = false;
static int  expect_error  = NO_ERROR;
static int  error_print   = 0;
static int  verbose_level = tim::get_env<int>("TIMEMORY_RUN_VERBOSE", 0);
//
//  string settings
//
static string_t main_fname         = "main";
static string_t argv0              = "";
static string_t cmdv0              = "";
static string_t default_components = "wall_clock";
static string_t prefer_library     = "";
//
//  global variables
//
static patch_pointer_t bpatch;
static call_expr_t*    initialize_expr = nullptr;
static call_expr_t*    terminate_expr  = nullptr;
static snippet_vec_t   init_names;
static snippet_vec_t   fini_names;
static fmodset_t       available_module_functions;
static fmodset_t       instrumented_module_functions;
static regexvec_t      func_include;
static regexvec_t      func_exclude;
static regexvec_t      file_include;
static regexvec_t      file_exclude;
static strset_t        collection_includes;
static strset_t        collection_excludes;
static strvec_t        collection_paths = { "collections", "timemory/collections",
                                     "../share/timemory/collections" };
static auto regex_opts = std::regex_constants::egrep | std::regex_constants::optimize;
//
//======================================================================================//

// control debug printf statements
#define dprintf(...)                                                                     \
    if(tim::dmp::rank() == 0 && (debug_print || verbose_level > 0))                      \
        fprintf(stderr, __VA_ARGS__);                                                    \
    fflush(stderr);

// control verbose printf statements
#define verbprintf(LEVEL, ...)                                                           \
    if(tim::dmp::rank() == 0 && verbose_level >= LEVEL)                                  \
        fprintf(stdout, __VA_ARGS__);                                                    \
    fflush(stdout);

//======================================================================================//

template <typename... T>
void
consume_parameters(T&&...)
{}

//======================================================================================//

extern "C"
{
    bool are_file_include_exclude_lists_empty();
    void read_collection(const string_t& fname, strset_t& collection_set);
    bool process_file_for_instrumentation(const string_t& file_name);
    bool instrument_entity(const string_t& function_name);
    int  module_constraint(char* fname);
    int  routine_constraint(const char* fname);
    bool timemory_source_file_constraint(const string_t& fname);
}

//======================================================================================//

function_signature
get_func_file_line_info(module_t* mutatee_module, procedure_t* f);

function_signature
get_loop_file_line_info(module_t* mutatee_module, procedure_t* f, flow_graph_t* cfGraph,
                        basic_loop_t* loopToInstrument);

template <typename Tp>
void
insert_instr(address_space_t* mutatee, procedure_t* funcToInstr, Tp traceFunc,
             procedure_loc_t traceLoc, flow_graph_t* cfGraph = nullptr,
             basic_loop_t* loopToInstrument = nullptr);

void
errorFunc(error_level_t level, int num, const char** params);

procedure_t*
find_function(image_t* appImage, const string_t& functionName, strset_t = {});

void
error_func_real(error_level_t level, int num, const char* const* params);

void
error_func_fake(error_level_t level, int num, const char* const* params);

bool
find_func_or_calls(std::vector<const char*> names, bpvector_t<point_t*>& points,
                   image_t* appImage, procedure_loc_t loc = BPatch_locEntry);

bool
find_func_or_calls(const char* name, bpvector_t<point_t*>& points, image_t* image,
                   procedure_loc_t loc = BPatch_locEntry);

bool
load_dependent_libraries(address_space_t* bedit, char* bindings);

bool
c_stdlib_module_constraint(const string_t& file);

bool
c_stdlib_function_constraint(const string_t& func);

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
    string_t         m_params    = "()";
    string_t         m_file      = "";
    mutable string_t m_signature = "";

    TIMEMORY_DEFAULT_OBJECT(function_signature)

    function_signature(string_t _ret, string_t _name, string_t _file,
                       location_t _row = { 0, 0 }, location_t _col = { 0, 0 },
                       bool _loop = false, bool _info_beg = false, bool _info_end = false)
    : m_loop(_loop)
    , m_info_beg(_info_beg)
    , m_info_end(_info_end)
    , m_row(_row)
    , m_col(_col)
    , m_return(_ret)
    , m_name(tim::demangle(_name))
    , m_file(_file)
    {
        if(m_file.find('/') != string_t::npos)
            m_file = m_file.substr(m_file.find_last_of('/') + 1);
    }

    function_signature(string_t _ret, string_t _name, string_t _file,
                       std::vector<string_t> _params, location_t _row = { 0, 0 },
                       location_t _col = { 0, 0 }, bool _loop = false,
                       bool _info_beg = false, bool _info_end = false)
    : function_signature(_ret, _name, _file, _row, _col, _loop, _info_beg, _info_end)
    {
        std::stringstream ss;
        ss << "(";
        for(auto& itr : _params)
            ss << itr << ", ";
        m_params = ss.str();
        m_params = m_params.substr(0, m_params.length() - 2);
        m_params += ")";
    }

    static auto get(function_signature& sig) { return sig.get(); }

    string_t get() const
    {
        std::stringstream ss;
        if(use_return_info)
            ss << m_return << " ";
        ss << m_name;
        if(use_args_info)
            ss << m_params;
        if(m_loop && m_info_beg)
        {
            if(m_info_end)
            {
                ss << '/' << "[{" << m_row.first << "," << m_col.first << "}-{"
                   << m_row.second << "," << m_col.second << "}]";
            }
            else
            {
                ss << "[{" << m_row.first << "," << m_col.first << "}]";
            }
        }
        else
        {
            if(use_file_info && m_file.length() > 0)
                ss << '/' << m_file;
            if(use_line_info && m_row.first > 0)
                ss << ":" << m_row.first;
        }

        m_signature = ss.str();
        return m_signature;
    }
};
//
//======================================================================================//
//
struct module_function
{
    using width_t = std::array<size_t, 3>;

    static auto& get_width()
    {
        static width_t _instance = []() {
            width_t _tmp;
            _tmp.fill(0);
            return _tmp;
        }();
        return _instance;
    }

    static void reset_width() { get_width().fill(0); }

    static void update_width(const module_function& rhs)
    {
        get_width()[0] = std::max<size_t>(get_width()[0], rhs.module.length());
        get_width()[1] = std::max<size_t>(get_width()[1], rhs.function.length());
        get_width()[2] = std::max<size_t>(get_width()[2], rhs.signature.get().length());
    }

    module_function(const string_t& _module, const string_t& _func,
                    const function_signature& _sign)
    : module(_module)
    , function(_func)
    , signature(_sign)
    {}

    module_function(module_t* mod, procedure_t* proc)
    {
        char modname[FUNCNAMELEN];
        char fname[FUNCNAMELEN];

        mod->getName(modname, FUNCNAMELEN);
        proc->getName(fname, FUNCNAMELEN);

        module    = modname;
        function  = fname;
        signature = get_func_file_line_info(mod, proc);
    }

    friend bool operator<(const module_function& lhs, const module_function& rhs)
    {
        return (lhs.module == rhs.module)
                   ? ((lhs.function == rhs.function)
                          ? (lhs.signature.get() < rhs.signature.get())
                          : (lhs.function < rhs.function))
                   : (lhs.module < rhs.module);
    }

    friend std::ostream& operator<<(std::ostream& os, const module_function& rhs)
    {
        std::stringstream ss;

        static size_t absolute_max = 80;
        auto          w0           = std::min<size_t>(get_width()[0], absolute_max);
        auto          w1           = std::min<size_t>(get_width()[1], absolute_max);
        auto          w2           = std::min<size_t>(get_width()[2], absolute_max);

        auto _get_str = [](const std::string& _inc) {
            if(_inc.length() > absolute_max)
                return _inc.substr(0, absolute_max - 3) + "...";
            return _inc;
        };

        ss << std::setw(w0 + 8) << std::left << _get_str(rhs.module) << " "
           << std::setw(w1 + 8) << std::left << _get_str(rhs.function) << " "
           << std::setw(w2 + 8) << std::left << _get_str(rhs.signature.get());
        os << ss.str();
        return os;
    }

    string_t           module   = "";
    string_t           function = "";
    function_signature signature;
};
//
//======================================================================================//
//
static inline void
dump_info(const string_t& _oname, const fmodset_t& _data, int level)
{
    if(!debug_print && verbose_level < level)
        return;

    module_function::reset_width();
    for(const auto& itr : _data)
        module_function::update_width(itr);

    std::ofstream ofs(_oname);
    if(ofs)
    {
        verbprintf(level, "Dumping '%s'... ", _oname.c_str());
        for(const auto& itr : _data)
            ofs << itr << '\n';
        verbprintf(level, "Done\n");
    }
    ofs.close();

    module_function::reset_width();
}
//
//======================================================================================//
//
template <typename Tp>
snippet_pointer_t
get_snippet(Tp arg)
{
    return snippet_pointer_t(new const_expr_t(arg));
}
//
//======================================================================================//
//
inline snippet_pointer_t
get_snippet(string_t arg)
{
    return snippet_pointer_t(new const_expr_t(arg.c_str()));
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
    using snippet_pointer_t = std::shared_ptr<snippet_t>;

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
struct timemory_snippet_vec
{
    using entry_type = std::vector<timemory_call_expr>;
    using value_type = std::vector<call_expr_pointer_t>;

    template <typename... Args>
    void generate(procedure_t* func, Args&&... args)
    {
        auto _expr = timemory_call_expr(std::forward<Args>(args)...);
        auto _call = _expr.get(func);
        if(_call)
        {
            m_entries.push_back(_expr);
            m_data.push_back(_call);
            // m_data.push_back(entry_type{ _call, _expr });
        }
    }

    void append(snippet_vec_t& _obj)
    {
        for(auto& itr : m_data)
            _obj.push_back(itr.get());
    }

private:
    entry_type m_entries;
    value_type m_data;
};
//
//======================================================================================//
//
static inline address_space_t*
timemory_get_address_space(patch_pointer_t _bpatch, int _cmdc, char** _cmdv,
                           bool _rewrite, int _pid = -1, string_t _name = "")
{
    address_space_t* mutatee = nullptr;

    if(_rewrite)
    {
        verbprintf(1, "Opening '%s' for binary rewrite... ", _name.c_str());
        fflush(stderr);
        if(!_name.empty())
            mutatee = _bpatch->openBinary(_name.c_str(), false);
        if(!mutatee)
        {
            fprintf(stderr, "[timemory-run]> Failed to open binary '%s'\n",
                    _name.c_str());
            throw std::runtime_error("Failed to open binary");
        }
        verbprintf(1, "Done\n");
    }
    else if(_pid >= 0)
    {
        verbprintf(1, "Attaching to process %i... ", _pid);
        fflush(stderr);
        char* _cmdv0 = (_cmdc > 0) ? _cmdv[0] : nullptr;
        mutatee      = _bpatch->processAttach(_cmdv0, _pid);
        if(!mutatee)
        {
            fprintf(stderr, "[timemory-run]> Failed to connect to process %i\n",
                    (int) _pid);
            throw std::runtime_error("Failed to attach to process");
        }
        verbprintf(1, "Done\n");
    }
    else
    {
        verbprintf(1, "Creating process '%s'... ", _cmdv[0]);
        fflush(stderr);
        mutatee = _bpatch->processCreate(_cmdv[0], (const char**) _cmdv, nullptr);
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
        verbprintf(1, "Done\n");
    }

    return mutatee;
}
//
//======================================================================================//
//
TIMEMORY_NOINLINE inline void
timemory_thread_exit(thread_t* thread, BPatch_exitType exit_type)
{
    if(!thread)
        return;

    BPatch_process* app = thread->getProcess();

    if(!terminate_expr)
    {
        fprintf(stderr, "[timemory-run]> continuing execution\n");
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

    fprintf(stderr, "[timemory-run]> continuing execution\n");
    app->continueExecution();
}
//
//======================================================================================//
//
TIMEMORY_NOINLINE inline void
timemory_fork_callback(thread_t* parent, thread_t* child)
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
