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

#if !defined(TIMEMORY_PYUNITS_SOURCE)
#    define TIMEMORY_PYUNITS_SOURCE
#endif

#include "libpytimemory-component-bundle.hpp"

#include <cstdint>

using namespace tim::component;

//======================================================================================//
//
namespace pyprofile
{
//
using profiler_t           = tim::component_bundle<TIMEMORY_API, user_profiler_bundle>;
using profiler_vec_t       = std::vector<profiler_t>;
using profiler_label_map_t = std::unordered_map<std::string, profiler_vec_t>;
using profiler_index_map_t = std::unordered_map<uint32_t, profiler_label_map_t>;
using strset_t             = std::unordered_set<std::string>;
//
struct config
{
    bool                 is_running         = false;
    bool                 trace_c            = true;
    bool                 include_internal   = false;
    bool                 include_args       = false;
    bool                 include_line       = true;
    bool                 include_filename   = true;
    bool                 full_filepath      = false;
    int32_t              max_stack_depth    = std::numeric_limits<uint16_t>::max();
    int32_t              ignore_stack_depth = 0;
    int32_t              base_stack_depth   = -1;
    std::string          base_module_path   = "";
    strset_t             include_functions  = {};
    strset_t             include_filenames  = {};
    tim::scope::config   profiler_scope     = tim::scope::get_default();
    strset_t             exclude_functions  = { "FILE",       "FUNC",      "LINE",
                                   "get_fcode",  "__exit__",  "_handle_fromlist",
                                   "<module>",   "_shutdown", "isclass",
                                   "isfunction", "basename",  "_get_sep" };
    strset_t             exclude_filenames  = { "__init__.py",       "__main__.py",
                                   "functools.py",      "<frozen importlib._bootstrap>",
                                   "_pylab_helpers.py", "threading.py",
                                   "encoder.py",        "decoder.py" };
    profiler_index_map_t records            = {};
    int32_t verbose = tim::settings::verbose() + ((tim::settings::debug()) ? 16 : 0);
};
//
inline config&
get_config()
{
    static auto*              _instance    = new config{};
    static thread_local auto* _tl_instance = []() {
        static std::atomic<uint32_t> _count{ 0 };
        auto                         _cnt = _count++;
        if(_cnt == 0)
            return _instance;

        auto* _tmp              = new config{};
        _tmp->is_running        = _instance->is_running;
        _tmp->trace_c           = _instance->trace_c;
        _tmp->include_internal  = _instance->include_internal;
        _tmp->include_args      = _instance->include_args;
        _tmp->include_line      = _instance->include_line;
        _tmp->include_filename  = _instance->include_filename;
        _tmp->full_filepath     = _instance->full_filepath;
        _tmp->max_stack_depth   = _instance->max_stack_depth;
        _tmp->base_module_path  = _instance->base_module_path;
        _tmp->include_functions = _instance->include_functions;
        _tmp->include_filenames = _instance->include_filenames;
        _tmp->exclude_functions = _instance->exclude_functions;
        _tmp->exclude_filenames = _instance->exclude_filenames;
        _tmp->verbose           = _instance->verbose;
        return _tmp;
    }();
    return *_tl_instance;
}
//
int32_t
get_depth(PyFrameObject* frame)
{
    return (frame->f_back) ? (get_depth(frame->f_back) + 1) : 0;
}
//
void
profiler_function(py::object pframe, const char* swhat, py::object arg)
{
    if(!tim::settings::enabled())
        return;

    if(user_profiler_bundle::bundle_size() == 0)
    {
        if(tim::settings::debug())
            PRINT_HERE("%s", "Profiler bundle is empty");
        return;
    }

    static thread_local auto& _config        = get_config();
    static auto               _timemory_path = _config.base_module_path;

    auto* frame = reinterpret_cast<PyFrameObject*>(pframe.ptr());

    int what = (strcmp(swhat, "call") == 0)
                   ? PyTrace_CALL
                   : (strcmp(swhat, "c_call") == 0)
                         ? PyTrace_C_CALL
                         : (strcmp(swhat, "return") == 0)
                               ? PyTrace_RETURN
                               : (strcmp(swhat, "c_return") == 0) ? PyTrace_C_RETURN : -1;

    // only support PyTrace_{CALL,C_CALL,RETURN,C_RETURN}
    if(what < 0)
    {
        if(tim::settings::debug())
            PRINT_HERE("%s :: %s", "Ignoring what != {CALL,C_CALL,RETURN,C_RETURN}",
                       swhat);
        return;
    }

    // if PyTrace_C_{CALL,RETURN} is not enabled
    if(!_config.trace_c && (what == PyTrace_C_CALL || what == PyTrace_C_RETURN))
    {
        if(tim::settings::debug())
            PRINT_HERE("%s :: %s", "Ignoring C call/return", swhat);
        return;
    }

    // get the depth of the frame
    auto _fdepth = get_depth(frame);

    if(_config.base_stack_depth < 0)
        _config.base_stack_depth = _fdepth;

    bool    _iscall = (what == PyTrace_CALL || what == PyTrace_C_CALL);
    int32_t _sdepth = _fdepth - _config.base_stack_depth - 3;
    // if frame exceeds max stack-depth
    if(_iscall && _sdepth > _config.max_stack_depth)
    {
        if(tim::settings::debug())
            PRINT_HERE("skipping %i > %i", (int) _sdepth, (int) _config.max_stack_depth);
        return;
    }

    // get the function name
    auto _get_funcname = [&]() -> std::string {
        return py::cast<std::string>(frame->f_code->co_name);
    };

    // get the filename
    auto _get_filename = [&]() -> std::string {
        return py::cast<std::string>(frame->f_code->co_filename);
    };

    // get the basename of the filename
    auto _get_basename = [&](const std::string& _fullpath) {
        if(_fullpath.find('/') != std::string::npos)
            return _fullpath.substr(_fullpath.find_last_of('/') + 1);
        return _fullpath;
    };

    // get the arguments
    auto _get_args = [&]() {
        auto inspect = py::module::import("inspect");
        return py::cast<std::string>(
            inspect.attr("formatargvalues")(*inspect.attr("getargvalues")(pframe)));
    };

    // get the final label
    auto _get_label = [&](auto& _func, auto& _filename, auto& _fullpath) {
        // append the arguments
        if(_config.include_args)
            _func = TIMEMORY_JOIN("", _func, _get_args());
        // append the filename
        if(_config.include_filename)
        {
            if(_config.full_filepath)
                _func = TIMEMORY_JOIN('/', _func, std::move(_fullpath));
            else
                _func = TIMEMORY_JOIN('/', _func, std::move(_filename));
        }
        // append the line number
        if(_config.include_line)
            _func = TIMEMORY_JOIN(':', _func, frame->f_lineno);
        return _func;
    };

    auto& _only_funcs = _config.include_functions;
    auto& _skip_funcs = _config.exclude_functions;
    auto  _func       = _get_funcname();

    if(!_only_funcs.empty() && _only_funcs.find(_func) == _only_funcs.end())
        return;

    if(_skip_funcs.find(_func) != _skip_funcs.end())
    {
        auto _manager = tim::manager::instance();
        if(!_manager || _manager->is_finalized() || _func == "_shutdown")
        {
            auto sys       = py::module::import("sys");
            auto threading = py::module::import("threading");
            sys.attr("setprofile")(py::none{});
            threading.attr("setprofile")(py::none{});
        }
        return;
    }

    auto& _only_files = _config.include_filenames;
    auto& _skip_files = _config.exclude_filenames;
    auto  _full       = _get_filename();
    auto  _file       = _get_basename(_full);

    if(!_config.include_internal &&
       strncmp(_full.c_str(), _timemory_path.c_str(), _timemory_path.length()) == 0)
        return;

    if(!_only_files.empty() && (_only_files.find(_file) == _only_files.end() &&
                                _only_files.find(_full) == _only_files.end()))
    {
#if defined(DEBUG)
        if(tim::settings::debug())
        {
            std::stringstream _opts;
            for(const auto& itr : _only_files)
                _opts << "| " << itr;
            PRINT_HERE("Skipping: [%s | %s | %s] due to [%s]", _func.c_str(),
                       _file.c_str(), _full.c_str(), _opts.str().substr(2).c_str());
        }
#endif
        return;
    }

    if(_skip_files.find(_file) != _skip_files.end() ||
       _skip_files.find(_full) != _skip_files.end())
        return;

    DEBUG_PRINT_HERE("%8s | %s%s | %s | %s", swhat, _func.c_str(), _get_args().c_str(),
                     _file.c_str(), _full.c_str());

    auto _label = _get_label(_func, _file, _full);

    // start function
    auto _profiler_call = [&]() {
        auto& _entry = _config.records[_fdepth][_label];
        _entry.emplace_back(profiler_t{ _label, _config.profiler_scope });
        _entry.back().start();
    };

    // stop function
    auto _profiler_return = [&]() {
        auto fitr = _config.records.find(_fdepth);
        if(fitr == _config.records.end())
            return;
        auto litr = fitr->second.find(_label);
        if(litr == fitr->second.end())
            return;
        if(litr->second.empty())
            return;
        litr->second.back().stop();
        litr->second.pop_back();
    };

    // process what
    switch(what)
    {
        case PyTrace_CALL:
        case PyTrace_C_CALL: _profiler_call(); break;
        case PyTrace_RETURN:
        case PyTrace_C_RETURN: _profiler_return(); break;
        default: break;
    }

    // don't do anything with arg
    tim::consume_parameters(arg);
}
//
py::module
generate(py::module& _pymod)
{
    py::module _prof = _pymod.def_submodule("profiler", "Profiling functions");

    static auto _scope_set = [](bool _flat, bool _timeline) {
        get_config().profiler_scope = tim::scope::config{ _flat, _timeline };
    };

    pycomponent_bundle::generate<user_profiler_bundle>(
        _prof, "profiler_bundle", "User-bundle for Python profiling interface",
        _scope_set);

    auto _init = []() {
        try
        {
            auto _file =
                py::module::import("timemory").attr("__file__").cast<std::string>();
            if(_file.find('/') != std::string::npos)
                _file = _file.substr(0, _file.find_last_of('/'));
            get_config().base_module_path = _file;
        } catch(py::cast_error& e)
        {
            std::cerr << "[profiler_init]> " << e.what() << std::endl;
        }
        if(get_config().is_running)
            return;
        get_config().records.clear();
        get_config().base_stack_depth = -1;
        get_config().is_running       = true;
    };

    auto _fini = []() {
        if(!get_config().is_running)
            return;
        get_config().is_running       = false;
        get_config().base_stack_depth = -1;
        get_config().records.clear();
    };

    _prof.def("profiler_function", &profiler_function, "Profiling function");
    _prof.def("profiler_init", _init, "Initialize the profiler");
    _prof.def("profiler_finalize", _fini, "Finalize the profiler");

    py::class_<config> _pyconfig(_prof, "config", "Profiler configuration");

#define CONFIGURATION_PROPERTY(NAME, TYPE, DOC, ...)                                     \
    _pyconfig.def_property_static(NAME, [](py::object) { return __VA_ARGS__; },          \
                                  [](py::object, TYPE val) { __VA_ARGS__ = val; }, DOC);

    CONFIGURATION_PROPERTY("_is_running", bool, "Profiler is currently running",
                           get_config().is_running)
    CONFIGURATION_PROPERTY("trace_c", bool, "Enable tracing C functions",
                           get_config().trace_c)
    CONFIGURATION_PROPERTY("include_internal", bool, "Include functions within timemory",
                           get_config().include_internal)
    CONFIGURATION_PROPERTY("include_args", bool, "Encode the function arguments",
                           get_config().include_args)
    CONFIGURATION_PROPERTY("include_line", bool, "Encode the function line number",
                           get_config().include_line)
    CONFIGURATION_PROPERTY("include_filename", bool,
                           "Encode the function filename (see also: full_filepath)",
                           get_config().include_filename)
    CONFIGURATION_PROPERTY("full_filepath", bool,
                           "Display the full filepath (instead of file basename)",
                           get_config().full_filepath)
    CONFIGURATION_PROPERTY("max_stack_depth", int32_t, "Maximum stack depth to profile",
                           get_config().max_stack_depth)
    CONFIGURATION_PROPERTY("verbosity", int32_t, "Verbosity of the logging",
                           get_config().verbose)

    static auto _get_strset = [](const strset_t& _targ) {
        auto _out = py::list{};
        for(auto itr : _targ)
            _out.append(itr);
        return _out;
    };

    static auto _set_strset = [](py::list _inp, strset_t& _targ) {
        for(auto itr : _inp)
            _targ.insert(itr.cast<std::string>());
    };

#define CONFIGURATION_PROPERTY_LAMBDA(NAME, DOC, GET, SET)                               \
    _pyconfig.def_property_static(NAME, GET, SET, DOC);
#define CONFIGURATION_STRSET(NAME, DOC, ...)                                             \
    {                                                                                    \
        auto GET = [](py::object) { return _get_strset(__VA_ARGS__); };                  \
        auto SET = [](py::object, py::list val) { _set_strset(val, __VA_ARGS__); };      \
        CONFIGURATION_PROPERTY_LAMBDA(NAME, DOC, GET, SET)                               \
    }

    CONFIGURATION_STRSET("only_functions", "Function names to collect exclusively",
                         get_config().include_functions)
    CONFIGURATION_STRSET("only_filenames", "File names to collect exclusively",
                         get_config().include_filenames)
    CONFIGURATION_STRSET("skip_functions", "Function names to filter out of collection",
                         get_config().exclude_functions)
    CONFIGURATION_STRSET("skip_filenames", "Filenames to filter out of collection",
                         get_config().exclude_filenames)

    tim::operation::init<user_profiler_bundle>(
        tim::operation::mode_constant<tim::operation::init_mode::global>{});

    return _prof;
}
}  // namespace pyprofile
//
//======================================================================================//
