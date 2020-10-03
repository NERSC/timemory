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
using profiler_t = tim::component_bundle<TIMEMORY_API, user_profiler_bundle>;
//
using profiler_vec_t = std::vector<profiler_t>;
//
using profiler_label_map_t = std::unordered_map<std::string, profiler_vec_t>;
//
using profiler_index_map_t = std::unordered_map<uint32_t, profiler_label_map_t>;
//
using strset_t = std::unordered_set<std::string>;
//
struct config
{
    bool        is_running               = false;
    bool        trace_c                  = true;
    bool        include_internal         = false;
    bool        include_args             = false;
    bool        include_line             = true;
    bool        include_filename         = true;
    bool        full_filepath            = false;
    int32_t     max_stack_depth          = std::numeric_limits<uint16_t>::max();
    int32_t     ignore_stack_depth       = 0;
    int32_t     base_stack_depth         = -1;
    std::string base_module_path         = "";
    strset_t    always_skipped_functions = { "FILE",      "FUNC",     "LINE",
                                          "get_fcode", "__exit__", "_handle_fromlist",
                                          "<module>" };
    strset_t    always_skipped_filenames = { "__init__.py", "__main__.py", "functools.py",
                                          "<frozen importlib._bootstrap>" };
    profiler_index_map_t records         = {};
};
//
inline config&
get_config()
{
    static auto* _instance = new config{};
    return *_instance;
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
    if(user_profiler_bundle::bundle_size() == 0)
    {
        if(tim::settings::debug())
            PRINT_HERE("%s", "Profiler bundle is empty");
        return;
    }

    static auto _timemory_path = get_config().base_module_path;

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
            PRINT_HERE("%s", "Ignoring what != {CALL,C_CALL,RETURN,C_RETURN}");
        return;
    }

    // if PyTrace_C_{CALL,RETURN} is not enabled
    if(!get_config().trace_c && (what == PyTrace_C_CALL || what == PyTrace_C_RETURN))
    {
        if(tim::settings::debug())
            PRINT_HERE("%s", "Ignoring C call/return");
        return;
    }

    // get the depth of the frame
    auto _fdepth = get_depth(frame);

    if(get_config().base_stack_depth < 0)
        get_config().base_stack_depth = _fdepth;

    bool    _iscall = (what == PyTrace_CALL || what == PyTrace_C_CALL);
    int32_t _sdepth = _fdepth - get_config().base_stack_depth - 3;
    // if frame exceeds max stack-depth
    if(_iscall && _sdepth > get_config().max_stack_depth)
    {
        if(tim::settings::debug())
            PRINT_HERE("skipping %i > %i", (int) _sdepth,
                       (int) get_config().max_stack_depth);
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

    // get the final filename
    auto _get_label = [&](auto& _func, auto& _filename, auto& _fullpath) {
        // append the arguments
        if(get_config().include_args)
            _func = TIMEMORY_JOIN("", _func, _get_args());
        // append the filename
        if(get_config().include_filename)
        {
            if(get_config().full_filepath)
                _func = TIMEMORY_JOIN('/', _func, std::move(_fullpath));
            else
                _func = TIMEMORY_JOIN('/', _func, std::move(_filename));
        }
        // append the line number
        if(get_config().include_line)
            _func = TIMEMORY_JOIN(':', _func, frame->f_lineno);
        return _func;
    };

    // get the iterator at the stack depth
    auto _profiler_index = [&](bool _insert) {
        auto _itr = get_config().records.find(_fdepth);
        if(_insert && _itr == get_config().records.end())
            _itr = get_config().records.insert({ _fdepth, profiler_label_map_t{} }).first;
        return _itr;
    };

    // get the stack-depth iterator for a label
    auto _profiler_label = [&](bool _insert, auto itr, const auto& label) {
        auto _itr = itr->second.find(label);
        if(_insert)
        {
            if(_itr == itr->second.end())
                _itr = itr->second.insert({ label, profiler_vec_t{} }).first;
            _itr->second.emplace_back(profiler_t{ label });
        }
        return _itr;
    };

    // start function
    auto _profiler_call = [&]() {
        auto& _skip_funcs = get_config().always_skipped_functions;
        auto  _func       = _get_funcname();
        if(_skip_funcs.find(_func) != _skip_funcs.end())
            return;

        auto& _skip_files = get_config().always_skipped_filenames;
        auto  _full       = _get_filename();
        auto  _file       = _get_basename(_full);

        if(!get_config().include_internal &&
           strncmp(_full.c_str(), _timemory_path.c_str(), _timemory_path.length()) == 0)
        {
            //++get_config().ignore_stack_depth;
            return;
        }

        if(_skip_files.find(_file) != _skip_files.end() ||
           _skip_files.find(_full) != _skip_files.end())
            return;

        DEBUG_PRINT_HERE("%8s | %s%s | %s | %s", swhat, _func.c_str(),
                         _get_args().c_str(), _file.c_str(), _full.c_str());

        auto itr_idx = _profiler_index(true);
        auto itr_lbl = _profiler_label(true, itr_idx, _get_label(_func, _file, _full));
        itr_lbl->second.back().start();
    };

    // stop function
    auto _profiler_return = [&]() {
        // if(get_config().ignore_stack_depth > 0)
        //{
        //    --get_config().ignore_stack_depth;
        //    return;
        //}

        auto& _skip_funcs = get_config().always_skipped_functions;
        auto  _func       = _get_funcname();
        if(_skip_funcs.find(_func) != _skip_funcs.end())
            return;

        auto& _skip_files = get_config().always_skipped_filenames;
        auto  _full       = _get_filename();
        auto  _file       = _get_basename(_full);

        if(strncmp(_full.c_str(), _timemory_path.c_str(), _timemory_path.length()) == 0)
            return;

        if(_skip_files.find(_file) != _skip_files.end() ||
           _skip_files.find(_full) != _skip_files.end())
            return;

        DEBUG_PRINT_HERE("%8s | %s%s | %s | %s", swhat, _func.c_str(),
                         _get_args().c_str(), _file.c_str(), _full.c_str());

        auto itr_idx = _profiler_index(false);
        if(itr_idx == get_config().records.end())
            return;
        auto itr_lbl = _profiler_label(false, itr_idx, _get_label(_func, _file, _full));
        if(itr_lbl == itr_idx->second.end())
            return;
        itr_lbl->second.back().stop();
        itr_lbl->second.pop_back();
        if(itr_lbl->second.empty())
        {
            itr_idx->second.erase(itr_lbl->first);
            if(itr_idx->second.empty())
            {
                get_config().records.erase(itr_idx->first);
                if(get_config().records.empty())
                    get_config().base_stack_depth = -1;
            }
        }
    };

    // process what
    switch(what)
    {
        case PyTrace_CALL:
        case PyTrace_C_CALL: _profiler_call(); break;
        case PyTrace_RETURN:
        case PyTrace_C_RETURN: _profiler_return(); break;
        case PyTrace_C_EXCEPTION:
        case PyTrace_EXCEPTION:
        case PyTrace_LINE:
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

    pycomponent_bundle::generate<user_profiler_bundle>(
        _prof, "profiler_bundle", "User-bundle for Python profiling interface");

    auto _init = []() {
        user_profiler_bundle::global_init();
        auto _file = py::module::import("timemory").attr("__file__").cast<std::string>();
        if(_file.find('/') != std::string::npos)
            _file = _file.substr(0, _file.find_last_of('/'));
        get_config().base_module_path = _file;
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
    _prof.def("initialize", _init, "Initialize the profiler");
    _prof.def("finalize", _fini, "Finalize the profiler");

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
    CONFIGURATION_PROPERTY("skip_functions", strset_t,
                           "Function names to filter out of collection",
                           get_config().always_skipped_functions)
    CONFIGURATION_PROPERTY("skip_filenames", strset_t,
                           "Filenames to filter out of collection",
                           get_config().always_skipped_filenames)

    return _prof;
}
}  // namespace pyprofile
//
//======================================================================================//
