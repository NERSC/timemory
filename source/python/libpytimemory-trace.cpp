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
#include "timemory/library.h"

#include <cstdint>

using namespace tim::component;

namespace pytrace
{
//
template <typename KeyT, typename MappedT>
using uomap_t = std::unordered_map<KeyT, MappedT>;
//
using tracer_t          = tim::component_bundle<TIMEMORY_API, user_trace_bundle>;
using tracer_line_map_t = uomap_t<size_t, tracer_t>;
using tracer_code_map_t = uomap_t<PyCodeObject*, tracer_line_map_t>;
using tracer_iterator_t = typename tracer_line_map_t::iterator;
using strset_t          = std::unordered_set<std::string>;
using strvec_t          = std::vector<std::string>;
using file_line_map_t   = std::map<std::string, strvec_t>;
//
struct config
{
    bool        is_running               = false;
    bool        include_internal         = false;
    bool        include_args             = false;
    bool        include_line             = true;
    bool        include_filename         = true;
    bool        full_filepath            = false;
    std::string base_module_path         = "";
    strset_t    always_skipped_functions = { "FILE",       "FUNC",      "LINE",
                                          "get_fcode",  "__exit__",  "_handle_fromlist",
                                          "<module>",   "_shutdown", "isclass",
                                          "isfunction", "basename",  "_get_sep" };
    strset_t    always_skipped_filenames = {
        "__init__.py",       "__main__.py",
        "functools.py",      "<frozen importlib._bootstrap>",
        "_pylab_helpers.py", "threading.py",
        "encoder.py",        "decoder.py"
    };
    tracer_code_map_t records = {};
};
//
inline config&
get_config()
{
    static auto*              _instance    = new config{};
    static thread_local auto* _tl_instance = [&]() {
        static std::atomic<uint32_t> _count{ 0 };
        auto                         _cnt = _count++;
        if(_cnt == 0)
            return _instance;

        auto* _tmp                     = new config{};
        _tmp->is_running               = _instance->is_running;
        _tmp->include_internal         = _instance->include_internal;
        _tmp->include_args             = _instance->include_args;
        _tmp->include_line             = _instance->include_line;
        _tmp->include_filename         = _instance->include_filename;
        _tmp->full_filepath            = _instance->full_filepath;
        _tmp->base_module_path         = _instance->base_module_path;
        _tmp->always_skipped_functions = _instance->always_skipped_functions;
        _tmp->always_skipped_filenames = _instance->always_skipped_filenames;
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
py::function
tracer_function(py::object pframe, const char* swhat, py::object arg)
{
    if(!tim::settings::enabled())
        return py::none{};

    if(user_trace_bundle::bundle_size() == 0)
    {
        if(tim::settings::debug())
            PRINT_HERE("%s", "Tracer bundle is empty");
        return py::none{};
    }

    int what = (strcmp(swhat, "line") == 0)
                   ? PyTrace_LINE
                   : (strcmp(swhat, "call") == 0)
                         ? PyTrace_CALL
                         : (strcmp(swhat, "return") == 0) ? PyTrace_RETURN : -1;

    // only support PyTrace_{LINE,CALL,RETURN}
    if(what < 0)
    {
        if(tim::settings::debug())
            PRINT_HERE("%s :: %s", "Ignoring what != {LINE,CALL,RETURN}", swhat);
        return py::none{};
    }

    auto* frame = reinterpret_cast<PyFrameObject*>(pframe.ptr());
    //
    static thread_local tracer_iterator_t _last;
    static thread_local bool              _has_last   = false;
    static thread_local auto              _file_lines = file_line_map_t{};
    static thread_local auto&             _config     = get_config();
    auto&                                 _mpath      = _config.base_module_path;
    auto&                                 _skip_funcs = _config.always_skipped_functions;
    auto&                                 _skip_files = _config.always_skipped_filenames;
    //
    if(_has_last)
        _has_last = (_last->second.stop(), false);
    //
    auto& _code = frame->f_code;
    auto& _line = frame->f_lineno;
    auto& _file = _code->co_filename;
    auto& _name = _code->co_name;
    // auto& _flno = _code->co_firstlineno;

    // get the function name
    auto _get_funcname = [&]() -> std::string { return py::cast<std::string>(_name); };

    // get the filename
    auto _get_filename = [&]() -> std::string { return py::cast<std::string>(_file); };

    // get the basename of the filename
    auto _get_basename = [&](const std::string& _fullpath) {
        if(_fullpath.find('/') != std::string::npos)
            return _fullpath.substr(_fullpath.find_last_of('/') + 1);
        return _fullpath;
    };

    static size_t _maxw      = 0;
    auto          _get_lines = [&](const auto& _fullpath) -> strvec_t& {
        auto litr = _file_lines.find(_fullpath);
        if(litr != _file_lines.end())
            return litr->second;
        auto os        = py::module::import("os.path");
        auto linecache = py::module::import("linecache");
        if(os.attr("exists")(_fullpath))
            linecache.attr("clearcache")();
        auto _lines = linecache.attr("getlines")(_fullpath).template cast<strvec_t>();
        for(size_t i = 0; i < _lines.size() - 1; ++i)
        {
            if(_lines.at(i).empty())
                continue;
            auto& itr = _lines.at(i);
            if(itr.find('@') == 0)
            {
                if(itr.find("@trace") == 0)
                    itr = _lines.at(i + 1);
            }
            for(auto c : { '\n', '\r' })
            {
                size_t pos = 0;
                while((pos = itr.find(c)) != std::string::npos)
                    itr = itr.replace(pos, 1, "");
            }
            _maxw = std::max<size_t>(_maxw, itr.length() + 1);
        }
        // auto inspect = py::module::import("inspect");
        litr = _file_lines.insert({ _fullpath, _lines }).first;
        return litr->second;
    };

    // get the arguments
    auto _get_args = [&]() {
        auto inspect = py::module::import("inspect");
        return py::cast<std::string>(
            inspect.attr("formatargvalues")(*inspect.attr("getargvalues")(pframe)));
    };

    // get the final filename
    auto _get_label = [&](auto& _func, auto& _filename, auto& _fullpath, auto& _flines) {
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
        {
            auto              _w = log10(_flines.size()) + 1;
            std::stringstream _sline;
            _sline.fill('0');
            _sline << std::setw(_w) << _line;
            _func = TIMEMORY_JOIN(':', _func, _sline.str());
        }
        return _func;
    };

    auto _func = _get_funcname();
    if(_skip_funcs.find(_func) != _skip_funcs.end())
    {
        auto _manager = tim::manager::instance();
        if(!_manager || _manager->is_finalized() || _func == "_shutdown")
        {
            auto sys       = py::module::import("sys");
            auto threading = py::module::import("threading");
            sys.attr("settrace")(py::none{});
            threading.attr("settrace")(py::none{});
        }
        return py::none{};
    }
    auto _full = _get_filename();
    auto _base = _get_basename(_full);

    if(!_config.include_internal &&
       strncmp(_full.c_str(), _mpath.c_str(), _mpath.length()) == 0)
        return py::none{};

    if(_skip_files.find(_base) != _skip_files.end() ||
       _skip_files.find(_full) != _skip_files.end())
        return py::none{};

    auto& _flines = _get_lines(_full);
    auto  _prefix = TIMEMORY_JOIN("", '[', _get_label(_func, _base, _full, _flines), ']');
    std::stringstream _slabel;
    _slabel << std::setw(_maxw) << std::left << _flines.at(_line - 1) << ' ' << _prefix;
    auto _label = _slabel.str();
    // auto _label = TIMEMORY_JOIN(' ', _prefix, _flines.at(_line - 1));

    DEBUG_PRINT_HERE("%8s | %s%s | %s", swhat, _func.c_str(), _get_args().c_str(),
                     _label.c_str());

    // get the iterator at the stack depth
    auto _get_code_itr = [&](bool _insert) {
        auto _itr = _config.records.find(_code);
        if(_insert && _itr == _config.records.end())
            _itr = _config.records.insert({ _code, tracer_line_map_t{} }).first;
        return _itr;
    };

    // get the stack-depth iterator for a label
    auto _get_line_itr = [&](bool _insert, auto itr, const auto& line) {
        auto _itr = itr->second.find(line);
        if(_insert)
        {
            if(_itr == itr->second.end())
            {
                _itr = itr->second
                           .insert({ line, tracer_t{ _label, true, tim::scope::flat{} } })
                           .first;
            }
            else
            {
                _itr->second.stop();
            }
        }
        return _itr;
    };

    // start function
    auto _tracer_call = [&]() {
        auto itr_code = _get_code_itr(true);
        auto itr_line = _get_line_itr(true, itr_code, _line);
        itr_line->second.start();
        _last     = itr_line;
        _has_last = true;
    };

    // stop function
    auto _tracer_return = [&]() {
        auto itr_code = _get_code_itr(false);
        if(itr_code == _config.records.end())
            return;
        auto itr_line = _get_line_itr(false, itr_code, _line);
        if(itr_line == itr_code->second.end())
            return;
        itr_line->second.stop();
    };

    // line function
    auto _tracer_line = [&]() {
        auto itr_code = _get_code_itr(true);
        auto itr_line = _get_line_itr(true, itr_code, _line);
        itr_line->second.start();
        _last     = itr_line;
        _has_last = true;
    };

    // process what
    switch(what)
    {
        case PyTrace_LINE: _tracer_line(); break;
        case PyTrace_CALL: _tracer_call(); break;
        case PyTrace_RETURN: _tracer_return(); break;
        default: break;
    }

    // don't do anything with arg
    tim::consume_parameters(arg);

    if(what == PyTrace_CALL)
        return py::cpp_function{ &tracer_function };
    return py::none{};
}
//
py::module
generate(py::module& _pymod)
{
    py::module _trace = _pymod.def_submodule(
        "trace", "Python tracing functions and C/C++/Fortran-compatible library "
                 "functions (subject to throttling)");

    pycomponent_bundle::generate<user_trace_bundle>(
        _trace, "trace_bundle", "User-bundle for Python tracing interface");

    _trace.def("init", &timemory_trace_init, "Initialize Tracing",
               py::arg("args") = "wall_clock", py::arg("read_command_line") = false,
               py::arg("cmd") = "");
    _trace.def("finalize", &timemory_trace_finalize, "Finalize Tracing");
    _trace.def("is_throttled", &timemory_is_throttled, "Check if key is throttled",
               py::arg("key"));
    _trace.def("push", &timemory_push_trace, "Push Trace", py::arg("key"));
    _trace.def("pop", &timemory_pop_trace, "Pop Trace", py::arg("key"));

    auto _init = []() {
        user_trace_bundle::global_init();
        auto _file = py::module::import("timemory").attr("__file__").cast<std::string>();
        if(_file.find('/') != std::string::npos)
            _file = _file.substr(0, _file.find_last_of('/'));
        get_config().base_module_path = _file;
        if(get_config().is_running)
            return;
        get_config().records.clear();
        get_config().is_running = true;
    };

    auto _fini = []() {
        if(!get_config().is_running)
            return;
        get_config().is_running = false;
        get_config().records.clear();
    };

    _trace.def("tracer_function", &tracer_function, "Tracing function");
    _trace.def("tracer_init", _init, "Initialize the tracer");
    _trace.def("tracer_finalize", _fini, "Finalize the tracer");

    py::class_<config> _pyconfig(_trace, "config", "Tracer configuration");

#define CONFIGURATION_PROPERTY(NAME, TYPE, DOC, ...)                                     \
    _pyconfig.def_property_static(NAME, [](py::object) { return __VA_ARGS__; },          \
                                  [](py::object, TYPE val) { __VA_ARGS__ = val; }, DOC);

    CONFIGURATION_PROPERTY("_is_running", bool, "Tracer is currently running",
                           get_config().is_running)
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
    CONFIGURATION_PROPERTY("skip_functions", strset_t,
                           "Function names to filter out of collection",
                           get_config().always_skipped_functions)
    CONFIGURATION_PROPERTY("skip_filenames", strset_t,
                           "Filenames to filter out of collection",
                           get_config().always_skipped_filenames)

    return _trace;
}
}  // namespace pytrace
//
//======================================================================================//
