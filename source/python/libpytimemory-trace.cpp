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
using string_t            = std::string;
using frame_object_t      = PyFrameObject;
using tracer_t            = tim::lightweight_tuple<user_trace_bundle>;
using tracer_line_map_t   = uomap_t<size_t, tracer_t>;
using tracer_code_map_t   = uomap_t<string_t, tracer_line_map_t>;
using tracer_iterator_t   = typename tracer_line_map_t::iterator;
using function_vec_t      = std::vector<tracer_iterator_t>;
using function_code_map_t = std::unordered_map<frame_object_t*, function_vec_t>;
using strset_t            = std::unordered_set<string_t>;
using strvec_t            = std::vector<string_t>;
using decor_line_map_t    = uomap_t<string_t, std::set<size_t>>;
using file_line_map_t     = uomap_t<string_t, strvec_t>;
//
struct config
{
    bool                is_running        = false;
    bool                include_internal  = false;
    bool                include_args      = false;
    bool                include_line      = true;
    bool                include_filename  = true;
    bool                full_filepath     = false;
    int32_t             max_stack_depth   = std::numeric_limits<uint16_t>::max();
    int32_t             base_stack_depth  = -1;
    string_t            base_module_path  = "";
    strset_t            include_functions = {};
    strset_t            include_filenames = {};
    strset_t            exclude_functions = { "FILE",      "FUNC",     "LINE",
                                   "get_fcode", "__exit__", "_handle_fromlist",
                                   "_shutdown", "isclass",  "isfunction",
                                   "basename",  "_get_sep" };
    strset_t            exclude_filenames = { "__init__.py",       "__main__.py",
                                   "functools.py",      "<frozen importlib._bootstrap>",
                                   "_pylab_helpers.py", "threading.py",
                                   "encoder.py",        "decoder.py" };
    tracer_code_map_t   records           = {};
    function_code_map_t functions         = {};
    tim::scope::config  tracer_scope      = tim::scope::config{ true, false, false };
    int32_t verbose = tim::settings::verbose() + ((tim::settings::debug()) ? 16 : 0);
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

        auto* _tmp              = new config{};
        _tmp->is_running        = _instance->is_running;
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
get_depth(frame_object_t* frame)
{
    return (frame->f_back) ? (get_depth(frame->f_back) + 1) : 0;
}
//
py::function
tracer_function(py::object pframe, const char* swhat, py::object arg)
{
    static thread_local auto& _config = get_config();

    if(!tim::settings::enabled())
        return py::none{};

    if(user_trace_bundle::bundle_size() == 0)
    {
        if(_config.verbose > 1)
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
        if(_config.verbose > 2)
            PRINT_HERE("%s :: %s", "Ignoring what != {LINE,CALL,RETURN}", swhat);
        return py::none{};
    }

    auto* frame = reinterpret_cast<frame_object_t*>(pframe.ptr());
    //
    using pushed_funcs_t = uomap_t<string_t, std::set<frame_object_t*>>;
    //
    static thread_local tracer_iterator_t _last         = {};
    static thread_local bool              _has_last     = false;
    static thread_local auto              _file_lines   = file_line_map_t{};
    static thread_local auto              _decor_lines  = decor_line_map_t{};
    static thread_local auto              _file_lskip   = strset_t{};
    static thread_local auto              _pushed_funcs = pushed_funcs_t{};
    auto&                                 _mpath        = _config.base_module_path;
    auto&                                 _only_funcs   = _config.include_functions;
    auto&                                 _only_files   = _config.include_filenames;
    auto&                                 _skip_funcs   = _config.exclude_functions;
    auto&                                 _skip_files   = _config.exclude_filenames;
    //
    if(_has_last)
    {
        // _last->second.stop().pop().push();
        _last->second.stop();
        _has_last = false;
    }
    //
    auto  _code = frame->f_code;
    auto  _line = frame->f_lineno;
    auto& _file = _code->co_filename;
    auto& _name = _code->co_name;

    // get the depth of the frame
    auto _fdepth = get_depth(frame);

    if(_config.base_stack_depth < 0)
        _config.base_stack_depth = _fdepth;

    bool    _iscall = (what == PyTrace_CALL);
    int32_t _sdepth = _fdepth - _config.base_stack_depth - 3;
    // if frame exceeds max stack-depth
    if(_iscall && _sdepth > _config.max_stack_depth)
    {
        if(_config.verbose > 1)
            PRINT_HERE("skipping %i > %i", (int) _sdepth, (int) _config.max_stack_depth);
        return py::none{};
    }

    // get the function name
    auto _get_funcname = [&]() -> string_t { return py::cast<string_t>(_name); };

    // get the filename
    auto _get_filename = [&]() -> string_t { return py::cast<string_t>(_file); };

    // get the basename of the filename
    auto _get_basename = [&](const string_t& _fullpath) {
        if(_fullpath.find('/') != string_t::npos)
            return _fullpath.substr(_fullpath.find_last_of('/') + 1);
        return _fullpath;
    };

    auto _sanitize_source_line = [](string_t& itr) {
        for(auto c : { '\n', '\r', '\t' })
        {
            size_t pos = 0;
            while((pos = itr.find(c)) != string_t::npos)
                itr = itr.replace(pos, 1, "");
        }
        return itr;
    };

    static size_t _maxw      = 0;
    auto          _get_lines = [&](const auto& _fullpath) -> strvec_t& {
        auto litr = _file_lines.find(_fullpath);
        if(litr != _file_lines.end())
            return litr->second;
        auto        linecache = py::module::import("linecache");
        static bool _once     = false;
        if(!_once)
            _once = (linecache.attr("clearcache")(), true);
        auto _lines = linecache.attr("getlines")(_fullpath).template cast<strvec_t>();
        for(size_t i = 0; i < _lines.size(); ++i)
        {
            auto& itr = _lines.at(i);
            _sanitize_source_line(itr);
            _maxw      = std::max<size_t>(_maxw, itr.length() + 1);
            auto _apos = itr.find_first_not_of(" \t");
            if(_apos != std::string::npos && i + 1 < _lines.size())
            {
                if(itr.at(_apos) == '@')
                {
                    auto& _ditr = _lines.at(i + 1);
                    if(_ditr.find("def ") == _apos)
                        _decor_lines[_fullpath].insert(i);
                }
            }
        }
        litr = _file_lines.insert({ _fullpath, _lines }).first;
        return litr->second;
    };

    // get the arguments
    auto _get_args = [&]() {
        auto inspect = py::module::import("inspect");
        return py::cast<string_t>(
            inspect.attr("formatargvalues")(*inspect.attr("getargvalues")(pframe)));
    };

    // get the final label
    auto _get_label = [&](auto _func, const auto& _filename, const auto& _fullpath,
                          const auto& _flines, auto _fline) {
        // append the arguments
        if(_config.include_args)
            _func = TIMEMORY_JOIN("", _func, _get_args());
        // append the filename
        if(_config.include_filename)
        {
            if(_config.full_filepath)
                _func = TIMEMORY_JOIN('/', _func, _fullpath);
            else
                _func = TIMEMORY_JOIN('/', _func, _filename);
        }
        // append the line number
        if(_config.include_line)
        {
            auto              _w = log10(_flines.size()) + 1;
            std::stringstream _sline;
            _sline.fill('0');
            _sline << std::setw(_w) << _fline;
            _func = TIMEMORY_JOIN(':', _func, _sline.str());
        }
        return _func;
    };

    auto _func = _get_funcname();

    if(!_only_funcs.empty() && _only_funcs.find(_func) == _only_funcs.end())
    {
        if(_config.verbose > 1)
            PRINT_HERE("Skipping non-included function: %s", _func.c_str());
        return py::none{};
    }

    if(_skip_funcs.find(_func) != _skip_funcs.end())
    {
        if(_config.verbose > 1)
            PRINT_HERE("Skipping designated function: '%s'", _func.c_str());
        auto _manager = tim::manager::instance();
        if(!_manager || _manager->is_finalized() || _func == "_shutdown")
        {
            if(_config.verbose > 1)
                PRINT_HERE("Shutdown detected: %s", _func.c_str());
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
    {
        if(_config.verbose > 2)
            PRINT_HERE("Skipping internal function: %s", _func.c_str());
        return py::none{};
    }

    if(!_only_files.empty() && (_only_files.find(_base) == _only_files.end() &&
                                _only_files.find(_full) == _only_files.end()))
    {
#if defined(DEBUG)
        if(tim::settings::debug())
        {
            std::stringstream _opts;
            for(const auto& itr : _only_files)
                _opts << "| " << itr;
            PRINT_HERE("Skipping: [%s | %s | %s] due to [%s]", _func.c_str(),
                       _base.c_str(), _full.c_str(), _opts.str().substr(2).c_str());
        }
#else
        if(_config.verbose > 2)
            PRINT_HERE("Skipping non-included file: %s", _base.c_str());
#endif
        return py::none{};
    }

    if(_skip_files.find(_base) != _skip_files.end() ||
       _skip_files.find(_full) != _skip_files.end())
    {
        if(_config.verbose > 1)
            PRINT_HERE("Skipping designated file: '%s'", _base.c_str());
        return py::none{};
    }

    strvec_t* _plines = nullptr;
    try
    {
        if(_file_lskip.count(_full) == 0)
            _plines = &_get_lines(_full);
    } catch(std::exception& e)
    {
        if(_config.verbose > -1)
            PRINT_HERE("Exception thrown when retrieving lines for file '%s'. Functions "
                       "in this file will not be traced:\n%s",
                       _full.c_str(), e.what());
        _file_lskip.insert(_full);
        return py::none{};
    }

    if(!_plines)
    {
        if(_config.verbose > 3)
            PRINT_HERE("No source code lines for '%s'. Returning", _full.c_str());
        return py::none{};
    }

    auto& _flines = *_plines;

    //----------------------------------------------------------------------------------//
    //
    auto _get_trace_lines = [&]() -> tracer_line_map_t& {
        auto titr = _config.records.find(_full);
        if(titr != _config.records.end())
            return titr->second;
        auto _tvec = tracer_line_map_t{};
        _tvec.reserve(_flines.size());
        for(size_t i = 0; i < _flines.size(); ++i)
        {
            auto&             itr     = _flines.at(i);
            auto              _flabel = _get_label(_func, _base, _full, _flines, i + 1);
            auto              _prefix = TIMEMORY_JOIN("", '[', _flabel, ']');
            std::stringstream _slabel;
            _slabel << std::setw(_maxw) << std::left << itr;
            int64_t _rem = tim::settings::max_width();
            // three spaces for '>>>'
            _rem -= _maxw + _prefix.length() + 3;
            // space for 'XY|' for thread
            _rem -= (tim::settings::collapse_threads()) ? 3 : 0;
            // space for 'XY|' for process
            _rem -= (tim::settings::collapse_processes()) ? 3 : 0;
            // make sure not less than zero
            _rem = std::max<int64_t>(_rem, 0);
            _slabel << std::setw(_rem) << std::right << "" << ' ' << _prefix;
            auto _label = _slabel.str();
            DEBUG_PRINT_HERE("%8s | %s%s | %s", swhat, _func.c_str(), _get_args().c_str(),
                             _label.c_str());
            // create a tracer for the line
            _tvec.emplace(i, tracer_t{ _label, _config.tracer_scope });
        }
        titr = _config.records.emplace(_full, _tvec).first;
        return titr->second;
    };

    // the tracers for the fullpath
    auto& _tlines = _get_trace_lines();

    //----------------------------------------------------------------------------------//
    // the first time a frame is encountered, use the inspect module to process the
    // source lines. Essentially, this function finds all the source code lines in
    // the frame, generates an label via the source code, and then "pushes" that
    // label into the storage. Then we are free to call start/stop repeatedly
    // and only when the pop is applied does the storage instance get updated.
    // NOTE: this means the statistics are not correct.
    //
    auto _push_tracer = [&](auto object) {
        if(_pushed_funcs[_full].count(frame) == 0)
        {
            auto inspect = py::module::import("inspect");
            try
            {
                py::object srclines = py::none{};
                try
                {
                    srclines = inspect.attr("getsourcelines")(object);
                } catch(std::exception& e)
                {
                    if(tim::settings::debug())
                        std::cerr << e.what() << std::endl;
                }
                if(!srclines.is_none())
                {
                    auto _get_docstring = [](const string_t& _str, size_t _pos) {
                        auto _q1 = _str.find("'''", _pos);
                        auto _q2 = _str.find("\"\"\"", _pos);
                        return std::min<size_t>(_q1, _q2);
                    };

                    auto pysrclist       = srclines.cast<py::list>()[0].cast<py::list>();
                    auto _srclines       = strvec_t{};
                    auto _skip_docstring = false;
                    for(auto itr : pysrclist)
                    {
                        auto sline = itr.cast<std::string>();
                        _sanitize_source_line(sline);
                        if(sline.empty())
                            continue;
                        auto _pos = sline.find_first_not_of(" \t");
                        if(_pos < sline.length() && sline[_pos] == '#')
                            continue;
                        // if we are not currently inside a doc-string, search for
                        // it as the first non-whitespace character
                        if(!_skip_docstring)
                        {
                            auto _dbeg = _get_docstring(sline, _pos);
                            if(_dbeg == _pos)
                            {
                                DEBUG_PRINT_HERE("Doc-string detected in: %s",
                                                 sline.c_str());
                                // check if the doc-string is terminated in the same line
                                auto _dend = _get_docstring(sline, _dbeg + 3);
                                if(_dend == std::string::npos)
                                    _skip_docstring = true;
                                else
                                {
                                    DEBUG_PRINT_HERE("Doc-string terminated in: %s",
                                                     sline.c_str());
                                }
                                // skip this line bc there is a doc-string
                                continue;
                            }
                        }
                        // if currently skipping, look for the terminating doc-string
                        if(_skip_docstring)
                        {
                            auto _chk = _get_docstring(sline, 0);
                            if(_chk != std::string::npos)
                            {
                                DEBUG_PRINT_HERE("Doc-string terminated in: %s",
                                                 sline.c_str());
                                _skip_docstring = false;
                                // if the doc-string is the last set of characters
                                // or if there is nothing but spaces/tabs/quotes/etc.
                                // after the doc-string, skip this line
                                if(_chk + 4 >= sline.length() ||
                                   sline.find_first_not_of(" \t\n\r'\"") ==
                                       std::string::npos)
                                    continue;
                            }
                        }
                        if(_skip_docstring)
                            continue;
                        // only add if not in doc-string
                        _srclines.emplace_back(sline);
                    }
                    //
                    if(tim::settings::debug())
                    {
                        std::cout << "\nSource lines:\n";
                        for(const auto& itr : _srclines)
                            std::cout << "    " << itr << '\n';
                        std::cout << std::endl;
                    }
                    //
                    auto ibeg = (_line == 0) ? _line : _line - 1;
                    auto iend = std::min<size_t>(_tlines.size(), ibeg + pysrclist.size());
                    for(size_t i = ibeg; i < iend; ++i)
                    {
                        auto& _tracer = _tlines.at(i);
                        for(auto& sitr : _srclines)
                        {
                            if(_tracer.key().find(sitr) != std::string::npos)
                            {
                                _tracer.push();
                                break;
                            }
                        }
                    }
                }
            } catch(py::cast_error& e)
            {
                std::cerr << e.what() << std::endl;
            }
        }
    };

    //----------------------------------------------------------------------------------//
    //
    _push_tracer(pframe);
    _pushed_funcs[_full].insert(frame);

    //----------------------------------------------------------------------------------//
    // start function
    //
    auto _tracer_call = [&]() {
        // auto itr = _tlines.find(_line - 1);
        // if(itr == _tlines.end())
        //     return;
        // auto& _entry = _config.functions[frame];
        // if(_entry.empty())
        // {
        //     // if(_decor_lines[_full].count(_line - 1) > 0)
        //     //     ++itr;
        //     itr->second.push();
        //     itr->second.start();
        // }
        // _entry.emplace_back(itr);
    };

    //----------------------------------------------------------------------------------//
    // stop function
    //
    auto _tracer_return = [&]() {
        // auto fitr = _config.functions.find(frame);
        // if(fitr == _config.functions.end() || fitr->second.empty())
        //     return;
        // auto itr = fitr->second.back();
        // fitr->second.pop_back();
        // if(fitr->second.empty())
        //     itr->second.stop();
    };

    //----------------------------------------------------------------------------------//
    // line function
    //
    auto _tracer_line = [&]() {
        auto itr = _tlines.find(_line - 1);
        if(itr == _tlines.end())
            return;
        itr->second.start();
        _has_last = true;
        _last     = itr;
    };

    //----------------------------------------------------------------------------------//
    // process what
    //
    switch(what)
    {
        case PyTrace_LINE: _tracer_line(); break;
        case PyTrace_CALL: _tracer_call(); break;
        case PyTrace_RETURN: _tracer_return(); break;
        default: break;
    }

    // don't do anything with arg
    tim::consume_parameters(arg);

    if(_config.verbose > 3)
        PRINT_HERE("Returning trace function for %s of '%s'", swhat, _func.c_str());

    return py::cpp_function{ &tracer_function };
}
//
py::module
generate(py::module& _pymod)
{
    py::module _trace = _pymod.def_submodule(
        "trace", "Python tracing functions and C/C++/Fortran-compatible library "
                 "functions (subject to throttling)");

    py::class_<PyCodeObject>  _code_object(_pymod, "code_object", "PyCodeObject");
    py::class_<PyFrameObject> _frame_object(_pymod, "frame_object", "PyFrameObject");

    tim::consume_parameters(_code_object, _frame_object);

    static auto _set_scope = [](bool _flat, bool _timeline) {
        get_config().tracer_scope = tim::scope::config{ _flat, _timeline };
    };

    pycomponent_bundle::generate<user_trace_bundle>(
        _trace, "trace_bundle", "User-bundle for Python tracing interface", _set_scope);

    _trace.def("init", &timemory_trace_init, "Initialize Tracing",
               py::arg("args") = "wall_clock", py::arg("read_command_line") = false,
               py::arg("cmd") = "");
    _trace.def("finalize", &timemory_trace_finalize, "Finalize Tracing");
    _trace.def("is_throttled", &timemory_is_throttled, "Check if key is throttled",
               py::arg("key"));
    _trace.def("push", &timemory_push_trace, "Push Trace", py::arg("key"));
    _trace.def("pop", &timemory_pop_trace, "Pop Trace", py::arg("key"));

    auto _init = []() {
        auto _verbose = get_config().verbose;
        CONDITIONAL_PRINT_HERE(_verbose > 1, "%s", "Initializing trace");
        try
        {
            auto _file = py::module::import("timemory").attr("__file__").cast<string_t>();
            if(_file.find('/') != string_t::npos)
                _file = _file.substr(0, _file.find_last_of('/'));
            get_config().base_module_path = _file;
        } catch(py::cast_error& e)
        {
            std::cerr << "[trace_init]> " << e.what() << std::endl;
        }

        if(get_config().is_running)
        {
            CONDITIONAL_PRINT_HERE(_verbose > 1, "%s", "Trace already running");
            return;
        }
        CONDITIONAL_PRINT_HERE(_verbose < 2 && _verbose > 0, "%s", "Initializing trace");
        CONDITIONAL_PRINT_HERE(_verbose > 0, "%s",
                               "Resetting trace state for initialization");
        get_config().records.clear();
        get_config().functions.clear();
        get_config().is_running = true;
    };

    auto _fini = []() {
        auto _verbose = get_config().verbose;
        CONDITIONAL_PRINT_HERE(_verbose > 2, "%s", "Finalizing trace");
        if(!get_config().is_running)
        {
            CONDITIONAL_PRINT_HERE(_verbose > 2, "%s", "Trace already finalized");
            return;
        }
        CONDITIONAL_PRINT_HERE(_verbose > 0 && _verbose < 3, "%s", "Finalizing trace");
        get_config().is_running = false;
        CONDITIONAL_PRINT_HERE(_verbose > 1, "%s", "Popping records from call-stack");
        for(auto& ritr : get_config().records)
        {
            for(auto& itr : ritr.second)
                itr.second.pop();
        }
        CONDITIONAL_PRINT_HERE(_verbose > 1, "%s", "Destroying records");
        get_config().records.clear();
        get_config().functions.clear();
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

    tim::operation::init<user_trace_bundle>(
        tim::operation::mode_constant<tim::operation::init_mode::global>{});

    return _trace;
}
}  // namespace pytrace
//
//======================================================================================//
