//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#pragma once

//======================================================================================//

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "pybind11/cast.h"
#include "pybind11/chrono.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

#include "timemory/auto_list.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/auto_tuple.hpp"
#include "timemory/component_list.hpp"
#include "timemory/component_tuple.hpp"
#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/mpi.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/timemory.hpp"

//======================================================================================//

extern "C"
{
#include "timemory/ctimemory.h"
}

namespace py = pybind11;
using namespace std::placeholders;  // for _1, _2, _3...
using namespace py::literals;
using namespace tim::component;

using auto_timer_t =
    tim::auto_tuple<wall_clock, system_clock, user_clock, cpu_clock, cpu_util>;
using auto_usage_t =
    tim::auto_tuple<current_rss, peak_rss, num_minor_page_faults, num_major_page_faults,
                    voluntary_context_switch, priority_context_switch>;
using auto_list_t =
    tim::auto_list<real_clock, system_clock, user_clock, cpu_clock, monotonic_clock,
                   monotonic_raw_clock, thread_cpu_clock, process_cpu_clock, cpu_util,
                   thread_cpu_util, process_cpu_util, current_rss, peak_rss, stack_rss,
                   data_rss, num_swap, num_io_in, num_io_out, num_minor_page_faults,
                   num_major_page_faults, num_msg_sent, num_msg_recv, num_signals,
                   voluntary_context_switch, priority_context_switch, cuda_event>;

using tim_timer_t       = typename auto_timer_t::component_type;
using rss_usage_t       = typename auto_usage_t::component_type;
using component_list_t  = typename auto_list_t::component_type;
using manager_t         = tim::manager;
using sys_signal_t      = tim::sys_signal;
using signal_settings_t = tim::signal_settings;
using signal_set_t      = signal_settings_t::signal_set_t;
using farray_t          = py::array_t<double, py::array::c_style | py::array::forcecast>;

using component_enum_vec = std::vector<TIMEMORY_COMPONENT>;

//======================================================================================//

class manager_wrapper
{
public:
    manager_wrapper()
    : m_manager(manager_t::instance())
    {
    }

    ~manager_wrapper() {}

    // ensures thread-local version is called
    manager_t* get() { return manager_t::instance(); }

protected:
    manager_t* m_manager;
};

//======================================================================================//

class auto_timer_decorator
{
public:
    auto_timer_decorator(auto_timer_t* _ptr = nullptr)
    : m_ptr(_ptr)
    {
    }

    ~auto_timer_decorator() { delete m_ptr; }

    auto_timer_decorator& operator=(auto_timer_t* _ptr)
    {
        if(m_ptr)
            delete m_ptr;
        m_ptr = _ptr;
        return *this;
    }

private:
    auto_timer_t* m_ptr;
};

//======================================================================================//

class component_list_decorator
{
public:
    component_list_decorator(component_list_t* _ptr = nullptr)
    : m_ptr(_ptr)
    {
        if(m_ptr)
        {
            m_ptr->push();
            m_ptr->start();
        }
    }

    ~component_list_decorator()
    {
        if(m_ptr)
            m_ptr->stop();
        delete m_ptr;
    }

    component_list_decorator& operator=(component_list_t* _ptr)
    {
        if(m_ptr)
            m_ptr->stop();
        delete m_ptr;
        m_ptr = _ptr;
        if(m_ptr)
        {
            m_ptr->push();
            m_ptr->start();
        }
        return *this;
    }

private:
    component_list_t* m_ptr;
};

//======================================================================================//

#if _PYTHON_MAJOR_VERSION > 2
#    define PYOBJECT_SELF
#    define PYOBJECT_SELF_PARAM
#else
#    define PYOBJECT_SELF py::object,
#    define PYOBJECT_SELF_PARAM py::object
#endif

//======================================================================================//

namespace pytim
{
//======================================================================================//

using string_t = std::string;

//======================================================================================//
//
//                          TiMemory (general)
//
//======================================================================================//

int
get_line(int nback = 1)
{
    auto locals = py::dict("back"_a = nback);
    py::exec(R"(
             import sys
             result = int(sys._getframe(back).f_lineno)
             )",
             py::globals(), locals);
    auto ret = locals["result"].cast<int>();
    return ret;
}

//--------------------------------------------------------------------------------------//

string_t
get_func(int nback = 1)
{
    auto locals = py::dict("back"_a = nback);
    py::exec(R"(
             import sys
             result = ("{}".format(sys._getframe(back).f_code.co_name))
             )",
             py::globals(), locals);
    auto ret = locals["result"].cast<std::string>();
    return ret;
}

//--------------------------------------------------------------------------------------//

string_t
get_file(int nback = 2, bool only_basename = true, bool use_dirname = false,
         bool noquotes = false)
{
    auto locals = py::dict("back"_a = nback, "only_basename"_a = only_basename,
                           "use_dirname"_a = use_dirname, "noquotes"_a = noquotes);
    py::exec(R"(
             import sys
             import os
             from os.path import dirname
             from os.path import basename
             from os.path import join

             def get_fcode(back):
                 fname = '<module'
                 try:
                     fname = sys._getframe(back).f_code.co_filename
                 except:
                     fname = '<module>'
                 return fname

             result = None
             if only_basename:
                 if use_dirname:
                     result = ("{}".format(join(basename(dirname(get_fcode(back))),
                       basename(get_fcode(back)))))
                 else:
                     result = ("{}".format(basename(get_fcode(back))))
             else:
                 result = ("{}".format(get_fcode(back)))

             if noquotes is False:
                 result = ("'{}'".format(result))
             )",
             py::globals(), locals);
    auto ret = locals["result"].cast<std::string>();
    return ret;
}

//--------------------------------------------------------------------------------------//

component_enum_vec
components_enum_to_vec(py::list enum_list)
{
    component_enum_vec vec;
    for(auto itr : enum_list)
        vec.push_back(itr.cast<TIMEMORY_COMPONENT>());
    return vec;
}

//--------------------------------------------------------------------------------------//

component_list_t*
create_component_list(std::string obj_tag, int lineno, const tim::language& lang,
                      bool report, const component_enum_vec& components)
{
    auto obj = new component_list_t(obj_tag, lineno, lang, report);
    for(std::size_t i = 0; i < components.size(); ++i)
    {
        TIMEMORY_COMPONENT component = static_cast<TIMEMORY_COMPONENT>(components[i]);
        switch(component)
        {
            case WALL_CLOCK: obj->get<real_clock>() = new real_clock(); break;
            case SYS_CLOCK: obj->get<system_clock>() = new system_clock(); break;
            case USER_CLOCK: obj->get<user_clock>() = new user_clock(); break;
            case CPU_CLOCK: obj->get<cpu_clock>() = new cpu_clock(); break;
            case MONOTONIC_CLOCK:
                obj->get<monotonic_clock>() = new monotonic_clock();
                break;
            case MONOTONIC_RAW_CLOCK:
                obj->get<monotonic_raw_clock>() = new monotonic_raw_clock();
                break;
            case THREAD_CPU_CLOCK:
                obj->get<thread_cpu_clock>() = new thread_cpu_clock();
                break;
            case PROCESS_CPU_CLOCK:
                obj->get<process_cpu_clock>() = new process_cpu_clock();
                break;
            case CPU_UTIL: obj->get<cpu_util>() = new cpu_util(); break;
            case THREAD_CPU_UTIL:
                obj->get<thread_cpu_util>() = new thread_cpu_util();
                break;
            case PROCESS_CPU_UTIL:
                obj->get<process_cpu_util>() = new process_cpu_util();
                break;
            case CURRENT_RSS: obj->get<current_rss>() = new current_rss(); break;
            case PEAK_RSS: obj->get<peak_rss>() = new peak_rss(); break;
            case STACK_RSS: obj->get<stack_rss>() = new stack_rss(); break;
            case DATA_RSS: obj->get<data_rss>() = new data_rss(); break;
            case NUM_SWAP: obj->get<num_swap>() = new num_swap(); break;
            case NUM_IO_IN: obj->get<num_io_in>() = new num_io_in(); break;
            case NUM_IO_OUT: obj->get<num_io_out>() = new num_io_out(); break;
            case NUM_MINOR_PAGE_FAULTS:
                obj->get<num_minor_page_faults>() = new num_minor_page_faults();
                break;
            case NUM_MAJOR_PAGE_FAULTS:
                obj->get<num_major_page_faults>() = new num_major_page_faults();
                break;
            case NUM_MSG_SENT: obj->get<num_msg_sent>() = new num_msg_sent(); break;
            case NUM_MSG_RECV: obj->get<num_msg_recv>() = new num_msg_recv(); break;
            case NUM_SIGNALS: obj->get<num_signals>() = new num_signals(); break;
            case VOLUNTARY_CONTEXT_SWITCH:
                obj->get<voluntary_context_switch>() = new voluntary_context_switch();
                break;
            case PRIORITY_CONTEXT_SWITCH:
                obj->get<priority_context_switch>() = new priority_context_switch();
                break;
            case CUDA_EVENT:
#if defined(TIMEMORY_USE_CUDA)
                obj->get<cuda_event>() = new cuda_event();
#endif
                break;
        }
    }
    return obj;
}

//--------------------------------------------------------------------------------------//

signal_set_t
signal_list_to_set(py::list signal_list)
{
    signal_set_t signal_set;
    for(auto itr : signal_list)
        signal_set.insert(itr.cast<sys_signal_t>());
    return signal_set;
}

//--------------------------------------------------------------------------------------//

signal_set_t
get_default_signal_set()
{
    return tim::signal_settings::enabled();
}

//--------------------------------------------------------------------------------------//

void
enable_signal_detection(py::list signal_list = py::list())
{
    auto _sig_set = (signal_list.size() == 0) ? get_default_signal_set()
                                              : signal_list_to_set(signal_list);
    tim::enable_signal_detection(_sig_set);
}

//--------------------------------------------------------------------------------------//

void
disable_signal_detection()
{
    tim::disable_signal_detection();
}

//--------------------------------------------------------------------------------------//

//======================================================================================//
//
//                              INITITALIZATION
//
//======================================================================================//

namespace init
{
//--------------------------------------------------------------------------------------//

manager_wrapper*
manager()
{
    return new manager_wrapper();
}

//--------------------------------------------------------------------------------------//

tim_timer_t*
timer(std::string prefix = "")
{
    if(prefix.empty())
    {
        std::stringstream keyss;
        keyss << get_func(1) << "@" << get_file(2) << ":" << get_line(1);
        prefix = keyss.str();
    }

    auto op_line = get_line(1);
    return new tim_timer_t(prefix, op_line, tim::language::pyc());
}

//--------------------------------------------------------------------------------------//

auto_timer_t*
auto_timer(const std::string& key, bool report_at_exit, int nback, bool added_args,
           py::args args, py::kwargs kwargs)
{
    tim::consume_parameters(args, kwargs);
    std::stringstream keyss;
    keyss << get_func(nback);

    if(added_args)
        keyss << key;
    else if(key != "" && key[0] != '@' && !added_args)
        keyss << "@";

    if(key != "" && !added_args)
        keyss << key;
    else
    {
        keyss << "@";
        keyss << get_file(nback + 1);
        keyss << ":";
        keyss << get_line(nback);
    }
    auto op_line = get_line(1);
    return new auto_timer_t(keyss.str(), op_line, tim::language::pyc(), report_at_exit);
}

//--------------------------------------------------------------------------------------//

rss_usage_t*
rss_usage(std::string prefix = "", bool record = false)
{
    if(prefix.empty())
    {
        std::stringstream keyss;
        keyss << get_func(1) << "@" << get_file(2) << ":" << get_line(1);
        prefix = keyss.str();
    }
    auto         op_line = get_line(1);
    rss_usage_t* _rss    = new rss_usage_t(prefix, op_line, tim::language::pyc());
    if(record)
        _rss->measure();
    return _rss;
}

//--------------------------------------------------------------------------------------//

component_list_t*
component_list(py::list components, const std::string& key, bool report_at_exit,
               int nback, bool added_args, py::args args, py::kwargs kwargs)
{
    tim::consume_parameters(args, kwargs);
    std::stringstream keyss;
    keyss << get_func(nback);

    if(added_args)
        keyss << key;
    else if(key != "" && key[0] != '@' && !added_args)
        keyss << "@";

    if(key != "" && !added_args)
        keyss << key;
    else
    {
        keyss << "@";
        keyss << get_file(nback + 1);
        keyss << ":";
        keyss << get_line(nback);
    }
    auto op_line = get_line(1);
    return create_component_list(keyss.str(), op_line, tim::language::pyc(),
                                 report_at_exit, components_enum_to_vec(components));
}

//----------------------------------------------------------------------------//

auto_timer_decorator*
timer_decorator(const std::string& func, const std::string& file, int line,
                const std::string& key, bool added_args, bool report_at_exit)
{
    auto_timer_decorator* _ptr = new auto_timer_decorator();
    if(!auto_timer_t::is_enabled())
        return _ptr;

    std::stringstream keyss;
    if(func != "<module>")
        keyss << func;

    // add arguments to end of function
    if(added_args)
        keyss << key;
    else if(func != "<module>" && key != "" && key[0] != '@' && !added_args)
        keyss << "@";

    if(key != "" && !added_args)
        keyss << key;
    else
    {
        keyss << "@";
        keyss << file;
        keyss << ":";
        keyss << line;
    }
    return &(*_ptr = new auto_timer_t(keyss.str(), line, tim::language::pyc(),
                                      report_at_exit));
}

//----------------------------------------------------------------------------//

component_list_decorator*
component_decorator(py::list components, const std::string& func, const std::string& file,
                    int line, const std::string& key, bool added_args,
                    bool report_at_exit)
{
    component_list_decorator* _ptr = new component_list_decorator();
    if(!manager_t::is_enabled())
        return _ptr;

    std::stringstream keyss;
    keyss << func;

    // add arguments to end of function
    if(added_args)
        keyss << key;
    else if(key != "" && key[0] != '@' && !added_args)
        keyss << "@";

    if(key != "" && !added_args)
        keyss << key;
    else
    {
        keyss << "@";
        keyss << file;
        keyss << ":";
        keyss << line;
    }
    return &(*_ptr = create_component_list(keyss.str(), line, tim::language::pyc(),
                                           report_at_exit,
                                           components_enum_to_vec(components)));
}

//--------------------------------------------------------------------------------------//

}  // namespace init

//======================================================================================//
//
//                              MANAGER
//
//======================================================================================//

namespace manager
{
//--------------------------------------------------------------------------------------//

string_t
write_ctest_notes(py::object man, std::string directory, bool append)
{
    py::list filenames = man.attr("text_files").cast<py::list>();

    std::stringstream ss;
    ss << std::endl;
    ss << "IF(NOT DEFINED CTEST_NOTES_FILES)" << std::endl;
    ss << "    SET(CTEST_NOTES_FILES )" << std::endl;
    ss << "ENDIF(NOT DEFINED CTEST_NOTES_FILES)" << std::endl;
    ss << std::endl;

    // loop over ASCII report filenames
    for(const auto& itr : filenames)
    {
        std::string fname = itr.cast<std::string>();
#if defined(_WIN32)
        while(fname.find("\\") != std::string::npos)
            fname = fname.replace(fname.find("\\"), 1, "/");
#endif
        ss << "LIST(APPEND CTEST_NOTES_FILES \"" << fname << "\")" << std::endl;
    }

    ss << std::endl;
    ss << "IF(NOT \"${CTEST_NOTES_FILES}\" STREQUAL \"\")" << std::endl;
    ss << "    LIST(REMOVE_DUPLICATES CTEST_NOTES_FILES)" << std::endl;
    ss << "ENDIF(NOT \"${CTEST_NOTES_FILES}\" STREQUAL \"\")" << std::endl;
    ss << std::endl;

    // create directory (easier in Python)
    auto locals = py::dict("directory"_a = directory);
    py::exec(R"(
             import os

             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)

             file_path = os.path.join(directory, "CTestNotes.cmake")
             )",
             py::globals(), locals);

    std::string   file_path = locals["file_path"].cast<std::string>();
    std::ofstream outf;
    if(append)
        outf.open(file_path.c_str(), std::ios::app);
    else
        outf.open(file_path.c_str());

    if(outf)
        outf << ss.str();
    outf.close();

    return file_path;
}

//--------------------------------------------------------------------------------------//

}  // namespace manager

//======================================================================================//
//
//                          OPTIONS
//
//======================================================================================//

namespace opt
{
//--------------------------------------------------------------------------------------//

void
safe_mkdir(string_t directory)
{
    auto locals = py::dict("directory"_a = directory);
    py::exec(R"(
             import os
             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)
             )",
             py::globals(), locals);
}

//--------------------------------------------------------------------------------------//

void
ensure_directory_exists(string_t file_path)
{
    auto locals = py::dict("file_path"_a = file_path);
    py::exec(R"(
             import os

             directory = os.path.dirname(file_path)
             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)
             )",
             py::globals(), locals);
}

//--------------------------------------------------------------------------------------//

py::object
add_arguments(py::object parser = py::none(), std::string fpath = ".")
{
    if(fpath == ".")
        fpath = tim::settings::output_path();

    auto locals = py::dict("parser"_a = parser, "fpath"_a = fpath);
    py::exec(R"(
             import sys
             import os
             from os.path import join
             import argparse
             import timemory

             if parser is None:
                parser = argparse.ArgumentParser()

             parser.add_argument('--output-path', required=False,
                                 default=fpath, type=str, help="Output directory")

             parser.add_argument('--output-prefix', required=False,
                                 default="", type=str,
                                 help="Filename prefix without path")

             parser.add_argument('--disable-timers', required=False,
                                 action='store_false',
                                 dest='use_timers',
                                 help="Disable timers for script")

             parser.add_argument('--enable-timers', required=False,
                                 action='store_true',
                                 dest='use_timers', help="Enable timers for script")

             parser.add_argument('--disable-timer-serialization',
                                 required=False, action='store_false',
                                 dest='serial_file',
                                 help="Disable serialization for timers")

             parser.add_argument('--enable-timer-serialization',
                                 required=False, action='store_true',
                                 dest='serial_file',
                                 help="Enable serialization for timers")

             parser.add_argument('--max-timer-depth',
                                 help="Maximum timer depth",
                                 type=int,
                                 default=timemory.options.default_max_depth())

             parser.add_argument('--enable-dart',
                                 help="Print DartMeasurementFile tag for plots",
                                 required=False, action='store_true')

             parser.add_argument('--write-ctest-notes',
                                 help="Write a CTestNotes.cmake file for TiMemory ASCII output",
                                 required=False, action='store_true')

             parser.set_defaults(use_timers=True)
             parser.set_defaults(serial_file=True)
             parser.set_defaults(enable_dart=False)
             parser.set_defaults(write_ctest_notes=False)
             )",
             py::globals(), locals);
    return locals["parser"].cast<py::object>();
}

//--------------------------------------------------------------------------------------//

void
parse_args(py::object args)
{
    auto locals = py::dict("args"_a = args);
    py::exec(R"(
             import sys
             import os
             import timemory

             # Function to add default output arguments
             timemory.options.use_timers = args.use_timers
             timemory.options.max_timer_depth = args.max_timer_depth
             timemory.options.echo_dart = args.enable_dart
             timemory.options.ctest_notes = args.write_ctest_notes
             timemory.options.output_path = args.output_path
             timemory.options.output_prefix = args.output_prefix
             timemory.toggle(args.use_timers)
             timemory.set_max_depth(args.max_timer_depth)

             _enable_serial = args.serial_file
             )",
             py::globals(), locals);
    tim::settings::json_output() = locals["_enable_serial"].cast<bool>();
}

//--------------------------------------------------------------------------------------//

py::object
add_arguments_and_parse(py::object parser = py::none(), std::string fpath = "")
{
    auto locals = py::dict("parser"_a = parser, "fpath"_a = fpath);
    py::exec(R"(
             import timemory

             # Combination of timemory.add_arguments and timemory.parse_args but returns
             parser = timemory.options.add_arguments(parser, fpath)
             args = parser.parse_args()
             timemory.options.parse_args(args)
             )",
             py::globals(), locals);
    return locals["args"].cast<py::object>();
}

//--------------------------------------------------------------------------------------//

py::object
add_args_and_parse_known(py::object parser = py::none(), std::string fpath = "")
{
    auto locals = py::dict("parser"_a = parser, "fpath"_a = fpath);
    py::exec(R"(
             import timemory

             # Combination of timing.add_arguments and timing.parse_args but returns
             parser = timemory.options.add_arguments(parser, fpath)
             args, left = parser.parse_known_args()
             timemory.options.parse_args(args)
             # replace sys.argv with unknown args only
             sys.argv = sys.argv[:1]+left
             )",
             py::globals(), locals);
    return locals["args"].cast<py::object>();
}

//--------------------------------------------------------------------------------------//

}  // namespace opt

//======================================================================================//

namespace decorators
{
class base_decorator
{
public:
    base_decorator() {}

    base_decorator(std::string key, bool add_args, bool is_class)
    : m_add_args(add_args)
    , m_is_class(is_class)
    , m_key(key)
    {
    }

    void parse_wrapped(py::function func, py::args args, py::kwargs kwargs)
    {
        auto locals = py::dict("func"_a = func, "args"_a = args, "kwargs"_a = kwargs);
        py::exec(R"(
                 is_class = False
                 if len(args) > 0 and args[0] is not None and inspect.isclass(type(args[0])):
                     is_class = True
                 )",
                 py::globals(), locals);
        m_is_class = locals["is_class"].cast<bool>();
    }

    std::string class_string(py::args args, py::kwargs kwargs)
    {
        auto locals = py::dict("args"_a = args, "kwargs"_a = kwargs, "_key"_a = m_key,
                               "_is_class"_a = m_is_class);
        py::exec(R"(
                 _str = ''
                 if _is_class and len(args) > 0 and args[0] is not None:
                     _str = '[{}]'.format(type(args[0]).__name__)

                     # this check guards against class methods calling class methods
                     if _str in _key:
                         _str = ''
                 )",
                 py::globals(), locals);
        return locals["_str"].cast<std::string>();
    }

    std::string arg_string(py::args args, py::kwargs kwargs)
    {
        auto _str   = class_string(args, kwargs);
        auto locals = py::dict("_str"_a = _str, "args"_a = args, "kwargs"_a = kwargs,
                               "_key"_a = m_key, "_is_class"_a = m_is_class,
                               "_add_args"_a = m_add_args);
        py::exec(R"(
                 if _add_args:
                     _str = '{}('.format(_str)
                     for i in range(0, len(args)):
                         if i == 0:
                             _str = '{}{}'.format(_str, args[i])
                         else:
                             _str = '{}, {}'.format(_str, args[i])

                     for key, val in kwargs:
                         _str = '{}, {}={}'.format(_str, key, val)

                     _str = '{})'.format(_str)
                 )",
                 py::globals(), locals);
        return locals["_str"].cast<std::string>();
    }

protected:
    bool        m_add_args = false;
    bool        m_is_class = false;
    std::string m_key      = "";
};

//======================================================================================//

class auto_timer : public base_decorator
{
public:
    auto_timer() {}

    auto_timer(std::string key, bool add_args, bool is_class, bool report_at_exit)
    : base_decorator(key, add_args, is_class)
    , m_report_at_exit(report_at_exit)
    {
        auto _n = 2;
        m_file  = get_file(_n);
        m_line  = get_line(_n - 1);
    }

    py::object call(py::function func)
    {
        PRINT_HERE("");

        auto locals =
            py::dict("_func"_a = func, "_key"_a = m_key, "_file"_a = m_file,
                     "_line"_a = m_line, "_is_class"_a = m_is_class,
                     "_add_args"_a = m_add_args, "_report_at_exit"_a = m_report_at_exit);
        py::exec(R"(
                 import inspect
                 import timemory
                 from functools import wraps


                 @wraps(_func)
                 def _function_wrapper(func = _func, _key = _key, _is_class = _is_class,
                                       _add_args = _add_args, _file = _file, _line = _line,
                                       _report_at_exit = _report_at_exit,
                                       *args, **kwargs):

                     if len(args) > 0 and args[0] is not None and inspect.isclass(type(args[0])):
                         _is_class = True
                     else:
                         _is_class = False

                     _str = ''
                     if _is_class and len(args) > 0 and args[0] is not None:
                         _str = '[{}]'.format(type(args[0]).__name__)
                         # this check guards against class methods calling class methods
                         if _str in _key:
                             _str = ''

                     if _add_args:
                         _str = '{}('.format(_str)
                         for i in range(0, len(args)):
                             if i == 0:
                                 _str = '{}{}'.format(_str, args[i])
                             else:
                                 _str = '{}, {}'.format(_str, args[i])

                         for key, val in kwargs:
                             _str = '{}, {}={}'.format(_str, key, val)

                         _str = '{})'.format(_str)


                     _key = '{}{}'.format(_key, _str)

                     t = timemory.timer_decorator(func.__name__, _file, _line,
                         _key, _add_args or _is_class, _report_at_exit)

                     results = func(*args, **kwargs)
                     del t
                     return results
                 )",
                 py::globals(), locals);

        return locals["_function_wrapper"].cast<py::function>();
    }

private:
    bool        m_report_at_exit = false;
    int         m_line           = 0;
    std::string m_file           = "";
};

namespace init
{
static ::pytim::decorators::auto_timer*
auto_timer(std::string key, bool add_args, bool is_class, bool report_at_exit)
{
    return new ::pytim::decorators::auto_timer(key, add_args, is_class, report_at_exit);
}
}
}

//======================================================================================//

}  // namespace pytim

//======================================================================================//
