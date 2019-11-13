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
// disables a bunch of warnings
//
#include "timemory/utility/macros.hpp"

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

#include "timemory/backends/mpi.hpp"
#include "timemory/bits/settings.hpp"
#include "timemory/enum.h"
#include "timemory/manager.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_timer.hpp"
#include "timemory/variadic/auto_tuple.hpp"
#include "timemory/variadic/component_list.hpp"
#include "timemory/variadic/component_tuple.hpp"

//======================================================================================//

namespace py = pybind11;
using namespace std::placeholders;  // for _1, _2, _3...
using namespace py::literals;
using namespace tim::component;

using auto_timer_t = tim::auto_timer;

using auto_usage_t =
    tim::auto_tuple<page_rss, peak_rss, num_minor_page_faults, num_major_page_faults,
                    voluntary_context_switch, priority_context_switch>;
using auto_list_t = tim::complete_auto_list_t;

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
    : m_manager(manager_t::instance().get())
    {}

    ~manager_wrapper() {}

    // ensures thread-local version is called
    manager_t* get() { return manager_t::instance().get(); }

protected:
    manager_t* m_manager;
};

//======================================================================================//

class auto_timer_decorator
{
public:
    auto_timer_decorator(auto_timer_t* _ptr = nullptr)
    : m_ptr(_ptr)
    {}

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

component_enum_vec
components_list_to_vec(py::list pystr_list)
{
    std::vector<std::string> str_list;
    for(auto itr : pystr_list)
        str_list.push_back(itr.cast<std::string>());
    return tim::enumerate_components(str_list);
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
create_component_list(std::string obj_tag, const component_enum_vec& components)
{
    auto obj = new component_list_t(obj_tag, true, tim::settings::flat_profile());
    tim::initialize(*obj, components);
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
timer(std::string key)
{
    return new tim_timer_t(key, true, tim::settings::flat_profile());
}

//--------------------------------------------------------------------------------------//

auto_timer_t*
auto_timer(std::string key, bool report_at_exit)
{
    return new auto_timer_t(key, tim::settings::flat_profile(), report_at_exit);
}

//--------------------------------------------------------------------------------------//

rss_usage_t*
rss_usage(std::string key, bool record)
{
    rss_usage_t* _rss = new rss_usage_t(key, true, tim::settings::flat_profile());
    if(record)
        _rss->measure();
    return _rss;
}

//--------------------------------------------------------------------------------------//

component_list_t*
component_list(py::list components, std::string key)
{
    return create_component_list(key, components_enum_to_vec(components));
}

//----------------------------------------------------------------------------//

auto_timer_decorator*
timer_decorator(const std::string& key, bool report_at_exit)
{
    auto_timer_decorator* _ptr = new auto_timer_decorator();
    if(!tim::settings::enabled())
        return _ptr;
    return &(*_ptr =
                 new auto_timer_t(key, tim::settings::flat_profile(), report_at_exit));
}

//----------------------------------------------------------------------------//

component_list_decorator*
component_decorator(py::list components, const std::string& key)
{
    component_list_decorator* _ptr = new component_list_decorator();
    if(!manager_t::is_enabled())
        return _ptr;

    return &(*_ptr = create_component_list(key, components_enum_to_vec(components)));
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

             parser.add_argument('--disable', required=False,
                                 action='store_false',
                                 dest='enabled',
                                 help="Disable timemory for script")

             parser.add_argument('--enable', required=False,
                                 action='store_true',
                                 dest='enabled', help="Enable timemory for script")

             parser.add_argument('--disable-serialization',
                                 required=False, action='store_false',
                                 dest='serialize',
                                 help="Disable serialization for timers")

             parser.add_argument('--enable-serialization',
                                 required=False, action='store_true',
                                 dest='serialize',
                                 help="Enable serialization for timers")

             parser.add_argument('--max-depth',
                                 help="Maximum depth",
                                 type=int,
                                 default=timemory.settings.max_depth)

             parser.add_argument('--enable-dart',
                                 help="Print DartMeasurementFile tag for plots",
                                 required=False, action='store_true')

             parser.add_argument('--write-ctest-notes',
                                 help="Write a CTestNotes.cmake file for TiMemory ASCII output",
                                 required=False, action='store_true')

             parser.set_defaults(enabled=True)
             parser.set_defaults(serialize=True)
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
             timemory.settings.enabled = args.enabled
             timemory.settings.output_path = args.output_path
             timemory.settings.output_prefix = args.output_prefix
             timemory.settings.max_depth = args.max_depth
             timemory.settings.json_output = args.serialize
             timemory.options.echo_dart = args.enable_dart
             timemory.options.ctest_notes = args.write_ctest_notes
             _enable_serial = args.serialize
             )",
             py::globals(), locals);
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

template <typename _Tuple>
struct construct_dict
{
    using Type = _Tuple;

    construct_dict(_Tuple& _tup, py::dict& _dict)
    {
        auto _label = std::get<0>(_tup);
        if(_label.size() > 0)
            _dict[_label.c_str()] = std::get<1>(_tup);
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct construct_dict<std::tuple<std::string, void*>>
{
    using Type = std::tuple<std::string, void*>;

    template <typename... _Args>
    construct_dict(_Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct dict
{
    static py::dict construct(std::tuple<_Types...>& _tup)
    {
        using apply_types = std::tuple<construct_dict<_Types>...>;
        py::dict _dict;
        ::tim::apply<void>::access<apply_types>(_tup, std::ref(_dict));
        return _dict;
    }
};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct dict<std::tuple<_Types...>>
{
    static py::dict construct(std::tuple<_Types...>& _tup)
    {
        return dict<_Types...>::construct(_tup);
    }
};

//======================================================================================//

struct settings
{};

//======================================================================================//

}  // namespace pytim

//======================================================================================//
