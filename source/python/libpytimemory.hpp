//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

#if !defined(TIMEMORY_PYBIND11_SOURCE)
#    define TIMEMORY_PYBIND11_SOURCE
#endif

//======================================================================================//
// disables a bunch of warnings
//
#include "timemory/utility/macros.hpp"

//======================================================================================//

#include "timemory/timemory.hpp"
//
#include "timemory/enum.h"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/insert.hpp"
#include "timemory/runtime/invoker.hpp"
#include "timemory/runtime/properties.hpp"

#include "pybind11/cast.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
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

//======================================================================================//

namespace py = pybind11;
using namespace std::placeholders;  // for _1, _2, _3...
using namespace py::literals;
using namespace tim::component;

using auto_list_t  = tim::available_auto_list_t;
using auto_timer_t = tim::auto_timer;
using tim_timer_t  = typename auto_timer_t::component_type;
using manager_t    = tim::manager;
using farray_t     = py::array_t<double, py::array::c_style | py::array::forcecast>;

//======================================================================================//

class manager_wrapper
{
public:
    manager_wrapper();
    ~manager_wrapper();
    manager_t* get();

protected:
    manager_t* m_manager;
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
using string_t = std::string;

//======================================================================================//
//
//                              INITITALIZATION
//
//======================================================================================//

namespace init
{
//
manager_wrapper*
manager()
{
    return new manager_wrapper();
}
//
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
#if defined(_WIN32) || defined(_WIN64)
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
                                 help="Write a CTestNotes.cmake file for timemory ASCII output",
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

}  // namespace pytim

//======================================================================================//
