// MIT License
//
// Copyright (c) 2018 Jonathan R. Madsen
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

#include <future>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <map>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/eval.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;
using namespace std::placeholders;  // for _1, _2, _3...
using namespace py::literals;

#include "timemory/namespace.hpp"
#include "timemory/timing_manager.hpp"
#include "timemory/timer.hpp"
#include "timemory/rss.hpp"
#include "timemory/auto_timer.hpp"

typedef NAME_TIM::util::timing_manager   timing_manager_t;
typedef NAME_TIM::util::timer            tim_timer_t;
typedef NAME_TIM::util::auto_timer       auto_timer_t;

//============================================================================//
//  Utility functions
//============================================================================//

//============================================================================//
//  Python wrappers
//============================================================================//

PYBIND11_MODULE(timemory, tim)
{
    py::add_ostream_redirect(tim, "ostream_redirect");

    std::string default_format
        =  " : %w wall, %u user + %s system = %t CPU [sec] (%p%)"
           " : RSS {tot,self}_{curr,peak}"
           " : (%c|%m)"
           " | (%C|%M) [MB]";

    auto tman_init = [&] () { return timing_manager_t::instance(); };

    auto timer_init = [&] (std::string begin)
    {
        return tim_timer_t(begin, "", default_format, false, 3);
    };

    auto get_line = [] (int nback = 1)
    {
        auto locals = py::dict("back"_a = nback);
        py::exec(R"(
                 import sys
                 result = int(sys._getframe(back).f_lineno)
                 )", py::globals(), locals);
        auto ret = locals["result"].cast<int>();
        return ret;
    };

    auto get_func = [] (int nback = 1)
    {
        auto locals = py::dict("back"_a = nback);
        py::exec(R"(
                 import sys
                 result = ("{}".format(sys._getframe(back).f_code.co_name))
                 )", py::globals(), locals);
        auto ret = locals["result"].cast<std::string>();
        return ret;
    };

    auto get_file = [] (int nback = 2, bool only_basename = true,
                    bool use_dirname = false, bool noquotes = false)
    {
        auto locals = py::dict("back"_a = nback,
                               "only_basename"_a = only_basename,
                               "use_dirname"_a = use_dirname,
                               "noquotes"_a = noquotes);
        py::exec(R"(
                 import sys
                 import os
                 from os.path import dirname
                 from os.path import basename
                 from os.path import join

                 def get_fcode(back):
                     return sys._getframe(back).f_code.co_filename

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
                 )", py::globals(), locals);
        auto ret = locals["result"].cast<std::string>();
        return ret;
    };

    auto add_arguments = [] (py::object parser, std::string fname = "")
    {
        auto locals = py::dict("fname"_a = fname,
                               "parser"_a = parser);
        py::exec(R"(
                 from os.path import basename

                 def get_file_tag(fname):
                     _l = basename(fname).split('.')
                     _l.pop()
                     return ("{}".format('_'.join(_l)))

                 def_fname = "timing_report"
                 if fname is not None:
                     def_fname = '_'.join(["timing_report", get_file_tag(fname)])

                 parser.add_argument('--output-dir', required=False,
                                     default='./', type=str, help="Output directory")
                 parser.add_argument('--filename', required=False,
                                     default=def_fname, type=str,
                     help="Filename for timing report w/o directory and w/o suffix")
                 parser.add_argument('--disable-timers', required=False,
                                     action='store_false',
                                     dest='use_timers',
                                     help="Disable timers for script")
                 parser.add_argument('--enable-timers', required=False,
                                     action='store_true',
                                     dest='use_timers', help="Enable timers for script")
                 parser.add_argument('--disable-timer-serialization',
                                     required=False, action='store_false',
                                     dest='serial_report',
                                     help="Disable serialization for timers")
                 parser.add_argument('--enable-timer-serialization',
                                     required=False, action='store_true',
                                     dest='serial_report',
                                     help="Enable serialization for timers")
                 parser.add_argument('--max-timer-depth',
                                     help="Maximum timer depth",
                                     type=int,
                                     default=65536)

                 parser.set_defaults(use_timers=True)
                 parser.set_defaults(serial_report=False)

                 )", py::globals(), locals);
    };

    auto parse_args = [] (py::object args)
    {
        auto locals = py::dict("args"_a = args);
        py::exec(R"(
                 from os.path import join

                 use_timers = args.use_timers
                 serial_report = args.serial_report
                 max_depth = args.max_timer_depth

                 _report_fname = "{}.{}".format(args.timing_fname, "out")
                 _serial_fname = "{}.{}".format(args.timing_fname, "json")
                 report_fname = join(args.output_dir, _report_fname);
                 serial_fname = join(args.output_dir, _serial_fname);

                 )", py::globals(), locals);

        auto report_fname = locals["report_fname"].cast<std::string>();
        auto use_timer = locals["use_timers"].cast<bool>();
        auto max_depth = locals["max_depth"].cast<int32_t>();
        //auto serial_fname = locals["serial_fname"].cast<std::string>();
        //auto serial_report = locals["serial_report"].cast<bool>();

        timing_manager_t::instance()->set_max_depth(max_depth);
        timing_manager_t::instance()->set_output_stream(report_fname);
        timing_manager_t::instance()->enable(use_timer);

    };

    auto add_arguments_and_parse = [&] (py::object parser,
            std::string fname = "")
    {
        add_arguments(parser, fname);
        auto locals = py::dict("parser"_a = parser);
        py::exec(R"(
                 args = parser.parse_args()
                 )",
                 py::globals(), locals);

        auto args = locals["args"];
        parse_args(args);
        return args;
    };

    auto auto_timer_init = [&] (const std::string& key = "")
    {
        std::stringstream keyss;
        keyss << get_func(1);
        if(key != "" && key[0] != '@')
            keyss << "@";
        keyss << key;
        auto op_line = get_line(2);
        return new auto_timer_t(keyss.str(), op_line, "pyc");
    };

    // we have to wrap each return type
    py::class_<timing_manager_t> tman(tim, "timing_manager");
    py::class_<tim_timer_t> timer(tim, "timer");
    py::class_<auto_timer_t> auto_timer(tim, "auto_timer");

    tman.def(py::init<>(tman_init), "Initialization");
    tman.def("report", &timing_manager_t::print, "Report timing manager");
    tman.def("size", &timing_manager_t::size, "Size of timing manager");
    tman.def("clear", &timing_manager_t::clear, "Clear the timing manager");
    tman.def("serialize", &timing_manager_t::write_serialization,
             "Serialize the timing manager to JSON");
    tman.def("set_max_depth", &timing_manager_t::set_max_depth,
             "Set the max depth of the timers");
    tman.def("get_max_depth", &timing_manager_t::get_max_depth,
             "Get the max depth of the timers");

    timer.def(py::init(timer_init), "Initialization");
    timer.def("real_elapsed", &tim_timer_t::real_elapsed, "Elapsed wall clock");
    timer.def("sys_elapsed", &tim_timer_t::system_elapsed, "Elapsed system clock");
    timer.def("user_elapsed", &tim_timer_t::user_elapsed, "Elapsed user time");
    timer.def("start", &tim_timer_t::start, "Start timer");
    timer.def("stop", &tim_timer_t::stop, "Stop timer");
    timer.def("report", &tim_timer_t::print, "Report timer");

    auto_timer.def(py::init(auto_timer_init), "Initialization",
                   py::arg("key") = "",
                   py::return_value_policy::take_ownership);

    tim.def("LINE", get_line, "Function that emulates __LINE__ macro",
            py::arg("nback") = 1);
    tim.def("FUNC", get_func, "Function that emulates __FUNC__ macro",
            py::arg("nback") = 1);
    tim.def("FILE", get_file, "Function that emulates __FILE__ macro",
            py::arg("nback") = 2, py::arg("basename_only") = true,
            py::arg("use_dirname") = false, py::arg("noquotes") = false);
    tim.def("max_depth",
            [&]() { return timing_manager_t::instance()->get_max_depth(); },
            "Max depth of auto-timers");
    tim.def("toggle",
            [&] (bool timers_on) { timing_manager_t::instance()->enable(timers_on); },
            "Enable/disable auto-timers", py::arg("timers_on") = true);
    tim.def("enabled",
            [&] () { return timing_manager_t::instance()->is_enabled(); },
            "Return if the auto-timers are enabled or disabled");
    tim.def("add_arguments", add_arguments,
            "Function to add default output arguments");
    tim.def("parse_args", parse_args,
            "Function to handle the output arguments");
    tim.def("add_arguments_and_parse", add_arguments_and_parse,
            "Combination of add_arguments and parse_args but returns");



}
