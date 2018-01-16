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
#include "timemory/signal_detection.hpp"

typedef NAME_TIM::util::timing_manager   timing_manager_t;
typedef NAME_TIM::util::timer            tim_timer_t;
typedef NAME_TIM::util::auto_timer       auto_timer_t;


//============================================================================//
//  Python wrappers
//============================================================================//

PYBIND11_MODULE(timemory, tim)
{
    py::add_ostream_redirect(tim, "ostream_redirect");

    auto tman_init = [=] () { return timing_manager_t::instance(); };

    auto timer_init = [=] (std::string begin)
    {
        std::string default_format
            =  " : %w wall, %u user + %s system = %t CPU [sec] (%p%)"
               " : RSS {tot,self}_{curr,peak}"
               " : (%C|%M)"
               " | (%c|%m) [MB]";
        return new tim_timer_t(begin, "", default_format, false, 3);
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

    auto auto_timer_init = [=] (const std::string& key = "")
    {
        std::stringstream keyss;
        keyss << get_func(1);
        if(key != "" && key[0] != '@')
            keyss << "@";
        keyss << key;
        auto op_line = get_line(2);
        return new auto_timer_t(keyss.str(), op_line, "pyc");
    };

    auto enable_signal_detection = [=] () { NAME_TIM::EnableSignalDetection(); };
    auto disable_signal_detection = [=] () { NAME_TIM::DisableSignalDetection(); };

    //========================================================================//
    //
    //  Binding implementation
    //
    //========================================================================//

    tim.def("LINE",
            get_line,
            "Function that emulates __LINE__ macro",
            py::arg("nback") = 1);
    tim.def("FUNC",
            get_func,
            "Function that emulates __FUNC__ macro",
            py::arg("nback") = 1);
    tim.def("FILE",
            get_file,
            "Function that emulates __FILE__ macro",
            py::arg("nback") = 2,
            py::arg("basename_only") = true,
            py::arg("use_dirname") = false,
            py::arg("noquotes") = false);
    tim.def("set_max_depth",
            [=] (int32_t ndepth)
            { timing_manager_t::instance()->set_max_depth(ndepth); },
            "Max depth of auto-timers");
    tim.def("get_max_depth",
            [=]()
            { return timing_manager_t::instance()->get_max_depth(); },
            "Max depth of auto-timers");
    tim.def("toggle",
            [=] (bool timers_on)
            { timing_manager_t::instance()->enable(timers_on); },
            "Enable/disable auto-timers",
            py::arg("timers_on") = true);
    tim.def("enabled",
            [=] ()
            { return timing_manager_t::instance()->is_enabled(); },
            "Return if the auto-timers are enabled or disabled");
    tim.def("enable_signal_detection",
            enable_signal_detection,
            "Enable signal detection");
    tim.def("disable_signal_detection",
            disable_signal_detection,
            "Enable signal detection");

    py::module timemory_util = tim.import("timemory-supp");
    tim.add_object("util", timemory_util);

    //------------------------------------------------------------------------//
    // Classes

    py::class_<timing_manager_t> tman(tim, "timing_manager");
    py::class_<tim_timer_t> timer(tim, "timer");
    py::class_<auto_timer_t> auto_timer(tim, "auto_timer");

    //------------------------------------------------------------------------//

    timer.def(py::init(timer_init),
              "Initialization",
              py::return_value_policy::take_ownership);
    timer.def("real_elapsed",
              [=] (py::object timer)
              { return timer.cast<tim_timer_t*>()->real_elapsed(); },
              "Elapsed wall clock");
    timer.def("sys_elapsed",
              [=] (py::object timer)
              { return timer.cast<tim_timer_t*>()->system_elapsed(); },
              "Elapsed system clock");
    timer.def("user_elapsed",
              [=] (py::object timer)
              { return timer.cast<tim_timer_t*>()->user_elapsed(); },
              "Elapsed user time");
    timer.def("start",
              [=] (py::object timer)
              { timer.cast<tim_timer_t*>()->start(); },
              "Start timer");
    timer.def("stop",
              [=] (py::object timer)
              { timer.cast<tim_timer_t*>()->stop(); },
              "Stop timer");
    timer.def("report",
              [=] (py::object timer)
              { timer.cast<tim_timer_t*>()->print(); },
              "Report timer");

    //------------------------------------------------------------------------//

    tman.def(py::init<>(tman_init), "Initialization",
             py::return_value_policy::reference);
    tman.def("report",
             [=] (py::object tman)
             {
                 auto locals = py::dict();
                 py::exec(R"(
                          import timemory as tim
                          repfnm = tim.util.opts.report_fname
                          serfnm = tim.util.opts.serial_fname
                          do_ret = tim.util.opts.report_file
                          do_ser = tim.util.opts.serial_report
                          )", py::globals(), locals);
                 auto repfnm = locals["repfnm"].cast<std::string>();
                 auto serfnm = locals["serfnm"].cast<std::string>();
                 auto _do_rep = locals["do_ret"].cast<bool>();
                 auto _do_ser = locals["do_ser"].cast<bool>();
                 if(_do_rep)
                     tman.cast<timing_manager_t*>()->set_output_stream(repfnm);
                 tman.cast<timing_manager_t*>()->report();
                 if(_do_ser)
                     tman.cast<timing_manager_t*>()->write_serialization(serfnm);
             },
             "Report timing manager");
    tman.def("set_output_file",
             [=] (py::object tman, std::string fname)
             {
                 timing_manager_t* _tman = tman.cast<timing_manager_t*>();
                 auto locals = py::dict("fname"_a = fname);
                 py::exec(R"(
                          import timemory as tim
                          tim.util.opts.set_report(fname)
                          )", py::globals(), locals);
                 _tman->set_output_stream(fname);
             },
             "Set the output stream file");
    tman.def("size",
             [=] (py::object tman)
             { return tman.cast<timing_manager_t*>()->size(); },
             "Size of timing manager");
    tman.def("clear",
             [=] (py::object tman)
             { return tman.cast<timing_manager_t*>()->clear(); },
             "Clear the timing manager");
    tman.def("serialize",
             [=] (py::object tman, std::string fname = "")
             {
                  if(fname.empty())
                  {
                      auto locals = py::dict();
                      py::exec(R"(
                               import timemory as tim
                               result = tim.util.opts.serial_fname
                               )", py::globals(), locals);
                      fname = locals["result"].cast<std::string>();
                  }
                  tman.cast<timing_manager_t*>()->write_serialization(fname);
             },
             "Serialize the timing manager to JSON",
             py::arg("fname") = "");
    tman.def("set_max_depth",
             [=] (py::object tman, int depth)
             { tman.cast<timing_manager_t*>()->set_max_depth(depth); },
             "Set the max depth of the timers");
    tman.def("get_max_depth",
             [=] (py::object tman)
             { return tman.cast<timing_manager_t*>()->get_max_depth(); },
             "Get the max depth of the timers");
    tman.def("at",
             [=] (py::object tman, int i)
             {
                 tim_timer_t& _t = tman.cast<timing_manager_t*>()->at(i);
                 return &_t;
             },
             "Set the max depth of the timers",
             py::return_value_policy::reference);

    //------------------------------------------------------------------------//

    auto_timer.def(py::init(auto_timer_init), "Initialization",
                   py::arg("key") = "",
                   py::return_value_policy::take_ownership);

    //------------------------------------------------------------------------//

}
