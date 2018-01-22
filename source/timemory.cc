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
#include <memory>

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
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std::placeholders;  // for _1, _2, _3...
using namespace py::literals;

#include "timemory/namespace.hpp"
#include "timemory/timing_manager.hpp"
#include "timemory/timer.hpp"
#include "timemory/rss.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/rss.hpp"

typedef NAME_TIM::timing_manager      timing_manager_t;
typedef NAME_TIM::timer               tim_timer_t;
typedef NAME_TIM::auto_timer          auto_timer_t;
typedef NAME_TIM::rss::usage                rss_usage_t;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> farray_t;

class timing_manager_wrapper
{
public:
    timing_manager_wrapper()
    : m_manager(timing_manager_t::instance())
    { }

    ~timing_manager_wrapper()
    { }

    timing_manager_t* get() { return m_manager; }

private:
    timing_manager_t* m_manager;
};

//============================================================================//
//  Python wrappers
//============================================================================//

PYBIND11_MODULE(timemory, tim)
{
    py::add_ostream_redirect(tim, "ostream_redirect");

    auto tman_init = [&] () { return new timing_manager_wrapper(); };


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
                 )", py::globals(), locals);
        auto ret = locals["result"].cast<std::string>();
        return ret;
    };

    auto set_timer_default_format = [=] (std::string format)
    {
        auto locals = py::dict("format"_a = format);
        py::exec(R"(
                 import timemory as tim
                 tim.default_format = format
                 )", py::globals(), locals);
        // update C++
        tim_timer_t::set_default_format(format);
        return format;
    };

    auto get_timer_default_format = [=] ()
    {
        auto locals = py::dict();
        py::exec(R"(
                 import timemory as tim
                 format = tim.default_format
                 )", py::globals(), locals);
        auto format = locals["format"].cast<std::string>();
        // in case changed in python, update C++
        tim_timer_t::set_default_format(format);
        return format;
    };

    auto timer_init = [=] (std::string begin = "", std::string format = "")
    {
        if(begin.empty())
        {
            std::stringstream keyss;
            keyss << get_func(1) << "@" << get_file(2) << ":" << get_line(1);
            begin = keyss.str();
        }

        if(format.empty())
            format = get_timer_default_format();

        return new tim_timer_t(begin, "", format, false, 3);
    };

    auto auto_timer_init = [=] (const std::string& key = "", int nback = 1)
    {
        std::stringstream keyss;
        keyss << get_func(nback);
        if(key != "" && key[0] != '@')
            keyss << "@";
        if(key != "")
            keyss << key;
        else
        {
            keyss << "@";
            keyss << get_file(nback+1);
            keyss << ":";
            keyss << get_line(nback);
        }
        auto op_line = get_line();
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
            [=] ()
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
    tim.attr("default_format")
            =  tim_timer_t::default_format;
    tim.def("set_default_format",
            set_timer_default_format,
            "Set the default format of the timers");
    tim.def("get_default_format",
            get_timer_default_format,
            "Get the default format of the timers");

    //------------------------------------------------------------------------//

    py::module timemory_util = tim.import("timemory-supp");
    tim.add_object("util", timemory_util);

    //------------------------------------------------------------------------//
    // Classes

    py::class_<timing_manager_wrapper> tman(tim, "timing_manager");
    py::class_<tim_timer_t> timer(tim, "timer");
    py::class_<auto_timer_t> auto_timer(tim, "auto_timer");

    //------------------------------------------------------------------------//

    timer.def(py::init(timer_init),
              "Initialization",
              py::return_value_policy::take_ownership,
              py::arg("begin") = "", py::arg("format") = "");
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
              [=] (py::object timer, bool no_min = true)
              { timer.cast<tim_timer_t*>()->print(no_min); },
              "Report timer",
              py::arg("no_min") = true);

    //------------------------------------------------------------------------//

    tman.def(py::init<>(tman_init), "Initialization",
             py::return_value_policy::take_ownership);
    tman.def("report",
             [=] (py::object tman, bool no_min = false, bool serialize = true,
                  std::string serial_filename = "output.json")
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

                 timing_manager_t* _tman
                         = tman.cast<timing_manager_wrapper*>()->get();

                 // set the output stream
                 if(_do_rep)
                     _tman->set_output_stream(repfnm);

                 // report ASCII output
                 _tman->report(no_min);

                 // handle the serialization
                 if(!_do_ser && serialize)
                 {
                     _do_ser = true;
                     serfnm = serial_filename;
                 }

                 if(_do_ser && timing_manager_t::instance()->size() > 0)
                     _tman->write_serialization(serfnm);
             },
             "Report timing manager",
             py::arg("no_min") = false,
             py::arg("serialize") = true,
             py::arg("serial_filename") = "output.json");
    tman.def("set_output_file",
             [=] (py::object tman, std::string fname)
             {
                 timing_manager_t* _tman = tman.cast<timing_manager_wrapper*>()->get();
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
             { return tman.cast<timing_manager_wrapper*>()->get()->size(); },
             "Size of timing manager");
    tman.def("clear",
             [=] (py::object tman)
             { return tman.cast<timing_manager_wrapper*>()->get()->clear(); },
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
                  tman.cast<timing_manager_wrapper*>()->get()->write_serialization(fname);
             },
             "Serialize the timing manager to JSON",
             py::arg("fname") = "");
    tman.def("set_max_depth",
             [=] (py::object tman, int depth)
             { tman.cast<timing_manager_wrapper*>()->get()->set_max_depth(depth); },
             "Set the max depth of the timers");
    tman.def("get_max_depth",
             [=] (py::object tman)
             { return tman.cast<timing_manager_wrapper*>()->get()->get_max_depth(); },
             "Get the max depth of the timers");
    tman.def("at",
             [=] (py::object tman, int i)
             {
                 tim_timer_t& _t = tman.cast<timing_manager_wrapper*>()->get()->at(i);
                 return &_t;
             },
             "Set the max depth of the timers",
             py::return_value_policy::reference);
    tman.def("merge",
             [=] (py::object tman, bool div_clocks)
             { tman.cast<timing_manager_wrapper*>()->get()->merge(div_clocks); },
             "Merge the thread-local timers",
             py::arg("div_clocks") = true);
    tman.def("json",
             [=] (py::object tman)
             {
                 std::stringstream ss;
                 tman.cast<timing_manager_wrapper*>()->get()->write_json(ss);
                 py::module _json = py::module::import("json");
                 return _json.attr("loads")(ss.str());
             }, "Get JSON serialization of timing manager");
    // keep from being garbage collected
    tim.attr("_static_timing_manager") = new timing_manager_wrapper();

    //------------------------------------------------------------------------//

    auto_timer.def(py::init(auto_timer_init), "Initialization",
                   py::arg("key") = "", py::arg("nback") = 1,
                   py::return_value_policy::take_ownership);

    tim.def("auto_decorator",
            [=] (py::function pyfunc, int n)
            {
                auto locals = py::dict();
                py::exec(R"(
                         func = tim.FUNC(2)
                         line = tim.LINE(2)
                         file = tim.FILE()
                         )", py::globals(), locals);
                auto func = locals["func"].cast<std::string>();
                auto line = locals["line"].cast<int32_t>();
                auto file = locals["file"].cast<std::string>();
                std::stringstream ss;
                ss << func << "@" << file << ":" << line;
                auto_timer_t autotimer = auto_timer_t(ss.str(), line);
                return pyfunc(n);
            },
            "Decorator function");

    tim.def("report",
            [=] (bool no_min = true)
            {
                timing_manager_t::instance()->report(no_min);
            },
            "Report the timing manager (default: no_min = True)",
            py::arg("no_min") = true);

    //------------------------------------------------------------------------//

}
