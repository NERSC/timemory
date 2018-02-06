// MIT License
//
// Copyright (c) 2018, The Regents of the University of California, 
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
#include "timemory/mpi.hpp"

typedef NAME_TIM::timing_manager                timing_manager_t;
typedef NAME_TIM::timer                         tim_timer_t;
typedef NAME_TIM::auto_timer                    auto_timer_t;
typedef NAME_TIM::rss::usage                    rss_usage_t;
typedef NAME_TIM::rss::usage                    rss_usage_t;
typedef NAME_TIM::sys_signal                    sys_signal_t;
typedef NAME_TIM::signal_settings               signal_settings_t;
typedef signal_settings_t::signal_set_t         signal_set_t;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> farray_t;

//============================================================================//

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

class auto_timer_decorator
{
public:
    auto_timer_decorator(auto_timer_t* _ptr = nullptr)
    : m_ptr(_ptr)
    { }

    ~auto_timer_decorator()
    {
        delete m_ptr;
    }

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

//============================================================================//

py::module load_module(const std::string& module,
                       const std::string& path)
{
    py::dict locals;
    locals["module_name"] = py::cast(module);
    locals["path"]        = py::cast(path);

    py::exec(R"(
             import imp
             new_module = imp.load_module(module_name, open(path), path, ('py', 'U', imp.PY_SOURCE))
             )", py::globals(), locals);

    return locals["new_module"].cast<py::module>();
}

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

    auto auto_timer_init = [=] (const std::string& key = "",
                                bool report_at_exit = false,
                                int nback = 1,
                                bool added_args = false)
    {
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
            keyss << get_file(nback+1);
            keyss << ":";
            keyss << get_line(nback);
        }
        auto op_line = get_line();
        return new auto_timer_t(keyss.str(), op_line, "pyc", report_at_exit);
    };

    auto timer_decorator_init = [=] (const std::string& func,
                                     const std::string& file,
                                     int line,
                                     const std::string& key,
                                     bool added_args,
                                     bool report_at_exit)
    {
        auto_timer_decorator* _ptr = new auto_timer_decorator();
        if(!auto_timer_t::alloc_next())
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
        return &(*_ptr = new auto_timer_t(keyss.str(), line, "pyc", report_at_exit));
    };

    auto rss_usage_init = [=] (bool record = false)
    {
        return (record) ? new rss_usage_t(0) : new rss_usage_t();
    };

    auto signal_list_to_set = [=] (py::list signal_list) -> signal_set_t
    {
        signal_set_t signal_set;
        for(auto itr : signal_list)
            signal_set.insert(itr.cast<sys_signal_t>());
        return signal_set;
    };

    auto get_default_signal_set = [=] () -> signal_set_t
    {
        return NAME_TIM::signal_settings::enabled();
    };

    auto enable_signal_detection = [=] (py::list signal_list = py::list())
    {
        auto _sig_set = (signal_list.size() == 0)
                        ? get_default_signal_set()
                        : signal_list_to_set(signal_list);
        NAME_TIM::EnableSignalDetection(_sig_set);
    };

    auto disable_signal_detection = [=] ()
    {
        NAME_TIM::DisableSignalDetection();
    };

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
            "Enable signal detection",
            py::arg("signal_list") = py::list());
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
    tim.def("has_mpi_support",
            [=] ()
            { return NAME_TIM::has_mpi_support(); },
            "Return if the TiMemory library has MPI support");

    //------------------------------------------------------------------------//
    // Classes

    py::class_<timing_manager_wrapper> tman(tim, "timing_manager");
    py::class_<tim_timer_t> timer(tim, "timer");
    py::class_<auto_timer_t> auto_timer(tim, "auto_timer");
    py::class_<auto_timer_decorator> timer_decorator(tim, "timer_decorator");
    py::class_<rss_usage_t> rss_usage(tim, "rss_usage");

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
    timer.def("__str__",
              [=] (py::object timer, bool no_min = true)
              { return timer.cast<tim_timer_t*>()->as_string(no_min); },
              "Stringify timer",
              py::arg("no_min") = true);
    timer.def("__iadd__",
             [=] (py::object timer, py::object _rss)
             {
                 *(timer.cast<tim_timer_t*>()) +=
                         *(_rss.cast<rss_usage_t*>());
                 return timer;
             },
             "Add RSS measurement");
    timer.def("__isub__",
             [=] (py::object timer, py::object _rss)
             {
                 *(timer.cast<tim_timer_t*>()) -=
                         *(_rss.cast<rss_usage_t*>());
                 return timer;
             },
             "Subtract RSS measurement");

    //------------------------------------------------------------------------//

    tman.def(py::init<>(tman_init), "Initialization",
             py::return_value_policy::take_ownership);
    tman.def("report",
             [=] (py::object tman, bool no_min = false, bool serialize = true,
                  std::string serial_filename = "")
             {
                 auto locals = py::dict();
                 py::exec(R"(
                          import timemory.options as options
                          repfnm = options.report_fname
                          serfnm = options.serial_fname
                          do_ret = options.report_file
                          do_ser = options.serial_file
                          outdir = options.output_dir
                          options.ensure_directory_exists('{}/test.txt'.format(outdir))
                          )", py::globals(), locals);

                 auto outdir = locals["outdir"].cast<std::string>();
                 auto repfnm = locals["repfnm"].cast<std::string>();
                 auto serfnm = locals["serfnm"].cast<std::string>();
                 auto _do_rep = locals["do_ret"].cast<bool>();
                 auto _do_ser = locals["do_ser"].cast<bool>();

                 if(repfnm.find(outdir) != 0)
                     repfnm = outdir + "/" + repfnm;

                 if(serfnm.find(outdir) != 0)
                     serfnm = outdir + "/" + serfnm;

                 timing_manager_t* _tman
                         = tman.cast<timing_manager_wrapper*>()->get();

                 // set the output stream
                 if(_do_rep)
                 {
                     std::cout << "Outputting timing_manager to '" << repfnm
                               << "'..." << std::endl;
                     _tman->set_output_stream(repfnm);
                 }

                 // report ASCII output
                 _tman->report(no_min);

                 // handle the serialization
                 if(!_do_ser && serialize)
                 {
                     _do_ser = true;
                     if(!serial_filename.empty())
                         serfnm = serial_filename;
                     else if(serfnm.empty())
                         serfnm = "output.json";
                 }

                 if(_do_ser && timing_manager_t::instance()->size() > 0)
                 {
                     std::cout << "Serializing timing_manager to '" << serfnm
                               << "'..." << std::endl;
                     _tman->write_serialization(serfnm);
                 }
             },
             "Report timing manager",
             py::arg("no_min") = false,
             py::arg("serialize") = true,
             py::arg("serial_filename") = "");
    tman.def("__str__",
             [=] (py::object tman)
             {
                 timing_manager_t* _tman
                         = tman.cast<timing_manager_wrapper*>()->get();
                 std::stringstream ss;
                 bool no_min = true;
                 _tman->report(ss, no_min);
                 return ss.str();
             },
             "Stringify the timing manager report");
    tman.def("set_output_file",
             [=] (py::object tman, std::string fname)
             {
                 timing_manager_t* _tman = tman.cast<timing_manager_wrapper*>()->get();
                 auto locals = py::dict("fname"_a = fname);
                 py::exec(R"(
                          import timemory as tim
                          tim.options.set_report(fname)
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
                               import timemory.options as options
                               import os
                               result = options.serial_fname
                               if not options.output_dir in result:
                                   result = os.path.join(options.output_dir, result)
                               options.ensure_directory_exists(result)
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
    tman.def("__iadd__",
             [=] (py::object tman, py::object _rss)
             {
                 *(tman.cast<timing_manager_wrapper*>()->get()) +=
                         *(_rss.cast<rss_usage_t*>());
                 return tman;
             },
             "Add RSS measurement");
    tman.def("__isub__",
             [=] (py::object tman, py::object _rss)
             {
                 *(tman.cast<timing_manager_wrapper*>()->get()) -=
                         *(_rss.cast<rss_usage_t*>());
                 return tman;
             },
             "Subtract RSS measurement");

    // keep from being garbage collected
    tim.attr("_static_timing_manager") = new timing_manager_wrapper();

    //------------------------------------------------------------------------//

    auto_timer.def(py::init(auto_timer_init),
                   "Initialization",
                   py::arg("key") = "",
                   py::arg("report_at_exit") = false,
                   py::arg("nback") = 1,
                   py::arg("added_args") = false,
                   py::return_value_policy::take_ownership);

    timer_decorator.def(py::init(timer_decorator_init),
                        "Initialization",
                        py::return_value_policy::automatic);

    //------------------------------------------------------------------------//

    rss_usage.def(py::init(rss_usage_init),
                  "Initialization of RSS measurement class",
                  py::return_value_policy::take_ownership,
                  py::arg("record") = false);
    rss_usage.def("record",
                  [=] (py::object self)
                  {
                      self.cast<rss_usage_t*>()->record();
                  },
                  "Record the RSS usage");
    rss_usage.def("__str__",
                  [=] (py::object self)
                  {
                      std::stringstream ss;
                      ss << *(self.cast<rss_usage_t*>());
                      return ss.str();
                  },
                  "Stringify the rss usage");
    rss_usage.def("__iadd__",
                  [=] (py::object self, py::object rhs)
                  {
                      *(self.cast<rss_usage_t*>())
                            += *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Add rss usage");
    rss_usage.def("__isub__",
                  [=] (py::object self, py::object rhs)
                  {
                      *(self.cast<rss_usage_t*>())
                            -= *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Subtract rss usage");
    rss_usage.def("__add__",
                  [=] (py::object self, py::object rhs)
                  {
                      rss_usage_t* _rss
                            = new rss_usage_t(*(self.cast<rss_usage_t*>()));
                      *_rss += *(rhs.cast<rss_usage_t*>());
                      return _rss;
                  },
                  "Add rss usage",
                  py::return_value_policy::take_ownership);
    rss_usage.def("__sub__",
                  [=] (py::object self, py::object rhs)
                  {
                      rss_usage_t* _rss
                            = new rss_usage_t(*(self.cast<rss_usage_t*>()));
                      *_rss -= *(rhs.cast<rss_usage_t*>());
                      return _rss;
                  },
                  "Subtract rss usage",
                  py::return_value_policy::take_ownership);
    rss_usage.def("current",
                  [=] (py::object self)
                  {
                      return self.cast<rss_usage_t*>()->current();
                  },
                  "Return the current rss usage");
    rss_usage.def("peak",
                  [=] (py::object self)
                  {
                      return self.cast<rss_usage_t*>()->peak();
                  },
                  "Return the current rss usage");


    //------------------------------------------------------------------------//

    tim.def("report",
            [=] (bool no_min = true)
            { timing_manager_t::instance()->report(no_min); },
            "Report the timing manager (default: no_min = True)",
            py::arg("no_min") = true);

    tim.def("clear",
            [=] ()
            { timing_manager_t::instance()->clear(); },
            "Clear the timing manager");

    tim.def("size",
            [=] ()
            { return timing_manager_t::instance()->size(); },
            "Size of the timing manager");

    tim.def("set_exit_action",
            [=] (py::function func)
            {
                auto _func = [=] (int errcode) -> void
                {
                    func(errcode);
                };
                //typedef tim::signal_settings::signal_function_t signal_function_t;
                typedef std::function<void(int)> signal_function_t;
                using std::placeholders::_1;
                signal_function_t _f = std::bind<void>(_func, _1);
                tim::signal_settings::set_exit_action(_f);
            },
            "Set the exit action when a signal is raised -- function must accept integer");

    //------------------------------------------------------------------------//
    //
    //      Dummy submodules
    //
    //------------------------------------------------------------------------//

    //auto _util = py::module::import("timemory.util");
    //auto _plot = py::module::import("timemory.plotting");
    //auto _mpis = py::module::import("timemory.mpi_support");

    /*
    py::module utl = tim.def_submodule("util",          "Utility submodule");
    py::module plt = tim.def_submodule("plotting",      "Plotting submodule");
    py::module mpi = tim.def_submodule("mpi_support",   "MPI info submodule");

    auto _get_path = [=] (std::string subpath)
    {
        auto locals = py::dict("subpath"_a = subpath);
        py::exec(R"(
                 _f = __file__
                 try:
                     # if this succeeds, we are not calling from timemory.__init__
                     # and thus, want to use this as base
                     import timemory
                     _f = timemory.__file__
                 except:
                     # else, we are likely in a submodule so go up one directory
                     _f = os.path.dirname(_f)
                 import os
                 import sys
                 __file_path = os.path.abspath(os.path.dirname(_f))
                 __path = os.path.join(__file_path, subpath)
                 )", py::globals(), locals);
        return locals["__path"].cast<std::string>().c_str();
    };

    tim.def("__add_util_object",
            [&] ()
    {
        auto _util = py::eval_file(_get_path("util/util.py"), py::globals(), utl);
        tim.add_object("util", _util, true);
    }, "Add the utility object (fallback method)");

    tim.def("__add_plotting_object",
            [&] ()
    {
        auto _plot = py::eval_file(_get_path("plotting/plotting.py"), py::globals(), plt);
        tim.add_object("plotting", _plot, true);
    }, "Add the utility object (fallback method)");

    tim.def("__add_mpi_support_object",
            [&] ()
    {
        auto _mpis = py::eval_file(_get_path("mpi_support/mpi_support.py"), py::globals(), mpi);
        tim.add_object("mpi_support", _mpis, true);
    }, "Add the utility object (fallback method)");*/

    tim.def("load_module", &load_module, "Load a python submodule");

    tim.def("add_object",
            [&] (std::string _name, py::module _obj)
            {
                tim.add_object(_name.c_str(), _obj);
            }, "Add an object to module");

    //========================================================================//
    //
    //      Signals submodule
    //
    //========================================================================//

    py::module sig = tim.def_submodule("signals",       "Signals submodule");

    //------------------------------------------------------------------------//

    py::enum_<sys_signal_t> sys_signal_enum(sig, "sys_signal", py::arithmetic(),
                                            "Signals for TiMemory module");

    //------------------------------------------------------------------------//

    sys_signal_enum
            .value("Hangup", sys_signal_t::sHangup)
            .value("Interrupt", sys_signal_t::sInterrupt)
            .value("Quit", sys_signal_t::sQuit)
            .value("Illegal", sys_signal_t::sIllegal)
            .value("Trap", sys_signal_t::sTrap)
            .value("Abort", sys_signal_t::sAbort)
            .value("Emulate", sys_signal_t::sEmulate)
            .value("FPE", sys_signal_t::sFPE)
            .value("Kill", sys_signal_t::sKill)
            .value("Bus", sys_signal_t::sBus)
            .value("SegFault", sys_signal_t::sSegFault)
            .value("System", sys_signal_t::sSystem)
            .value("Pipe", sys_signal_t::sPipe)
            .value("Alarm", sys_signal_t::sAlarm)
            .value("Terminate", sys_signal_t::sTerminate)
            .value("Urgent", sys_signal_t::sUrgent)
            .value("Stop", sys_signal_t::sStop)
            .value("CPUtime", sys_signal_t::sCPUtime)
            .value("FileSize", sys_signal_t::sFileSize)
            .value("VirtualAlarm", sys_signal_t::sVirtualAlarm)
            .value("ProfileAlarm", sys_signal_t::sProfileAlarm);

    //========================================================================//
    //
    //      Options submodule
    //
    //========================================================================//

    py::module opts = tim.def_submodule("options", "I/O options submodule");

    // ---------------------------------------------------------------------- //

    opts.attr("report_file") = false;
    opts.attr("serial_file") = true;
    opts.attr("use_timers") = true;
    opts.attr("max_timer_depth") = std::numeric_limits<uint16_t>::max();
    opts.attr("report_fname") = "timing_report.out";
    opts.attr("serial_fname") = "timing_report.json";
    opts.attr("output_dir") = ".";

    // ---------------------------------------------------------------------- //

    opts.def("default_max_depth",
            [=]()
    {
        return std::numeric_limits<uint16_t>::max();
    },
    "Return the default max depth");

    // ---------------------------------------------------------------------- //

    opts.def("ensure_directory_exists",
             [=] (std::string file_path)
    {
        auto locals = py::dict("file_path"_a = file_path);
        py::exec(R"(
                 import os
                 from os.path import dirname

                 directory = dirname(file_path)
                 if not os.path.exists(directory) and directory != '':
                     os.makedirs(directory)
                 )",
                 py::globals(), locals);

    },
    "mkdir -p $(basename file_path)");

    // ---------------------------------------------------------------------- //

    opts.def("set_report",
             [=] (std::string fname)
    {
        std::stringstream ss;
        ss << opts.attr("output_dir").cast<std::string>();
        if(ss.str().length() > 0 && ss.str()[ss.str().length()-1] != '/')
            ss << "/";
        ss << fname;
        opts.attr("report_fname") = ss.str().c_str();
        opts.attr("report_file") = true;
    },
    "Set the ASCII report filename");

    // ---------------------------------------------------------------------- //

    opts.def("set_serial",
             [=] (std::string fname)
    {
        std::stringstream ss;
        ss << opts.attr("output_dir").cast<std::string>();
        if(ss.str().length() > 0 && ss.str()[ss.str().length()-1] != '/')
            ss << "/";
        ss << fname;
        opts.attr("serial_fname") = ss.str().c_str();
        opts.attr("serial_file") = true;
    },
    "Set the JSON serialization filename");

    // ---------------------------------------------------------------------- //

    opts.def("add_arguments",
             [=] (py::object parser = py::none(), std::string fname = "")
    {
        auto locals = py::dict("parser"_a = parser,
                               "fname"_a = fname);
        py::exec(R"(
                 import sys
                 import os
                 from os.path import dirname
                 from os.path import join
                 import argparse

                 if parser is None:
                    parser = argparse.ArgumentParser()

                 # Function to add default output arguments
                 def get_file_tag(fname):
                     import os
                     _l = os.path.basename(fname).split('.')
                     if len(_l) > 1:
                         _l.pop()
                     return ("{}".format('_'.join(_l)))

                 def_fname = "timing_report"
                 if fname != "":
                     def_fname = '_'.join(["timing_report", get_file_tag(fname)])

                 parser.add_argument('--output-dir', required=False,
                                     default='.', type=str, help="Output directory")
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
                                     dest='serial_file',
                                     help="Disable serialization for timers")
                 parser.add_argument('--enable-timer-serialization',
                                     required=False, action='store_true',
                                     dest='serial_file',
                                     help="Enable serialization for timers")
                 parser.add_argument('--max-timer-depth',
                                     help="Maximum timer depth",
                                     type=int,
                                     default=options.default_max_depth())

                 parser.set_defaults(use_timers=True)
                 parser.set_defaults(serial_file=True)
                 )",
                 py::globals(), locals);
        return locals["parser"].cast<py::object>();
    },
    "Function to add default output arguments",
    py::arg("parser") = py::none(), py::arg("fname") = "");

    // ---------------------------------------------------------------------- //

    opts.def("parse_args",
             [=] (py::object args)
    {
        auto locals = py::dict("args"_a = args);
        py::exec(R"(
                 import sys
                 import os
                 from os.path import dirname
                 from os.path import basename
                 from os.path import join

                 # Function to add default output arguments
                 options.serial_file = args.serial_file
                 options.use_timers = args.use_timers
                 options.max_timer_depth = args.max_timer_depth
                 options.output_dir = args.output_dir

                 if args.filename:
                     options.set_report("{}.{}".format(args.filename, "out"))
                     options.set_serial("{}.{}".format(args.filename, "json"))

                 import timemory
                 timemory.toggle(options.use_timers)
                 timemory.set_max_depth(options.max_timer_depth)
                 )",
                 py::globals(), locals);
    },
    "Function to handle the output arguments");

    // ---------------------------------------------------------------------- //

    opts.def("add_arguments_and_parse",
             [=] (py::object parser = py::none(), std::string fname = "")
    {
        auto locals = py::dict("parser"_a = parser,
                               "fname"_a = fname);
        py::exec(R"(
                 import timemory.options as options

                 # Combination of timing.add_arguments and timing.parse_args but returns
                 parser = options.add_arguments(parser, fname)
                 args = parser.parse_args()
                 options.parse_args(args)
                 )",
                 py::globals(), locals);
        return locals["args"].cast<py::object>();
    },
    "Combination of timing.add_arguments and timing.parse_args but returns",
    py::arg("parser") = py::none(), py::arg("fname") = "");

    // ---------------------------------------------------------------------- //

    //========================================================================//

}
/*
//py::exec(R"(
//         timemory.util.__dict__.update(timemory._util.__dict__)
//         )", py::globals(), tim);


int main()
try
{
    Py_Initialize();
    //pybind11_init_timemory();
    std::cout << "Initializing..." << std::endl;
    py::object main     = py::module::import("__main__");
    py::object globals  = main.attr("__dict__");
    py::object module   = import("util", "util.py", globals);
    //py::object Strategy = module.attr("Strategy");
    //py::object strategy = Strategy(&server);

    return 0;
}
catch(const py::error_already_set&)
{
    std::cerr << ">>> Error! Uncaught exception:\n";
    PyErr_Print();
    return 1;
}*/
