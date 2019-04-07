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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "libtimemory.hpp"
#include <pybind11/pybind11.h>

//======================================================================================//
//  Python wrappers
//======================================================================================//

PYBIND11_MODULE(libtimemory, tim)
{
    //------------------------------------------------------------------------//
    // py::add_ostream_redirect(tim, "ostream_redirect");
    //------------------------------------------------------------------------//

    //========================================================================//
    //
    //                  MAIN libtimemory MODULE (part 1)
    //
    //========================================================================//
    tim.def("LINE", &pytim::get_line, "Function that emulates __LINE__ macro",
            py::arg("nback") = 1);
    //------------------------------------------------------------------------//
    tim.def("FUNC", &pytim::get_func, "Function that emulates __FUNC__ macro",
            py::arg("nback") = 1);
    //------------------------------------------------------------------------//
    tim.def("FILE", &pytim::get_file, "Function that emulates __FILE__ macro",
            py::arg("nback") = 2, py::arg("basename_only") = true,
            py::arg("use_dirname") = false, py::arg("noquotes") = false);
    //------------------------------------------------------------------------//
    tim.def("set_max_depth",
            [=](int32_t ndepth) { manager_t::instance()->set_max_depth(ndepth); },
            "Max depth of auto-timers");
    //------------------------------------------------------------------------//
    tim.def("get_max_depth", [=]() { return manager_t::instance()->get_max_depth(); },
            "Max depth of auto-timers");
    //------------------------------------------------------------------------//
    tim.def("toggle", [=](bool timers_on) { manager_t::instance()->enable(timers_on); },
            "Enable/disable auto-timers", py::arg("timers_on") = true);
    //------------------------------------------------------------------------//
    tim.def("enable", [=]() { manager_t::instance()->enable(true); },
            "Enable auto-timers");
    //------------------------------------------------------------------------//
    tim.def("disable", [=]() { manager_t::instance()->enable(false); },
            "Disable auto-timers");
    //------------------------------------------------------------------------//
    tim.def("is_enabled", [=]() { return manager_t::instance()->is_enabled(); },
            "Return if the auto-timers are enabled or disabled");
    //------------------------------------------------------------------------//
    tim.def("enabled", [=]() { return manager_t::instance()->is_enabled(); },
            "Return if the auto-timers are enabled or disabled");
    //------------------------------------------------------------------------//
    tim.def("enable_signal_detection", &pytim::enable_signal_detection,
            "Enable signal detection", py::arg("signal_list") = py::list());
    //------------------------------------------------------------------------//
    tim.def("disable_signal_detection", &pytim::disable_signal_detection,
            "Enable signal detection");
    //------------------------------------------------------------------------//
    tim.def("has_mpi_support", [=]() { return tim::has_mpi_support(); },
            "Return if the TiMemory library has MPI support");
    //------------------------------------------------------------------------//

    //========================================================================//
    //
    //      Units submodule
    //
    //========================================================================//
    py::module units = tim.def_submodule("units", "units for timing and memory");

    units.attr("psec")     = tim::units::psec;
    units.attr("nsec")     = tim::units::nsec;
    units.attr("usec")     = tim::units::usec;
    units.attr("msec")     = tim::units::msec;
    units.attr("csec")     = tim::units::csec;
    units.attr("dsec")     = tim::units::dsec;
    units.attr("sec")      = tim::units::sec;
    units.attr("byte")     = tim::units::byte;
    units.attr("kilobyte") = tim::units::kilobyte;
    units.attr("megabyte") = tim::units::megabyte;
    units.attr("gigabyte") = tim::units::gigabyte;
    units.attr("terabyte") = tim::units::terabyte;
    units.attr("petabyte") = tim::units::petabyte;

    //------------------------------------------------------------------------//
    //  Class declarations
    //------------------------------------------------------------------------//
    py::class_<manager_wrapper>      man(tim, "manager");
    py::class_<tim_timer_t>          timer(tim, "timer");
    py::class_<auto_timer_t>         auto_timer(tim, "auto_timer");
    py::class_<auto_timer_decorator> timer_decorator(tim, "timer_decorator");
    py::class_<rss_usage_t>          rss_usage(tim, "rss_usage");
    // py::class_<rss_delta_t>          rss_delta(tim, "rss_delta");
    // py::class_<manager_wrapper,
    //           std::unique_ptr<manager_wrapper, py::nodelete>>
    //        man(tim, "manager");

    //========================================================================//
    //
    //                          TIMER
    //
    //========================================================================//
    timer.def(py::init(&pytim::init::timer), "Initialization",
              py::return_value_policy::take_ownership, py::arg("prefix") = "");
    //------------------------------------------------------------------------//
    timer.def("real_elapsed",
              [=](py::object timer) {
                  tim_timer_t& _timer = *(timer.cast<tim_timer_t*>());
                  auto&        obj    = std::get<0>(_timer);
                  return obj.compute_display();
              },
              "Elapsed wall clock");
    //------------------------------------------------------------------------//
    timer.def("sys_elapsed",
              [=](py::object timer) {
                  tim_timer_t& _timer = *(timer.cast<tim_timer_t*>());
                  auto&        obj    = std::get<1>(_timer);
                  return obj.compute_display();
              },
              "Elapsed system clock");
    //------------------------------------------------------------------------//
    timer.def("user_elapsed",
              [=](py::object timer) {
                  tim_timer_t& _timer = *(timer.cast<tim_timer_t*>());
                  auto&        obj    = std::get<2>(_timer);
                  return obj.compute_display();
              },
              "Elapsed user time");
    //------------------------------------------------------------------------//
    timer.def("start", [=](py::object timer) { timer.cast<tim_timer_t*>()->start(); },
              "Start timer");
    //------------------------------------------------------------------------//
    timer.def("stop", [=](py::object timer) { timer.cast<tim_timer_t*>()->stop(); },
              "Stop timer");
    //------------------------------------------------------------------------//
    timer.def("report",
              [=](py::object timer) {
                  std::cout << *(timer.cast<tim_timer_t*>()) << std::endl;
              },
              "Report timer");
    //------------------------------------------------------------------------//
    timer.def("__str__",
              [=](py::object timer) {
                  std::stringstream ss;
                  ss << *(timer.cast<tim_timer_t*>());
                  return ss.str();
              },
              "Stringify timer");
    //------------------------------------------------------------------------//
    timer.def("reset", [=](py::object self) { self.cast<tim_timer_t*>()->reset(); },
              "Reset the timer");
    //------------------------------------------------------------------------//

    //========================================================================//
    //
    //                          TIMING MANAGER
    //
    //========================================================================//
    man.attr("reported_files") = py::list();
    //------------------------------------------------------------------------//
    man.attr("serialized_files") = py::list();
    //------------------------------------------------------------------------//
    man.def(py::init<>(&pytim::init::manager), "Initialization",
            py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    man.def("report", &pytim::manager::report, "Report timing manager",
            py::arg("ign_cutoff") = false, py::arg("serialize") = false,
            py::arg("serial_filename") = "");
    //------------------------------------------------------------------------//
    man.def("__str__",
            [=](py::object man) {
                manager_t*        _man = man.cast<manager_wrapper*>()->get();
                std::stringstream ss;
                bool              ign_cutoff = true;
                bool              endline    = false;
                _man->report(ss, ign_cutoff, endline);
                return ss.str();
            },
            "Stringify the timing manager report");
    //------------------------------------------------------------------------//
    man.def("set_output_file",
            [=](py::object man, std::string fname) {
                manager_t* _man   = man.cast<manager_wrapper*>()->get();
                auto       locals = py::dict("fname"_a = fname);
                py::exec(R"(
                          import timemory as tim
                          tim.options.set_report(fname)
                          )",
                         py::globals(), locals);
                _man->set_output_stream(fname.c_str());
            },
            "Set the output stream file");
    //------------------------------------------------------------------------//
    man.def("size",
            [=](py::object man) { return man.cast<manager_wrapper*>()->get()->size(); },
            "Size of timing manager");
    //------------------------------------------------------------------------//
    man.def("clear",
            [=](py::object man) { man.cast<manager_wrapper*>()->get()->clear(); },
            "Clear the timing manager");
    //------------------------------------------------------------------------//
    man.def("write_missing",
            [=](py::object /*man*/, std::string fname) {
                auto locals = py::dict("fname"_a = fname);
                py::exec(R"(
                         import timemory.options as options
                         options.ensure_directory_exists(fname)
                         )",
                         py::globals(), locals);
                // man.cast<manager_wrapper*>()->get()->write_missing(fname.c_str());
            },
            "Write TiMemory missing to file");
    //------------------------------------------------------------------------//
    man.def("serialize", &pytim::manager::serialize,
            "Serialize the timing manager to JSON", py::arg("fname") = "");
    //------------------------------------------------------------------------//
    man.def("set_max_depth",
            [=](py::object man, int depth) {
                man.cast<manager_wrapper*>()->get()->set_max_depth(depth);
            },
            "Set the max depth of the timers");
    //------------------------------------------------------------------------//
    man.def("get_max_depth",
            [=](py::object man) {
                return man.cast<manager_wrapper*>()->get()->get_max_depth();
            },
            "Get the max depth of the timers");
    //------------------------------------------------------------------------//
    /*
    man.def("at",
            [=](py::object man, int i) {
                tim_timer_t& _t = man.cast<manager_wrapper*>()->get()->at(i);
                return &_t;
            },
            "Set the max depth of the timers", py::return_value_policy::reference);*/
    //------------------------------------------------------------------------//
    man.def("merge",
            [=](py::object man) { man.cast<manager_wrapper*>()->get()->merge(); },
            "Merge the thread-local timers");
    //------------------------------------------------------------------------//
    man.def("json",
            [=](py::object man) {
                std::stringstream ss;
                man.cast<manager_wrapper*>()->get()->write_json(ss);
                py::module _json = py::module::import("json");
                return _json.attr("loads")(ss.str());
            },
            "Get JSON serialization of timing manager");
    //------------------------------------------------------------------------//
    man.def("write_ctest_notes", &pytim::manager::write_ctest_notes,
            "Write a CTestNotes.cmake file", py::arg("directory") = ".",
            py::arg("append") = false);
    //------------------------------------------------------------------------//

    //========================================================================//
    //
    //                      AUTO TIMER
    //
    //========================================================================//
    auto_timer.def(py::init(&pytim::init::auto_timer), "Initialization",
                   py::arg("key") = "", py::arg("report_at_exit") = false,
                   py::arg("nback") = 1, py::arg("added_args") = false,
                   py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    auto_timer.def("local_timer",
                   [=](py::object /*_auto_timer*/) {
                       // return _auto_timer.cast<auto_timer_t*>()->local_timer();
                   },
                   "Get the timer for the auto-timer instance");
    //------------------------------------------------------------------------//
    auto_timer.def("global_timer",
                   [=](py::object /*_auto_timer*/) {
                       // return
                       // _auto_timer.cast<auto_timer_t*>()->local_timer().summation_timer();
                   },
                   "Get the timer for all the auto-timer instances (from manager)");
    //------------------------------------------------------------------------//
    auto_timer.def("__str__",
                   [=](py::object _pyauto_timer) {
                       std::stringstream _ss;
                       auto_timer_t* _auto_timer = _pyauto_timer.cast<auto_timer_t*>();
                       _ss << _auto_timer->local_object();
                       return _ss.str();
                   },
                   "Print the auto timer");
    //------------------------------------------------------------------------//

    //========================================================================//
    //
    //                      RSS USAGE
    //
    //========================================================================//
    rss_usage.def(py::init(&pytim::init::rss_usage),
                  "Initialization of RSS measurement class",
                  py::return_value_policy::take_ownership, py::arg("prefix") = "",
                  py::arg("record") = false);
    //------------------------------------------------------------------------//
    rss_usage.def("record", [=](py::object self) { self.cast<rss_usage_t*>()->record(); },
                  "Record the RSS usage");
    //------------------------------------------------------------------------//
    rss_usage.def("__str__",
                  [=](py::object self) {
                      std::stringstream ss;
                      ss << *(self.cast<rss_usage_t*>());
                      return ss.str();
                  },
                  "Stringify the rss usage");
    //------------------------------------------------------------------------//
    rss_usage.def("__iadd__",
                  [=](py::object self, py::object rhs) {
                      *(self.cast<rss_usage_t*>()) += *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Add rss usage");
    //------------------------------------------------------------------------//
    rss_usage.def("__isub__",
                  [=](py::object self, py::object rhs) {
                      *(self.cast<rss_usage_t*>()) -= *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Subtract rss usage");
    //------------------------------------------------------------------------//
    rss_usage.def("__add__",
                  [=](py::object self, py::object rhs) {
                      rss_usage_t* _rss = new rss_usage_t(*(self.cast<rss_usage_t*>()));
                      *_rss += *(rhs.cast<rss_usage_t*>());
                      return _rss;
                  },
                  "Add rss usage", py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    rss_usage.def("__sub__",
                  [=](py::object self, py::object rhs) {
                      rss_usage_t* _rss = new rss_usage_t(*(self.cast<rss_usage_t*>()));
                      *_rss -= *(rhs.cast<rss_usage_t*>());
                      return _rss;
                  },
                  "Subtract rss usage", py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    rss_usage.def("current",
                  [=](py::object self, intmax_t /*_units*/) {
                      return std::get<0>(*self.cast<rss_usage_t*>()).compute_display();
                  },
                  "Return the current rss usage",
                  py::arg("units") = units.attr("megabyte"));
    //------------------------------------------------------------------------//
    rss_usage.def("peak",
                  [=](py::object self, intmax_t /*_units*/) {
                      return std::get<1>(*self.cast<rss_usage_t*>()).compute_display();
                  },
                  "Return the current rss usage",
                  py::arg("units") = units.attr("megabyte"));

    //========================================================================//
    //
    //                      MAIN libtimemory MODULE (part 2)
    //
    //========================================================================//
    tim.attr("timing_manager") = man;
    //------------------------------------------------------------------------//
    tim.def("report",
            [=](bool ign_cutoff, bool endline) {
                manager_t::instance()->report(ign_cutoff, endline);
            },
            "Report the timing manager (default: ign_cutoff = True, endline = "
            "True)",
            py::arg("ign_cutoff") = true, py::arg("endline") = true);
    //------------------------------------------------------------------------//
    tim.def("clear", [=]() { manager_t::instance()->clear(); },
            "Clear the timing manager");
    //------------------------------------------------------------------------//
    tim.def("size", [=]() { return manager_t::instance()->size(); },
            "Size of the timing manager");
    //------------------------------------------------------------------------//
    tim.def("set_exit_action",
            [=](py::function func) {
                auto _func = [=](int errcode) -> void { func(errcode); };
                // typedef tim::signal_settings::signal_function_t
                // signal_function_t;
                typedef std::function<void(int)> signal_function_t;
                using std::placeholders::_1;
                signal_function_t _f = std::bind<void>(_func, _1);
                tim::signal_settings::set_exit_action(_f);
            },
            "Set the exit action when a signal is raised -- function must accept "
            "integer");
    //------------------------------------------------------------------------//

    //========================================================================//
    //
    //      Signals submodule
    //
    //========================================================================//
    py::module sig = tim.def_submodule("signals", "Signals submodule");
    //------------------------------------------------------------------------//
    py::enum_<sys_signal_t> sys_signal_enum(sig, "sys_signal", py::arithmetic(),
                                            "Signals for TiMemory module");
    //------------------------------------------------------------------------//
    sys_signal_enum.value("Hangup", sys_signal_t::sHangup)
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
    // ---------------------------------------------------------------------- //

    //========================================================================//
    //
    //      Options submodule
    //
    //========================================================================//
    py::module opts = tim.def_submodule("options", "I/O options submodule");
    // ---------------------------------------------------------------------- //
    opts.attr("report_file")        = false;
    opts.attr("serial_file")        = true;
    opts.attr("use_timers")         = true;
    opts.attr("max_timer_depth")    = std::numeric_limits<uint16_t>::max();
    opts.attr("report_filename")    = "timing_report.out";
    opts.attr("serial_filename")    = "timing_report.json";
    opts.attr("output_dir")         = ".";
    opts.attr("echo_dart")          = false;
    opts.attr("ctest_notes")        = false;
    opts.attr("matplotlib_backend") = std::string("default");

    using pytim::string_t;

    auto set_report = [=](string_t fname) {
        std::stringstream ss;
        std::string       output_dir = opts.attr("output_dir").cast<std::string>();
        if(fname.find(output_dir) != 0)
            ss << output_dir;
        if(ss.str().length() > 0 && ss.str()[ss.str().length() - 1] != '/')
            ss << "/";
        ss << fname;
        opts.attr("report_filename") = ss.str();
        opts.attr("report_file")     = true;
        return ss.str();
    };

    auto set_serial = [=](string_t fname) {
        std::stringstream ss;
        std::string       output_dir = opts.attr("output_dir").cast<std::string>();
        if(fname.find(output_dir) != 0)
            ss << output_dir;
        if(ss.str().length() > 0 && ss.str()[ss.str().length() - 1] != '/')
            ss << "/";
        ss << fname;
        opts.attr("serial_filename") = ss.str();
        opts.attr("serial_file")     = true;
        return ss.str();
    };
    // ---------------------------------------------------------------------- //
    opts.def("default_max_depth", [=]() { return std::numeric_limits<uint16_t>::max(); },
             "Return the default max depth");
    // ---------------------------------------------------------------------- //
    opts.def("safe_mkdir", &pytim::opt::safe_mkdir,
             "if [ ! -d <directory> ]; then mkdir -p <directory> ; fi");
    // ---------------------------------------------------------------------- //
    opts.def("ensure_directory_exists", &pytim::opt::ensure_directory_exists,
             "mkdir -p $(basename file_path)");
    // ---------------------------------------------------------------------- //
    opts.def("set_report", set_report, "Set the ASCII report filename");
    // ---------------------------------------------------------------------- //
    opts.def("set_serial", set_serial, "Set the JSON serialization filename");
    // ---------------------------------------------------------------------- //
    opts.def("add_arguments", &pytim::opt::add_arguments,
             "Function to add default output arguments", py::arg("parser") = py::none(),
             py::arg("fname") = "");
    // ---------------------------------------------------------------------- //
    opts.def("parse_args", &pytim::opt::parse_args,
             "Function to handle the output arguments");
    // ---------------------------------------------------------------------- //
    opts.def("add_arguments_and_parse", &pytim::opt::add_arguments_and_parse,
             "Combination of timing.add_arguments and timing.parse_args but returns",
             py::arg("parser") = py::none(), py::arg("fname") = "");
    // ---------------------------------------------------------------------- //
    opts.def("add_args_and_parse_known", &pytim::opt::add_args_and_parse_known,
             "Combination of timing.add_arguments and timing.parse_args. Returns "
             "TiMemory args and replaces sys.argv with the unknown args (used to "
             "fix issue with unittest module)",
             py::arg("parser") = py::none(), py::arg("fname") = "");
    // ---------------------------------------------------------------------- //
}
