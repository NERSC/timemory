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

#include "pytimemory.hpp"

//============================================================================//
//  Python wrappers
//============================================================================//

PYBIND11_MODULE(timemory, tim)
{
    //------------------------------------------------------------------------//
    py::add_ostream_redirect(tim, "ostream_redirect");
    //------------------------------------------------------------------------//

    //========================================================================//
    //
    //                  MAIN timemory MODULE (part 1)
    //
    //========================================================================//
    tim.def("LINE",
            &pytim::get_line,
            "Function that emulates __LINE__ macro",
            py::arg("nback") = 1);
    //------------------------------------------------------------------------//
    tim.def("FUNC",
            &pytim::get_func,
            "Function that emulates __FUNC__ macro",
            py::arg("nback") = 1);
    //------------------------------------------------------------------------//
    tim.def("FILE",
            &pytim::get_file,
            "Function that emulates __FILE__ macro",
            py::arg("nback") = 2,
            py::arg("basename_only") = true,
            py::arg("use_dirname") = false,
            py::arg("noquotes") = false);
    //------------------------------------------------------------------------//
    tim.def("set_max_depth",
            [=] (int32_t ndepth)
            { manager_t::instance()->set_max_depth(ndepth); },
            "Max depth of auto-timers");
    //------------------------------------------------------------------------//
    tim.def("get_max_depth",
            [=] ()
            { return manager_t::instance()->get_max_depth(); },
            "Max depth of auto-timers");
    //------------------------------------------------------------------------//
    tim.def("toggle",
            [=] (bool timers_on)
            { manager_t::instance()->enable(timers_on); },
            "Enable/disable auto-timers",
            py::arg("timers_on") = true);
    //------------------------------------------------------------------------//
    tim.def("enable",
            [=] ()
            { manager_t::instance()->enable(true); },
            "Enable auto-timers");
    //------------------------------------------------------------------------//
    tim.def("disable",
            [=] ()
            { manager_t::instance()->enable(false); },
            "Disable auto-timers");
    //------------------------------------------------------------------------//
    tim.def("is_enabled",
            [=] ()
            { return manager_t::instance()->is_enabled(); },
            "Return if the auto-timers are enabled or disabled");
    //------------------------------------------------------------------------//
    tim.def("enabled",
            [=] ()
            { return manager_t::instance()->is_enabled(); },
            "Return if the auto-timers are enabled or disabled");
    //------------------------------------------------------------------------//
    tim.def("enable_signal_detection",
            &pytim::enable_signal_detection,
            "Enable signal detection",
            py::arg("signal_list") = py::list());
    //------------------------------------------------------------------------//
    tim.def("disable_signal_detection",
            &pytim::disable_signal_detection,
            "Enable signal detection");
    //------------------------------------------------------------------------//
    tim.def("set_default_format",
            &pytim::format::set_timer_default,
            "Set the default format of the timers");
    //------------------------------------------------------------------------//
    tim.def("get_default_format",
            &pytim::format::get_timer_default,
            "Get the default format of the timers");
    //------------------------------------------------------------------------//
    tim.def("has_mpi_support",
            [=] ()
            { return tim::has_mpi_support(); },
            "Return if the TiMemory library has MPI support");
    //------------------------------------------------------------------------//
    tim.def("get_missing_report",
            [=] ()
            {
                std::stringstream _ss;
                manager_t::instance()->write_missing(_ss);
                return _ss.str();
            },
            "Get TiMemory missing as string");
    //------------------------------------------------------------------------//


    //========================================================================//
    //
    //      Units submodule
    //
    //========================================================================//
    py::module units = tim.def_submodule("units",
                                         "units for timing and memory");

    units.attr("psec") = tim::units::psec;
    units.attr("nsec") = tim::units::nsec;
    units.attr("usec") = tim::units::usec;
    units.attr("msec") = tim::units::msec;
    units.attr("csec") = tim::units::csec;
    units.attr("dsec") = tim::units::dsec;
    units.attr("sec") = tim::units::sec;
    units.attr("byte") = tim::units::byte;
    units.attr("kilobyte") = tim::units::kilobyte;
    units.attr("megabyte") = tim::units::megabyte;
    units.attr("gigabyte") = tim::units::gigabyte;
    units.attr("terabyte") = tim::units::terabyte;
    units.attr("petabyte") = tim::units::petabyte;

    //========================================================================//
    //
    //      Format submodule
    //
    //========================================================================//
    py::module fmt = tim.def_submodule("format",
                                       "timing and memory format submodule");

    //------------------------------------------------------------------------//
    //      format.timer
    //------------------------------------------------------------------------//
    py::class_<tim::format::timer> timing_fmt(fmt, "timer");

    timing_fmt.def(py::init(&pytim::init::timing_format),
                   "Initialize timing formatter",
                   py::return_value_policy::take_ownership,
                   py::arg("prefix") = "",
                   py::arg("format") = timer_format_t::default_format(),
                   py::arg("unit") = timer_format_t::default_unit(),
                   py::arg("rss_format") = py::none(),
                   py::arg("align_width") = false);

    timing_fmt.def("set_default",
                   [=] (PYOBJECT_SELF py::object _val)
                   {
                       timer_format_t* _fmt = _val.cast<timer_format_t*>();
                       timer_format_t::set_default(*_fmt);
                   },
                   "Set the default timer format");
    timing_fmt.def("get_default",
                   [=] (PYOBJECT_SELF_PARAM)
                   {
                       timer_format_t _fmt = timer_format_t::get_default();
                       return new timer_format_t(_fmt);
                   },
                   "Get the default timer format");
    timing_fmt.def("set_default_format",
                   [=] (PYOBJECT_SELF std::string _val)
                   { timer_format_t::default_format(_val); },
                   "Set the default timer format");
    timing_fmt.def("set_default_rss_format",
                   [=] (PYOBJECT_SELF py::object _val)
                   {
                       try
                       {
                           rss_format_t* _rss = _val.cast<rss_format_t*>();
                           timer_format_t::default_rss_format(*_rss);
                       }
                       catch(...)
                       {
                           std::string _fmt = _val.cast<std::string>();
                           auto _rss = timer_format_t::default_rss_format();
                           _rss.format(_fmt);
                           timer_format_t::default_rss_format(_rss);
                       }
                   },
                   "Set the default timer RSS format");
    timing_fmt.def("set_default_precision",
                   [=] (PYOBJECT_SELF const int16_t& _prec)
                   { timer_format_t::default_precision(_prec); },
                   "Set the default timer precision");
    timing_fmt.def("set_default_unit",
                   [=] (PYOBJECT_SELF const int64_t& _unit)
                   { timer_format_t::default_unit(_unit); },
                   "Set the default timer units");
    timing_fmt.def("set_default_width",
                   [=] (PYOBJECT_SELF const int16_t& _w)
                   { timer_format_t::default_width(_w); },
                   "Set the default timer field width");

    timing_fmt.def("copy_from",
                   [=] (py::object self, py::object rhs)
                   {
                       timer_format_t* _self = self.cast<timer_format_t*>();
                       timer_format_t* _rhs = rhs.cast<timer_format_t*>();
                       _self->copy_from(_rhs);
                       return self;
                   },
                   "Copy for format, precision, unit, width, etc. from another format");
    timing_fmt.def("set_format",
                   [=] (py::object obj, std::string _val)
                   { obj.cast<timer_format_t*>()->format(_val); },
                   "Set the timer format");
    timing_fmt.def("set_rss_format",
                   [=] (py::object obj, py::object _val)
                   {
                       rss_format_t* _rss = _val.cast<rss_format_t*>();
                       obj.cast<timer_format_t*>()->rss_format(*_rss);
                   },
                   "Set the timer RSS format");
    timing_fmt.def("set_precision",
                   [=] (py::object obj, const int16_t& _prec)
                   { obj.cast<timer_format_t*>()->precision(_prec); },
                   "Set the timer precision");
    timing_fmt.def("set_unit",
                   [=] (py::object obj, const int64_t& _unit)
                   { obj.cast<timer_format_t*>()->unit(_unit); },
                   "Set the timer units");
    timing_fmt.def("set_width",
                   [=] (py::object obj, const int16_t& _w)
                   { obj.cast<timer_format_t*>()->width(_w); },
                   "Set the timer field width");
    timing_fmt.def("set_use_align_width",
                   [=] (py::object obj, bool _val)
                   { obj.cast<timer_format_t*>()->align_width(_val); },
                   "Set the timer to use the alignment width");
    timing_fmt.def("set_prefix",
                   [=] (py::object self, std::string prefix)
                   {
                       timer_format_t* _self = self.cast<timer_format_t*>();
                       _self->prefix(prefix);
                   },
                   "Set the prefix of timer format");
    timing_fmt.def("set_suffix",
                   [=] (py::object self, std::string suffix)
                   {
                       timer_format_t* _self = self.cast<timer_format_t*>();
                       _self->suffix(suffix);
                   },
                   "Set the suffix of timer format");

    //------------------------------------------------------------------------//
    //      format.rss
    //------------------------------------------------------------------------//
    py::class_<tim::format::rss> memory_fmt(fmt, "rss");

    memory_fmt.def(py::init(&pytim::init::memory_format),
                   "Initialize memory formatter",
                   py::return_value_policy::take_ownership,
                   py::arg("prefix") = "",
                   py::arg("format") = rss_format_t::default_format(),
                   py::arg("unit") = rss_format_t::default_unit(),
                   py::arg("align_width") = false);

    memory_fmt.def("set_default",
                   [=] (PYOBJECT_SELF py::object _val)
                   {
                       rss_format_t* _fmt = _val.cast<rss_format_t*>();
                       rss_format_t::set_default(*_fmt);
                   },
                   "Set the default RSS format");
    memory_fmt.def("get_default",
                   [=] (PYOBJECT_SELF_PARAM)
                   {
                       rss_format_t _fmt = rss_format_t::get_default();
                       return new rss_format_t(_fmt);
                   },
                   "Get the default RSS format");
    memory_fmt.def("set_default_format",
                   [=] (PYOBJECT_SELF std::string _val)
                   { rss_format_t::default_format(_val); },
                   "Set the default RSS format");
    memory_fmt.def("set_default_precision",
                   [=] (PYOBJECT_SELF const int16_t& _prec)
                   { rss_format_t::default_precision(_prec); },
                   "Set the default RSS precision");
    memory_fmt.def("set_default_unit",
                   [=] (PYOBJECT_SELF const int64_t& _unit)
                   { rss_format_t::default_unit(_unit); },
                   "Set the default RSS units");
    memory_fmt.def("set_default_width",
                   [=] (PYOBJECT_SELF const int16_t& _w)
                   { rss_format_t::default_width(_w); },
                   "Set the default RSS field width");

    memory_fmt.def("copy_from",
                   [=] (py::object self, py::object rhs)
                   {
                       rss_format_t* _self = self.cast<rss_format_t*>();
                       rss_format_t* _rhs = rhs.cast<rss_format_t*>();
                       _self->copy_from(_rhs);
                       return self;
                   },
                   "Copy for format, precision, unit, width, etc. from another format");
    memory_fmt.def("set_format",
                   [=] (py::object obj, std::string _val)
                   { obj.cast<rss_format_t*>()->format(_val); },
                   "Set the RSS format");
    memory_fmt.def("set_precision",
                   [=] (py::object obj, const int16_t& _prec)
                   { obj.cast<rss_format_t*>()->precision(_prec); },
                   "Set the RSS precision");
    memory_fmt.def("set_unit",
                   [=] (py::object obj, const int64_t& _unit)
                   { obj.cast<rss_format_t*>()->unit(_unit); },
                   "Set the RSS units");
    memory_fmt.def("set_width",
                   [=] (py::object obj, const int16_t& _w)
                   { obj.cast<rss_format_t*>()->width(_w); },
                   "Set the RSS field width");
    memory_fmt.def("set_use_align_width",
                   [=] (py::object obj, bool _val)
                   { obj.cast<rss_format_t*>()->align_width(_val); },
                   "Set the RSS to use the alignment width");
    memory_fmt.def("set_prefix",
                   [=] (py::object self, std::string prefix)
                   {
                       rss_format_t* _self = self.cast<rss_format_t*>();
                       _self->prefix(prefix);
                   },
                   "Set the prefix of RSS format");
    memory_fmt.def("set_suffix",
                   [=] (py::object self, std::string suffix)
                   {
                       rss_format_t* _self = self.cast<rss_format_t*>();
                       _self->suffix(suffix);
                   },
                   "Set the suffix of RSS format");

    //------------------------------------------------------------------------//
    //  Class declarations
    //------------------------------------------------------------------------//
    py::class_<manager_wrapper>         man             (tim, "manager");
    py::class_<tim_timer_t>             timer           (tim, "timer");
    py::class_<auto_timer_t>            auto_timer      (tim, "auto_timer");
    py::class_<auto_timer_decorator>    timer_decorator (tim, "timer_decorator");
    py::class_<rss_usage_t>             rss_usage       (tim, "rss_usage");
    py::class_<rss_delta_t>             rss_delta       (tim, "rss_delta");
    //py::class_<manager_wrapper,
    //           std::unique_ptr<manager_wrapper, py::nodelete>>
    //        man(tim, "manager");


    //========================================================================//
    //
    //                          TIMER
    //
    //========================================================================//
    timer.def(py::init(&pytim::init::timer),
              "Initialization",
              py::return_value_policy::take_ownership,
              py::arg("prefix") = "", py::arg("format") = "");
    //------------------------------------------------------------------------//
    timer.def("real_elapsed",
              [=] (py::object timer)
              { return timer.cast<tim_timer_t*>()->real_elapsed(); },
              "Elapsed wall clock");
    //------------------------------------------------------------------------//
    timer.def("sys_elapsed",
              [=] (py::object timer)
              { return timer.cast<tim_timer_t*>()->system_elapsed(); },
              "Elapsed system clock");
    //------------------------------------------------------------------------//
    timer.def("user_elapsed",
              [=] (py::object timer)
              { return timer.cast<tim_timer_t*>()->user_elapsed(); },
              "Elapsed user time");
    //------------------------------------------------------------------------//
    timer.def("start",
              [=] (py::object timer)
              { timer.cast<tim_timer_t*>()->start(); },
              "Start timer");
    //------------------------------------------------------------------------//
    timer.def("stop",
              [=] (py::object timer)
              { timer.cast<tim_timer_t*>()->stop(); },
              "Stop timer");
    //------------------------------------------------------------------------//
    timer.def("report",
              [=] (py::object timer, bool ign_cutoff = true)
              { timer.cast<tim_timer_t*>()->print(ign_cutoff); },
              "Report timer",
              py::arg("ign_cutoff") = true);
    //------------------------------------------------------------------------//
    timer.def("__str__",
              [=] (py::object timer, bool ign_cutoff = true)
              { return timer.cast<tim_timer_t*>()->as_string(ign_cutoff); },
              "Stringify timer",
              py::arg("ign_cutoff") = true);
    //------------------------------------------------------------------------//
    timer.def("__iadd__",
             [=] (py::object timer, py::object _rss)
             {
                 *(timer.cast<tim_timer_t*>()) +=
                         *(_rss.cast<rss_usage_t*>());
                 return timer;
             },
             "Add RSS measurement");
    //------------------------------------------------------------------------//
    timer.def("__isub__",
             [=] (py::object timer, py::object _rss)
             {
                 *(timer.cast<tim_timer_t*>()) -=
                         *(_rss.cast<rss_usage_t*>());
                 return timer;
             },
             "Subtract RSS measurement");
    //------------------------------------------------------------------------//
    timer.def("get_format",
              [=] (py::object self)
              {
                  tim_timer_t* _self = self.cast<tim_timer_t*>();
                  auto _fmt = _self->format();
                  if(!_fmt.get())
                  {
                      _self->set_format(timer_format_t());
                      _fmt = _self->format();
                  }
                  return _fmt.get();
              },
              "Set the format of the timer",
              py::return_value_policy::reference_internal);
    //------------------------------------------------------------------------//
    timer.def("set_format",
              [=] (py::object timer, py::object fmt)
              {
                  tim_timer_t* _timer = timer.cast<tim_timer_t*>();
                  timer_format_t* _fmt = fmt.cast<timer_format_t*>();
                  _timer->set_format(*_fmt);
              },
              "Set the format of the timer");
    //------------------------------------------------------------------------//
    timer.def("reset",
              [=] (py::object self)
              {
                  self.cast<tim_timer_t*>()->reset();
              },
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
    man.def("report",
             &pytim::manager::report,
             "Report timing manager",
             py::arg("ign_cutoff") = false,
             py::arg("serialize") = false,
             py::arg("serial_filename") = "");
    //------------------------------------------------------------------------//
    man.def("__str__",
             [=] (py::object man)
             {
                 manager_t* _man
                         = man.cast<manager_wrapper*>()->get();
                 std::stringstream ss;
                 bool ign_cutoff = true;
                 bool endline = false;
                 _man->report(ss, ign_cutoff, endline);
                 return ss.str();
             },
             "Stringify the timing manager report");
    //------------------------------------------------------------------------//
    man.def("set_output_file",
             [=] (py::object man, std::string fname)
             {
                 manager_t* _man = man.cast<manager_wrapper*>()->get();
                 auto locals = py::dict("fname"_a = fname);
                 py::exec(R"(
                          import timemory as tim
                          tim.options.set_report(fname)
                          )", py::globals(), locals);
                 _man->set_output_stream(fname);
             },
             "Set the output stream file");
    //------------------------------------------------------------------------//
    man.def("size",
             [=] (py::object man)
             { return man.cast<manager_wrapper*>()->get()->size(); },
             "Size of timing manager");
    //------------------------------------------------------------------------//
    man.def("clear",
             [=] (py::object man)
             { man.cast<manager_wrapper*>()->get()->clear(); },
             "Clear the timing manager");
    //------------------------------------------------------------------------//
    man.def("write_missing",
            [=] (py::object man, std::string fname)
            {
                auto locals = py::dict("fname"_a = fname);
                py::exec(R"(
                         import timemory.options as options
                         options.ensure_directory_exists(fname)
                         )",
                         py::globals(), locals);
                man.cast<manager_wrapper*>()->get()->write_missing(fname);
            },
            "Write TiMemory missing to file");
    //------------------------------------------------------------------------//
    man.def("serialize",
             &pytim::manager::serialize,
             "Serialize the timing manager to JSON",
             py::arg("fname") = "");
    //------------------------------------------------------------------------//
    man.def("set_max_depth",
             [=] (py::object man, int depth)
             { man.cast<manager_wrapper*>()->get()->set_max_depth(depth); },
             "Set the max depth of the timers");
    //------------------------------------------------------------------------//
    man.def("get_max_depth",
             [=] (py::object man)
             { return man.cast<manager_wrapper*>()->get()->get_max_depth(); },
             "Get the max depth of the timers");
    //------------------------------------------------------------------------//
    man.def("at",
             [=] (py::object man, int i)
             {
                 tim_timer_t& _t = man.cast<manager_wrapper*>()->get()->at(i);
                 return &_t;
             },
             "Set the max depth of the timers",
             py::return_value_policy::reference);
    //------------------------------------------------------------------------//
    man.def("merge",
             [=] (py::object man, bool div_clocks)
             { man.cast<manager_wrapper*>()->get()->merge(div_clocks); },
             "Merge the thread-local timers",
             py::arg("div_clocks") = true);
    //------------------------------------------------------------------------//
    man.def("json",
             [=] (py::object man)
             {
                 std::stringstream ss;
                 man.cast<manager_wrapper*>()->get()->write_json(ss);
                 py::module _json = py::module::import("json");
                 return _json.attr("loads")(ss.str());
             }, "Get JSON serialization of timing manager");
    //------------------------------------------------------------------------//
    man.def("__iadd__",
             [=] (py::object man, py::object _rss)
             {
                 *(man.cast<manager_wrapper*>()->get()) +=
                         *(_rss.cast<rss_usage_t*>());
                 return man;
             },
             "Add RSS measurement");
    //------------------------------------------------------------------------//
    man.def("__isub__",
             [=] (py::object man, py::object _rss)
             {
                 *(man.cast<manager_wrapper*>()->get()) -=
                         *(_rss.cast<rss_usage_t*>());
                 return man;
             },
             "Subtract an rss usage from the entire list of timer");
    //------------------------------------------------------------------------//
    man.def("write_ctest_notes",
             &pytim::manager::write_ctest_notes,
             "Write a CTestNotes.cmake file",
             py::arg("directory") = ".",
             py::arg("append") = false);
    //------------------------------------------------------------------------//
    man.def("start_total_timer",
            [=] (py::object self)
            {
                 self.cast<manager_wrapper*>()->get()->start_total_timer();
            },
            "Start the global timer (only use if explicitly stopped)");
    //------------------------------------------------------------------------//
    man.def("stop_total_timer",
            [=] (py::object self)
            {
                 self.cast<manager_wrapper*>()->get()->stop_total_timer();
            },
            "Stop the global timer (for explicit measurement)");
    //------------------------------------------------------------------------//
    man.def("reset_total_timer",
            [=] (py::object self)
            {
                 self.cast<manager_wrapper*>()->get()->reset_total_timer();
            },
            "Reset the global timer (for explicit measurement)");
    //------------------------------------------------------------------------//
    man.def("update_total_timer_format",
            [=] (py::object self)
            {
                self.cast<manager_wrapper*>()->get()->update_total_timer_format();
            },
            "Update the format of the total timer to the default format");
    //------------------------------------------------------------------------//


    //========================================================================//
    //
    //                      AUTO TIMER
    //
    //========================================================================//
    auto_timer.def(py::init(&pytim::init::auto_timer),
                   "Initialization",
                   py::arg("key") = "",
                   py::arg("report_at_exit") = false,
                   py::arg("nback") = 1,
                   py::arg("added_args") = false,
                   py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    auto_timer.def("local_timer",
                   [=] (py::object _auto_timer)
                   { return _auto_timer.cast<auto_timer_t*>()->local_timer(); },
                   "Get the timer for the auto-timer instance");
    //------------------------------------------------------------------------//
    auto_timer.def("global_timer",
                   [=] (py::object _auto_timer)
                   { return _auto_timer.cast<auto_timer_t*>()->local_timer().summation_timer(); },
                   "Get the timer for all the auto-timer instances (from manager)");
    //------------------------------------------------------------------------//
    auto_timer.def("__str__",
                   [=] (py::object _pyauto_timer)
                   {
                       std::stringstream _ss;
                       auto_timer_t* _auto_timer
                               = _pyauto_timer.cast<auto_timer_t*>();
                       tim_timer_t _local = _auto_timer->local_timer();
                       _local.stop();
                       tim_timer_t _global = *_auto_timer->local_timer().summation_timer();
                       _global += _local;
                       _global.format()->align_width(false);
                       _global.report(_ss, false, true);
                       return _ss.str();
                   },
                   "Print the auto timer");
    //------------------------------------------------------------------------//
    timer_decorator.def(py::init(&pytim::init::timer_decorator),
                        "Initialization",
                        py::return_value_policy::automatic);
    //------------------------------------------------------------------------//


    //========================================================================//
    //
    //                      RSS USAGE
    //
    //========================================================================//
    rss_usage.def(py::init(&pytim::init::rss_usage),
                  "Initialization of RSS measurement class",
                  py::return_value_policy::take_ownership,
                  py::arg("prefix") = "",
                  py::arg("record") = false,
                  py::arg("format") = "");
    //------------------------------------------------------------------------//
    rss_usage.def("record",
                  [=] (py::object self)
                  {
                      self.cast<rss_usage_t*>()->record();
                  },
                  "Record the RSS usage");
    //------------------------------------------------------------------------//
    rss_usage.def("__str__",
                  [=] (py::object self)
                  {
                      std::stringstream ss;
                      ss << *(self.cast<rss_usage_t*>());
                      return ss.str();
                  },
                  "Stringify the rss usage");
    //------------------------------------------------------------------------//
    rss_usage.def("__iadd__",
                  [=] (py::object self, py::object rhs)
                  {
                      *(self.cast<rss_usage_t*>())
                            += *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Add rss usage");
    //------------------------------------------------------------------------//
    rss_usage.def("__isub__",
                  [=] (py::object self, py::object rhs)
                  {
                      *(self.cast<rss_usage_t*>())
                            -= *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Subtract rss usage");
    //------------------------------------------------------------------------//
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
    //------------------------------------------------------------------------//
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
    //------------------------------------------------------------------------//
    rss_usage.def("current",
                  [=] (py::object self, int64_t _units = tim::units::megabyte)
                  {
                      return self.cast<rss_usage_t*>()->current(_units);
                  },
                  "Return the current rss usage",
                  py::arg("units") = units.attr("megabyte"));
    //------------------------------------------------------------------------//
    rss_usage.def("peak",
                  [=] (py::object self, int64_t _units = tim::units::megabyte)
                  {
                      return self.cast<rss_usage_t*>()->peak(_units);
                  },
                  "Return the current rss usage",
                  py::arg("units") = units.attr("megabyte"));
    //------------------------------------------------------------------------//
    rss_usage.def("get_format",
              [=] (py::object self)
              {
                  rss_usage_t* _self = self.cast<rss_usage_t*>();
                  auto _fmt = _self->format();
                  if(!_fmt.get())
                  {
                      _self->set_format(rss_format_t());
                      _fmt = _self->format();
                  }
                  return _fmt.get();
              },
              "Set the format of the RSS usage",
              py::return_value_policy::reference_internal);
    //------------------------------------------------------------------------//
    rss_usage.def("set_format",
                  [=] (py::object rss, py::object fmt)
                  {
                      rss_usage_t* _rss = rss.cast<rss_usage_t*>();
                      rss_format_t* _fmt = fmt.cast<rss_format_t*>();
                      _rss->set_format(*_fmt);
                  },
                  "Set the format of the RSS usage");
    //------------------------------------------------------------------------//


    //========================================================================//
    //
    //                      RSS USAGE DELTA
    //
    //========================================================================//
    rss_delta.def(py::init(&pytim::init::rss_delta),
                  "Initialization of RSS measurement class",
                  py::return_value_policy::take_ownership,
                  py::arg("prefix") = "",
                  py::arg("format") = "");
    //------------------------------------------------------------------------//
    rss_delta.def("init",
                  [=] (py::object self)
                  {
                      self.cast<rss_delta_t*>()->init();
                  },
                  "Initialize the RSS delta usage");
    //------------------------------------------------------------------------//
    rss_delta.def("record",
                  [=] (py::object self)
                  {
                      self.cast<rss_delta_t*>()->record();
                  },
                  "Record the RSS delta usage");
    //------------------------------------------------------------------------//
    rss_delta.def("__str__",
                  [=] (py::object self)
                  {
                      std::stringstream ss;
                      ss << *(self.cast<rss_delta_t*>());
                      return ss.str();
                  },
                  "Stringify the RSS delta usage");
    //------------------------------------------------------------------------//
    rss_delta.def("__iadd__",
                  [=] (py::object self, py::object rhs)
                  {
                      *(self.cast<rss_delta_t*>())
                            += *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Add rss delta usage");
    //------------------------------------------------------------------------//
    rss_delta.def("__isub__",
                  [=] (py::object self, py::object rhs)
                  {
                      *(self.cast<rss_delta_t*>())
                            -= *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Subtract rss delta usage");
    //------------------------------------------------------------------------//
    rss_delta.def("__add__",
                  [=] (py::object self, py::object rhs)
                  {
                      rss_delta_t* _rss
                            = new rss_delta_t(*(self.cast<rss_delta_t*>()));
                      *_rss += *(rhs.cast<rss_usage_t*>());
                      return _rss;
                  },
                  "Add rss delta usage",
                  py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    rss_delta.def("__sub__",
                  [=] (py::object self, py::object rhs)
                  {
                      rss_delta_t* _rss
                            = new rss_delta_t(*(self.cast<rss_delta_t*>()));
                      *_rss -= *(rhs.cast<rss_usage_t*>());
                      return _rss;
                  },
                  "Subtract delta rss usage",
                  py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    rss_delta.def("total",
                  [=] (py::object self)
                  {
                      return new rss_usage_t(self.cast<rss_delta_t*>()->total());
                  },
                  "Return the total rss usage",
                  py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    rss_delta.def("self",
                  [=] (py::object self)
                  {
                      return new rss_usage_t(self.cast<rss_delta_t*>()->self());
                  },
                  "Return the self rss usage",
                  py::return_value_policy::take_ownership);
    //------------------------------------------------------------------------//
    rss_delta.def("get_format",
              [=] (py::object self)
              {
                  rss_delta_t* _self = self.cast<rss_delta_t*>();
                  auto _fmt = _self->format();
                  if(!_fmt.get())
                  {
                      _self->set_format(rss_format_t());
                      _fmt = _self->format();
                  }
                  return _fmt.get();
              },
              "Set the format of the RSS usage",
              py::return_value_policy::reference_internal);
    //------------------------------------------------------------------------//
    rss_delta.def("set_format",
                  [=] (py::object rss, py::object fmt)
                  {
                      rss_delta_t* _rss = rss.cast<rss_delta_t*>();
                      rss_format_t* _fmt = fmt.cast<rss_format_t*>();
                      _rss->set_format(*_fmt);
                  },
                  "Set the format of the RSS usage");
    //------------------------------------------------------------------------//

    //========================================================================//
    //
    //                      MAIN timemory MODULE (part 2)
    //
    //========================================================================//
    tim.attr("timing_manager") = man;
    //------------------------------------------------------------------------//
    tim.def("report",
            [=] (bool ign_cutoff = true, bool endline = true)
            { manager_t::instance()->report(ign_cutoff, endline); },
            "Report the timing manager (default: ign_cutoff = True, endline = True)",
            py::arg("ign_cutoff") = true, py::arg("endline") = true);
    //------------------------------------------------------------------------//
    tim.def("clear",
            [=] ()
            { manager_t::instance()->clear(); },
            "Clear the timing manager");
    //------------------------------------------------------------------------//
    tim.def("size",
            [=] ()
            { return manager_t::instance()->size(); },
            "Size of the timing manager");
    //------------------------------------------------------------------------//
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
    // ---------------------------------------------------------------------- //


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
    opts.attr("report_filename") = "timing_report.out";
    opts.attr("serial_filename") = "timing_report.json";
    opts.attr("output_dir") = ".";
    opts.attr("echo_dart") = false;
    opts.attr("ctest_notes") = false;
    opts.attr("matplotlib_backend") = std::string("default");

    using pytim::string_t;

    auto set_report = [=] (string_t fname)
    {
        std::stringstream ss;
        std::string output_dir = opts.attr("output_dir").cast<std::string>();
        if(fname.find(output_dir) != 0)
            ss << output_dir;
        if(ss.str().length() > 0 && ss.str()[ss.str().length()-1] != '/')
            ss << "/";
        ss << fname;
        opts.attr("report_filename") = ss.str().c_str();
        opts.attr("report_file") = true;
        return ss.str();
    };

    auto set_serial = [=] (string_t fname)
    {
        std::stringstream ss;
        std::string output_dir = opts.attr("output_dir").cast<std::string>();
        if(fname.find(output_dir) != 0)
            ss << output_dir;
        if(ss.str().length() > 0 && ss.str()[ss.str().length()-1] != '/')
            ss << "/";
        ss << fname;
        opts.attr("serial_filename") = ss.str().c_str();
        opts.attr("serial_file") = true;
        return ss.str();
    };
    // ---------------------------------------------------------------------- //
    opts.def("default_max_depth",
            [=]() { return std::numeric_limits<uint16_t>::max(); },
            "Return the default max depth");
    // ---------------------------------------------------------------------- //
    opts.def("safe_mkdir",
             &pytim::opt::safe_mkdir,
             "if [ ! -d <directory> ]; then mkdir -p <directory> ; fi");
    // ---------------------------------------------------------------------- //
    opts.def("ensure_directory_exists",
             &pytim::opt::ensure_directory_exists,
             "mkdir -p $(basename file_path)");
    // ---------------------------------------------------------------------- //
    opts.def("set_report",
             set_report,
             "Set the ASCII report filename");
    // ---------------------------------------------------------------------- //
    opts.def("set_serial",
             set_serial,
             "Set the JSON serialization filename");
    // ---------------------------------------------------------------------- //
    opts.def("add_arguments",
             &pytim::opt::add_arguments,
             "Function to add default output arguments",
             py::arg("parser") = py::none(), py::arg("fname") = "");
    // ---------------------------------------------------------------------- //
    opts.def("parse_args",
             &pytim::opt::parse_args,
             "Function to handle the output arguments");
    // ---------------------------------------------------------------------- //
    opts.def("add_arguments_and_parse",
             &pytim::opt::add_arguments_and_parse,
             "Combination of timing.add_arguments and timing.parse_args but returns",
             py::arg("parser") = py::none(), py::arg("fname") = "");
    // ---------------------------------------------------------------------- //
    opts.def("add_args_and_parse_known",
             &pytim::opt::add_args_and_parse_known,
             "Combination of timing.add_arguments and timing.parse_args. Returns "
             "TiMemory args and replaces sys.argv with the unknown args (used to "
             "fix issue with unittest module)",
             py::arg("parser") = py::none(), py::arg("fname") = "");
    // ---------------------------------------------------------------------- //

}
