// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include "libpytimemory.hpp"
#include "timemory/settings.hpp"
#include <pybind11/pybind11.h>

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

//======================================================================================//
//  Python wrappers
//======================================================================================//

PYBIND11_MODULE(libpytimemory, tim)
{
    //----------------------------------------------------------------------------------//
    using pytim::string_t;
    py::add_ostream_redirect(tim, "ostream_redirect");

    //==================================================================================//
    //
    //      Units submodule
    //
    //==================================================================================//
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

    //==================================================================================//
    //
    //      Components submodule
    //
    //==================================================================================//
    py::enum_<TIMEMORY_COMPONENT> components_enum(tim, "component", py::arithmetic(),
                                                  "Components for TiMemory module");
    //----------------------------------------------------------------------------------//
    components_enum.value("wall_clock", WALL_CLOCK)
        .value("sys_clock", SYS_CLOCK)
        .value("user_clock", USER_CLOCK)
        .value("cpu_clock", CPU_CLOCK)
        .value("monotonic_clock", MONOTONIC_CLOCK)
        .value("monotonic_raw_clock", MONOTONIC_RAW_CLOCK)
        .value("thread_cpu_clock", THREAD_CPU_CLOCK)
        .value("process_cpu_clock", PROCESS_CPU_CLOCK)
        .value("cpu_util", CPU_UTIL)
        .value("thread_cpu_util", THREAD_CPU_UTIL)
        .value("process_cpu_util", PROCESS_CPU_UTIL)
        .value("current_rss", CURRENT_RSS)
        .value("peak_rss", PEAK_RSS)
        .value("stack_rss", STACK_RSS)
        .value("data_rss", DATA_RSS)
        .value("num_swap", NUM_SWAP)
        .value("num_io_in", NUM_IO_IN)
        .value("num_io_out", NUM_IO_OUT)
        .value("num_minor_page_faults", NUM_MINOR_PAGE_FAULTS)
        .value("num_major_page_faults", NUM_MAJOR_PAGE_FAULTS)
        .value("num_msg_sent", NUM_MSG_SENT)
        .value("num_msg_recv", NUM_MSG_RECV)
        .value("num_signals", NUM_SIGNALS)
        .value("voluntary_context_switch", VOLUNTARY_CONTEXT_SWITCH)
        .value("priority_context_switch", PRIORITY_CONTEXT_SWITCH)
        .value("cuda_event", CUDA_EVENT)
        .value("papi_array", PAPI_ARRAY)
        .value("cpu_roofline_sp_flops", CPU_ROOFLINE_SP_FLOPS)
        .value("cpu_roofline_dp_flops", CPU_ROOFLINE_DP_FLOPS)
        .value("caliper", CALIPER)
        .value("trip_count", TRIP_COUNT)
        .value("read_bytes", READ_BYTES)
        .value("written_bytes", WRITTEN_BYTES)
        .value("cupti_event", CUPTI_EVENT);

    //==================================================================================//
    //
    //      Signals submodule
    //
    //==================================================================================//
    py::module sig = tim.def_submodule("signals", "Signals submodule");
    //----------------------------------------------------------------------------------//
    py::enum_<sys_signal_t> sys_signal_enum(sig, "sys_signal", py::arithmetic(),
                                            "Signals for TiMemory module");
    //----------------------------------------------------------------------------------//
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

    //==================================================================================//
    //
    //      CUPTI submodule
    //
    //==================================================================================//
    py::module cupti = tim.def_submodule("cupti", "cupti query");

    auto get_available_cupti_events = [=](int device) {
#if defined(TIMEMORY_USE_CUPTI)
        CUdevice cu_device;
        CUDA_DRIVER_API_CALL(cuInit(0));
        CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device, device));
        return tim::cupti::available_events(cu_device);
#else
        tim::consume_parameters(device);
        return py::list();
#endif
    };

    auto get_available_cupti_metrics = [=](int device) {
#if defined(TIMEMORY_USE_CUPTI)
        CUdevice cu_device;
        CUDA_DRIVER_API_CALL(cuInit(0));
        CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device, device));
        auto     ret = tim::cupti::available_metrics(cu_device);
        py::list l;
        for(const auto& itr : ret)
            l.append(py::cast<std::string>(itr));
        return l;
#else
        tim::consume_parameters(device);
        return py::list();
#endif
    };

    cupti.def("available_events", get_available_cupti_events,
              "Return the available CUPTI events", py::arg("device") = 0);
    cupti.def("available_metrics", get_available_cupti_metrics,
              "Return the available CUPTI metric", py::arg("device") = 0);

    //==================================================================================//
    //
    //      Options submodule
    //
    //==================================================================================//
    py::module opts = tim.def_submodule("options", "I/O options submodule");
    //----------------------------------------------------------------------------------//
    opts.attr("report_file")        = false;
    opts.attr("serial_file")        = true;
    opts.attr("use_timers")         = true;
    opts.attr("max_timer_depth")    = std::numeric_limits<uint16_t>::max();
    opts.attr("output_path")        = tim::settings::output_path();
    opts.attr("output_prefix")      = tim::settings::output_prefix();
    opts.attr("echo_dart")          = false;
    opts.attr("ctest_notes")        = false;
    opts.attr("matplotlib_backend") = std::string("default");

    //==================================================================================//
    //
    //      Class declarations
    //
    //==================================================================================//
    py::class_<manager_wrapper>               man(tim, "manager");
    py::class_<tim_timer_t>                   timer(tim, "timer");
    py::class_<auto_timer_t>                  auto_timer(tim, "auto_timer");
    py::class_<component_list_t>              comp_list(tim, "component_tuple");
    py::class_<auto_timer_decorator>          timer_decorator(tim, "timer_decorator");
    py::class_<component_list_decorator>      comp_decorator(tim, "component_decorator");
    py::class_<rss_usage_t>                   rss_usage(tim, "rss_usage");
    py::class_<pytim::decorators::auto_timer> decorate_auto_timer(tim, "decorate_timer");

    //==================================================================================//
    //
    //      Helper lambdas
    //
    //==================================================================================//
    auto set_output = [=](string_t fname) {
        tim::settings::output_path() = opts.attr("output_path").cast<std::string>();
        if(fname.find('/') < fname.length() - 1)
        {
            auto last_slash              = fname.find_last_of('/');
            auto dname                   = fname.substr(0, last_slash + 1);
            fname                        = fname.substr(last_slash + 1);
            tim::settings::output_path() = dname;
        }
        else if(fname.find_last_of('/') == fname.length() - 1)
        {
            auto last_slash              = fname.find_last_of('/');
            auto dname                   = fname.substr(0, last_slash + 1);
            fname                        = "";
            tim::settings::output_path() = dname;
        }
        tim::settings::output_prefix() = fname;
        opts.attr("output_path")       = tim::settings::output_path();
        opts.attr("output_prefix")     = tim::settings::output_prefix();
    };
    //----------------------------------------------------------------------------------//
    auto report = [&](std::string fname) {
        auto _path   = tim::settings::output_path();
        auto _prefix = tim::settings::output_prefix();
        if(fname.length() > 0)
        {
            set_output(fname);
        }

        tim::manager::print<tim::rusage_components_t>();
        tim::manager::print<tim::timing_components_t>();

        if(fname.length() > 0)
        {
            tim::settings::output_path()   = _path;
            tim::settings::output_prefix() = _prefix;
        }
    };
    //----------------------------------------------------------------------------------//
    auto set_rusage_child = [&]() {
#if !defined(_WINDOWS)
        tim::get_rusage_type() = RUSAGE_CHILDREN;
#endif
    };
    //----------------------------------------------------------------------------------//
    auto set_rusage_self = [&]() {
#if !defined(_WINDOWS)
        tim::get_rusage_type() = RUSAGE_SELF;
#endif
    };
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                  MAIN libpytimemory MODULE (part 1)
    //
    //==================================================================================//
    tim.def("report", report, "Print the data", py::arg("filename") = "");
    //----------------------------------------------------------------------------------//
    tim.def("LINE", &pytim::get_line, "Function that emulates __LINE__ macro",
            py::arg("nback") = 1);
    //----------------------------------------------------------------------------------//
    tim.def("FUNC", &pytim::get_func, "Function that emulates __FUNC__ macro",
            py::arg("nback") = 1);
    //----------------------------------------------------------------------------------//
    tim.def("FILE", &pytim::get_file, "Function that emulates __FILE__ macro",
            py::arg("nback") = 2, py::arg("basename_only") = true,
            py::arg("use_dirname") = false, py::arg("noquotes") = false);
    //----------------------------------------------------------------------------------//
    tim.def(
        "set_max_depth", [&](int32_t ndepth) { manager_t::max_depth(ndepth); },
        "Max depth of auto-timers");
    //----------------------------------------------------------------------------------//
    tim.def(
        "get_max_depth", [&]() { return manager_t::max_depth(); },
        "Max depth of auto-timers");
    //----------------------------------------------------------------------------------//
    tim.def(
        "toggle", [&](bool timers_on) { manager_t::enable(timers_on); },
        "Enable/disable auto-timers", py::arg("timers_on") = true);
    //----------------------------------------------------------------------------------//
    tim.def(
        "enable", [&]() { manager_t::enable(true); }, "Enable auto-timers");
    //----------------------------------------------------------------------------------//
    tim.def(
        "disable", [&]() { manager_t::enable(false); }, "Disable auto-timers");
    //----------------------------------------------------------------------------------//
    tim.def(
        "is_enabled", [&]() { return manager_t::is_enabled(); },
        "Return if the auto-timers are enabled or disabled");
    //----------------------------------------------------------------------------------//
    tim.def(
        "enabled", [&]() { return manager_t::is_enabled(); },
        "Return if the auto-timers are enabled or disabled");
    //----------------------------------------------------------------------------------//
    tim.def("enable_signal_detection", &pytim::enable_signal_detection,
            "Enable signal detection", py::arg("signal_list") = py::list());
    //----------------------------------------------------------------------------------//
    tim.def("disable_signal_detection", &pytim::disable_signal_detection,
            "Enable signal detection");
    //----------------------------------------------------------------------------------//
    tim.def(
        "has_mpi_support", [&]() { return tim::mpi::is_supported(); },
        "Return if the TiMemory library has MPI support");
    //----------------------------------------------------------------------------------//
    tim.def("set_rusage_children", set_rusage_child,
            "Set the rusage to record child processes");
    //----------------------------------------------------------------------------------//
    tim.def("set_rusage_self", set_rusage_self,
            "Set the rusage to record child processes");
    //----------------------------------------------------------------------------------//
    tim.def(
        "set_exit_action",
        [&](py::function func) {
            auto _func              = [&](int errcode) -> void { func(errcode); };
            using signal_function_t = std::function<void(int)>;
            using std::placeholders::_1;
            signal_function_t _f = std::bind<void>(_func, _1);
            tim::signal_settings::set_exit_action(_f);
        },
        "Set the exit action when a signal is raised -- function must accept "
        "integer");
    //----------------------------------------------------------------------------------//
    tim.def(
        "timemory_init",
        [&](py::list argv, std::string _prefix, std::string _suffix) {
            if(argv.size() < 1)
                return;
            char* _argv = const_cast<char*>(argv.begin()->cast<std::string>().c_str());
            tim::timemory_init(1, &_argv, _prefix, _suffix);
        },
        "Parse the environment and use argv[0] to set output path",
        py::arg("argv") = py::list(), py::arg("prefix") = "timemory-",
        py::arg("suffix") = "-output");

    //==================================================================================//
    //
    //                          TIMER
    //
    //==================================================================================//
    timer.def(py::init(&pytim::init::timer), "Initialization",
              py::return_value_policy::take_ownership, py::arg("prefix") = "");
    //----------------------------------------------------------------------------------//
    timer.def(
        "real_elapsed",
        [&](py::object pytimer) {
            tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
            auto&        obj    = std::get<0>(_timer);
            return obj.get_display();
        },
        "Elapsed wall clock");
    //----------------------------------------------------------------------------------//
    timer.def(
        "sys_elapsed",
        [&](py::object pytimer) {
            tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
            auto&        obj    = std::get<1>(_timer);
            return obj.get_display();
        },
        "Elapsed system clock");
    //----------------------------------------------------------------------------------//
    timer.def(
        "user_elapsed",
        [&](py::object pytimer) {
            tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
            auto&        obj    = std::get<2>(_timer);
            return obj.get_display();
        },
        "Elapsed user time");
    //----------------------------------------------------------------------------------//
    timer.def(
        "start", [&](py::object pytimer) { pytimer.cast<tim_timer_t*>()->start(); },
        "Start timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "stop", [&](py::object pytimer) { pytimer.cast<tim_timer_t*>()->stop(); },
        "Stop timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "report",
        [&](py::object pytimer) {
            std::cout << *(pytimer.cast<tim_timer_t*>()) << std::endl;
        },
        "Report timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "__str__",
        [&](py::object pytimer) {
            std::stringstream ss;
            ss << *(pytimer.cast<tim_timer_t*>());
            return ss.str();
        },
        "Stringify timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "reset", [&](py::object self) { self.cast<tim_timer_t*>()->reset(); },
        "Reset the timer");
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                          TIMING MANAGER
    //
    //==================================================================================//
    man.attr("text_files") = py::list();
    //----------------------------------------------------------------------------------//
    man.attr("json_files") = py::list();
    //----------------------------------------------------------------------------------//
    man.def(py::init<>(&pytim::init::manager), "Initialization",
            py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    man.def(
        "set_max_depth", [&](py::object, int depth) { manager_t::max_depth(depth); },
        "Set the max depth of the timers");
    //----------------------------------------------------------------------------------//
    man.def(
        "get_max_depth", [&](py::object) { return manager_t::max_depth(); },
        "Get the max depth of the timers");
    //----------------------------------------------------------------------------------//
    man.def("write_ctest_notes", &pytim::manager::write_ctest_notes,
            "Write a CTestNotes.cmake file", py::arg("directory") = ".",
            py::arg("append") = false);
    //----------------------------------------------------------------------------------//
    // man.def(
    //    "clear",
    //    [&](py::object) {
    //        manager_t::instance()->clear(component_list_t("clear", false));
    //    },
    //    "Clear the storage");
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                      AUTO TIMER
    //
    //==================================================================================//
    auto_timer.def(py::init(&pytim::init::auto_timer), "Initialization",
                   py::arg("key") = "", py::arg("report_at_exit") = false,
                   py::arg("nback") = 1, py::arg("added_args") = false,
                   py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    auto_timer.def(
        "__str__",
        [&](py::object _pyauto_timer) {
            std::stringstream _ss;
            auto_timer_t*     _auto_timer = _pyauto_timer.cast<auto_timer_t*>();
            _ss << _auto_timer->get_component_type();
            return _ss.str();
        },
        "Print the auto timer");
    //----------------------------------------------------------------------------------//
    timer_decorator.def(py::init(&pytim::init::timer_decorator), "Initialization",
                        py::return_value_policy::automatic);
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                      TIMEMORY_COMPONENT TUPLE
    //
    //==================================================================================//
    comp_list.def(py::init(&pytim::init::component_list), "Initialization",
                  py::arg("components") = py::list(), py::arg("key") = "",
                  py::arg("report_at_exit") = false, py::arg("nback") = 1,
                  py::arg("added_args") = false, py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "start", [&](py::object pytimer) { pytimer.cast<component_list_t*>()->start(); },
        "Start component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "stop", [&](py::object pytimer) { pytimer.cast<component_list_t*>()->stop(); },
        "Stop component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "report",
        [&](py::object pytimer) {
            std::cout << *(pytimer.cast<component_list_t*>()) << std::endl;
        },
        "Report component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "__str__",
        [&](py::object pytimer) {
            std::stringstream ss;
            ss << *(pytimer.cast<component_list_t*>());
            return ss.str();
        },
        "Stringify component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "reset", [&](py::object self) { self.cast<component_list_t*>()->reset(); },
        "Reset the component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "__str__",
        [&](py::object _pyauto_tuple) {
            std::stringstream _ss;
            component_list_t* _auto_tuple = _pyauto_tuple.cast<component_list_t*>();
            _ss << *_auto_tuple;
            return _ss.str();
        },
        "Print the component tuple");
    //----------------------------------------------------------------------------------//
    comp_decorator.def(py::init(&pytim::init::component_decorator), "Initialization",
                       py::return_value_policy::automatic);
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                      AUTO TIMER DECORATOR
    //
    //==================================================================================//
    decorate_auto_timer.def(
        py::init(&pytim::decorators::init::auto_timer), "Initialization",
        py::arg("key") = "", py::arg("add_args") = false, py::arg("is_class") = false,
        py::arg("report_at_exit") = false, py::return_value_policy::take_ownership);
    decorate_auto_timer.def("__call__", &pytim::decorators::auto_timer::call,
                            "Call operator");

    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                      RSS USAGE
    //
    //==================================================================================//
    rss_usage.def(py::init(&pytim::init::rss_usage),
                  "Initialization of RSS measurement class",
                  py::return_value_policy::take_ownership, py::arg("prefix") = "",
                  py::arg("record") = false);
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "record", [&](py::object self) { self.cast<rss_usage_t*>()->record(); },
        "Record the RSS usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__str__",
        [&](py::object self) {
            std::stringstream ss;
            ss << *(self.cast<rss_usage_t*>());
            return ss.str();
        },
        "Stringify the rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__iadd__",
        [&](py::object self, py::object rhs) {
            *(self.cast<rss_usage_t*>()) += *(rhs.cast<rss_usage_t*>());
            return self;
        },
        "Add rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__isub__",
        [&](py::object self, py::object rhs) {
            *(self.cast<rss_usage_t*>()) -= *(rhs.cast<rss_usage_t*>());
            return self;
        },
        "Subtract rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__add__",
        [&](py::object self, py::object rhs) {
            rss_usage_t* _rss = new rss_usage_t(*(self.cast<rss_usage_t*>()));
            *_rss += *(rhs.cast<rss_usage_t*>());
            return _rss;
        },
        "Add rss usage", py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__sub__",
        [&](py::object self, py::object rhs) {
            rss_usage_t* _rss = new rss_usage_t(*(self.cast<rss_usage_t*>()));
            *_rss -= *(rhs.cast<rss_usage_t*>());
            return _rss;
        },
        "Subtract rss usage", py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "current",
        [&](py::object self, int64_t /*_units*/) {
            return std::get<0>(*self.cast<rss_usage_t*>()).get_display();
        },
        "Return the current rss usage", py::arg("units") = units.attr("megabyte"));
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "peak",
        [&](py::object self, int64_t /*_units*/) {
            return std::get<1>(*self.cast<rss_usage_t*>()).get_display();
        },
        "Return the current rss usage", py::arg("units") = units.attr("megabyte"));

    //==================================================================================//
    //
    //                      MAIN libpytimemory MODULE (part 2)
    //
    //==================================================================================//

    //==================================================================================//
    //
    //      Options submodule
    //
    //==================================================================================//

    // ---------------------------------------------------------------------- //
    opts.def(
        "default_max_depth", [&]() { return std::numeric_limits<uint16_t>::max(); },
        "Return the default max depth");
    // ---------------------------------------------------------------------- //
    opts.def("safe_mkdir", &pytim::opt::safe_mkdir,
             "if [ ! -d <directory> ]; then mkdir -p <directory> ; fi");
    // ---------------------------------------------------------------------- //
    opts.def("ensure_directory_exists", &pytim::opt::ensure_directory_exists,
             "mkdir -p $(basename file_path)");
    // ---------------------------------------------------------------------- //
    opts.def("set_output", set_output,
             "Set the output prefix that extensions are addded to");
    // ---------------------------------------------------------------------- //
    opts.def("add_arguments", &pytim::opt::add_arguments,
             "Function to add default output arguments", py::arg("parser") = py::none(),
             py::arg("fpath") = ".");
    // ---------------------------------------------------------------------- //
    opts.def("parse_args", &pytim::opt::parse_args,
             "Function to handle the output arguments");
    // ---------------------------------------------------------------------- //
    opts.def("add_arguments_and_parse", &pytim::opt::add_arguments_and_parse,
             "Combination of timing.add_arguments and timing.parse_args but returns",
             py::arg("parser") = py::none(), py::arg("fpath") = ".");
    // ---------------------------------------------------------------------- //
    opts.def("add_args_and_parse_known", &pytim::opt::add_args_and_parse_known,
             "Combination of timing.add_arguments and timing.parse_args. Returns "
             "TiMemory args and replaces sys.argv with the unknown args (used to "
             "fix issue with unittest module)",
             py::arg("parser") = py::none(), py::arg("fpath") = ".");
    // ---------------------------------------------------------------------- //
}
