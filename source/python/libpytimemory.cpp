// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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

#include "timemory/components/ompt.hpp"
//
#include "libpytimemory.hpp"
//
#include "timemory/components/definition.hpp"

//======================================================================================//

#if defined(TIMEMORY_USE_MPIP_LIBRARY)
extern "C"
{
    extern uint64_t timemory_start_mpip();
    extern uint64_t timemory_stop_mpip(uint64_t);
}
#endif

//======================================================================================//

#if defined(TIMEMORY_USE_OMPT_LIBRARY)
extern "C"
{
    extern uint64_t timemory_start_ompt();
    extern uint64_t timemory_stop_ompt(uint64_t);
}
#endif

//======================================================================================//

manager_wrapper::manager_wrapper()
: m_manager(manager_t::instance().get())
{}

//--------------------------------------------------------------------------------------//

manager_wrapper::~manager_wrapper() {}

//--------------------------------------------------------------------------------------//

manager_t*
manager_wrapper::get()
{
    return manager_t::instance().get();
}

//======================================================================================//

struct pyenumeration
{
    template <size_t Idx>
    static void generate(py::enum_<TIMEMORY_NATIVE_COMPONENT>& _pyenum)
    {
        using T = typename tim::component::enumerator<Idx>::type;
        if(std::is_same<T, tim::component::placeholder<tim::component::nothing>>::value)
            return;
        using property_t = tim::component::properties<T>;
        std::string id   = property_t::enum_string();
        for(auto& itr : id)
            itr = tolower(itr);
        _pyenum.value(id.c_str(),
                      static_cast<TIMEMORY_NATIVE_COMPONENT>(property_t::value),
                      T::description().c_str());
    }

    template <size_t... Idx>
    static void components(py::enum_<TIMEMORY_NATIVE_COMPONENT>& _pyenum,
                           std::index_sequence<Idx...>)
    {
        TIMEMORY_FOLD_EXPRESSION(pyenumeration::generate<Idx>(_pyenum));
    }
};

//======================================================================================//
//  Python wrappers
//======================================================================================//

PYBIND11_MODULE(libpytimemory, tim)
{
    //----------------------------------------------------------------------------------//
    //
    static auto              _master_manager = manager_t::master_instance();
    static thread_local auto _worker_manager = manager_t::instance();
    if(_worker_manager != _master_manager)
    {
        printf("[%s]> tim::manager :: master != worker : %p vs. %p\n", __FUNCTION__,
               (void*) _master_manager.get(), (void*) _worker_manager.get());
    }

    //----------------------------------------------------------------------------------//
    //
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

    py::enum_<TIMEMORY_NATIVE_COMPONENT> components_enum(
        tim, "component", py::arithmetic(), "Components for timemory module");

    //----------------------------------------------------------------------------------//

    pyenumeration::components(components_enum,
                              std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});

    //==================================================================================//
    //
    //      Signals submodule
    //
    //==================================================================================//
    py::module sig = tim.def_submodule("signals", "Signals submodule");
    //----------------------------------------------------------------------------------//
    py::enum_<sys_signal_t> sys_signal_enum(sig, "sys_signal", py::arithmetic(),
                                            "Signals for timemory module");
    //----------------------------------------------------------------------------------//
    sys_signal_enum.value("Hangup", sys_signal_t::Hangup)
        .value("Interrupt", sys_signal_t::Interrupt)
        .value("Quit", sys_signal_t::Quit)
        .value("Illegal", sys_signal_t::Illegal)
        .value("Trap", sys_signal_t::Trap)
        .value("Abort", sys_signal_t::Abort)
        .value("Emulate", sys_signal_t::Emulate)
        .value("FPE", sys_signal_t::FPE)
        .value("Kill", sys_signal_t::Kill)
        .value("Bus", sys_signal_t::Bus)
        .value("SegFault", sys_signal_t::SegFault)
        .value("System", sys_signal_t::System)
        .value("Pipe", sys_signal_t::Pipe)
        .value("Alarm", sys_signal_t::Alarm)
        .value("Terminate", sys_signal_t::Terminate)
        .value("Urgent", sys_signal_t::Urgent)
        .value("Stop", sys_signal_t::Stop)
        .value("CPUtime", sys_signal_t::CPUtime)
        .value("FileSize", sys_signal_t::FileSize)
        .value("VirtualAlarm", sys_signal_t::VirtualAlarm)
        .value("ProfileAlarm", sys_signal_t::ProfileAlarm)
        .value("User1", sys_signal_t::User1)
        .value("User2", sys_signal_t::User2);

    //==================================================================================//
    //
    //      CUPTI submodule
    //
    //==================================================================================//
    py::module cupti = tim.def_submodule("cupti", "cupti query");

    auto get_available_cupti_events = [=](int device) {
#if defined(TIMEMORY_USE_CUPTI)
        CUdevice cu_device;
        TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
        TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device, device));
        return tim::cupti::available_events(cu_device);
#else
        tim::consume_parameters(device);
        return py::list();
#endif
    };

    auto get_available_cupti_metrics = [=](int device) {
#if defined(TIMEMORY_USE_CUPTI)
        CUdevice cu_device;
        TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
        TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device, device));
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
    opts.attr("echo_dart")          = false;
    opts.attr("ctest_notes")        = false;
    opts.attr("matplotlib_backend") = std::string("default");

    //==================================================================================//
    //
    //      Class declarations
    //
    //==================================================================================//
    py::class_<manager_wrapper> man(
        tim, "manager", "object which controls static data lifetime and finalization");

    py::class_<tim_timer_t> timer(tim, "timer",
                                  "Auto-timer that does not start/stop based on scope");

    py::class_<auto_timer_t> auto_timer(tim, "auto_timer", "Pre-configured bundle");

    py::class_<component_list_t> comp_list(tim, "component_tuple",
                                           "Generic component_tuple");

    py::class_<auto_timer_decorator> timer_decorator(
        tim, "timer_decorator", "Auto-timer type used in decorators");

    py::class_<component_list_decorator> comp_decorator(
        tim, "component_decorator", "Component list used in decorators");

    py::class_<pycomponent_bundle> comp_bundle(
        tim, "component_bundle", "Component bundle specific to Python interface");

    py::class_<rss_usage_t> rss_usage(tim, "rss_usage",
                                      "Pre-configured memory usage bundle");

    py::class_<pytim::settings> settings(tim, "settings",
                                         "Global configuration settings for timemory");

    //==================================================================================//
    //
    //      Helper lambdas
    //
    //==================================================================================//
    auto report = [&](std::string fname) {
        auto _path   = tim::settings::output_path();
        auto _prefix = tim::settings::output_prefix();

        using type_tuple = typename auto_list_t::type_tuple;
        tim::manager::get_storage<type_tuple>::print();

        if(fname.length() > 0)
        {
            tim::settings::output_path()   = _path;
            tim::settings::output_prefix() = _prefix;
        }
    };
    //----------------------------------------------------------------------------------//
    auto _as_json = [&]() {
        using type_tuple = typename auto_list_t::type_tuple;
        auto json_str    = manager_t::get_storage<type_tuple>::serialize();
        auto json_module = py::module::import("json");
        return json_module.attr("loads")(json_str);
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
    //
    //----------------------------------------------------------------------------------//
    //
    auto _start_mpip = [&]() {
#if defined(TIMEMORY_USE_MPIP_LIBRARY)
        return timemory_start_mpip();
#else
        return 0;
#endif
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto _stop_mpip = [&](uint64_t id) {
#if defined(TIMEMORY_USE_MPIP_LIBRARY)
        return timemory_stop_mpip(id);
#else
        tim::consume_parameters(id);
        return 0;
#endif
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto _init_ompt = [&]() {
#if defined(TIMEMORY_USE_OMPT_LIBRARY)
        return timemory_start_ompt();
#else
        return 0;
#endif
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto _stop_ompt = [&](uint64_t id) {
#if defined(TIMEMORY_USE_OMPT_LIBRARY)
        return timemory_stop_ompt(id);
#else
        tim::consume_parameters(id);
        return 0;
#endif
    };
    //----------------------------------------------------------------------------------//
    auto _init = [&](py::list argv, std::string _prefix, std::string _suffix) {
        if(argv.size() < 1)
            return;
        int    _argc = argv.size();
        char** _argv = new char*[argv.size()];
        for(int i = 0; i < _argc; ++i)
        {
            auto  _str    = argv[i].cast<std::string>();
            char* _argv_i = new char[_str.size()];
            std::strcpy(_argv_i, _str.c_str());
            _argv[i] = _argv_i;
        }
        tim::timemory_init(_argc, _argv, _prefix, _suffix);
        for(int i = 0; i < _argc; ++i)
            delete[] _argv[i];
        delete[] _argv;
    };
    //----------------------------------------------------------------------------------//
    auto _finalize = [&]() {
        try
        {
            if(!tim::get_env("TIMEMORY_SKIP_FINALIZE", false))
            {
                // python GC seems to cause occasional problems
                tim::settings::stack_clearing() = false;
                tim::timemory_finalize();
            }
        } catch(std::exception& e)
        {
#if defined(_UNIX)
            auto bt = tim::get_demangled_backtrace<32>();
            for(const auto& itr : bt)
            {
                std::cerr << "\nBacktrace:\n";
                if(itr.length() > 0)
                    std::cerr << itr << "\n";
                std::cerr << "\n" << std::flush;
            }
#endif
            PRINT_HERE("ERROR: %s", e.what());
        }
    };
    //----------------------------------------------------------------------------------//
    auto _init_mpi = [&]() {
        try
        {
            // tim::mpi::init();
        } catch(std::exception& e)
        {
            PRINT_HERE("ERROR: %s", e.what());
        }
    };
    //----------------------------------------------------------------------------------//
    auto _finalize_mpi = [&]() {
        try
        {
            // tim::mpi::finalize();
        } catch(std::exception& e)
        {
            PRINT_HERE("ERROR: %s", e.what());
        }
    };
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                  MAIN libpytimemory MODULE (part 1)
    //
    //==================================================================================//
    tim.def("report", report, "Print the data", py::arg("filename") = "");
    //----------------------------------------------------------------------------------//
    tim.def("toggle", [&](bool on) { tim::settings::enabled() = on; },
            "Enable/disable timemory", py::arg("on") = true);
    //----------------------------------------------------------------------------------//
    tim.def("enable", [&]() { tim::settings::enabled() = true; }, "Enable timemory");
    //----------------------------------------------------------------------------------//
    tim.def("disable", [&]() { tim::settings::enabled() = false; }, "Disable timemory");
    //----------------------------------------------------------------------------------//
    tim.def("is_enabled", [&]() { return tim::settings::enabled(); },
            "Return if timemory is enabled or disabled");
    //----------------------------------------------------------------------------------//
    tim.def("enabled", [&]() { return tim::settings::enabled(); },
            "Return if timemory is enabled or disabled");
    //----------------------------------------------------------------------------------//
    tim.def("enable_signal_detection", &pytim::enable_signal_detection,
            "Enable signal detection", py::arg("signal_list") = py::list());
    //----------------------------------------------------------------------------------//
    tim.def("disable_signal_detection", &pytim::disable_signal_detection,
            "Enable signal detection");
    //----------------------------------------------------------------------------------//
    tim.def("has_mpi_support", [&]() { return tim::mpi::is_supported(); },
            "Return if the timemory library has MPI support");
    //----------------------------------------------------------------------------------//
    tim.def("set_rusage_children", set_rusage_child,
            "Set the rusage to record child processes");
    //----------------------------------------------------------------------------------//
    tim.def("set_rusage_self", set_rusage_self,
            "Set the rusage to record child processes");
    //----------------------------------------------------------------------------------//
    tim.def("set_exit_action",
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
    tim.def("timemory_init", _init, "Initialize timemory", py::arg("argv") = py::list(),
            py::arg("prefix") = "timemory-", py::arg("suffix") = "-output");
    //----------------------------------------------------------------------------------//
    tim.def("timemory_finalize", _finalize,
            "Finalize timemory (generate output) -- important to call if using MPI");
    //----------------------------------------------------------------------------------//
    tim.def("initialize", _init, "Initialize timemory", py::arg("argv") = py::list(),
            py::arg("prefix") = "timemory-", py::arg("suffix") = "-output");
    //----------------------------------------------------------------------------------//
    tim.def("finalize", _finalize,
            "Finalize timemory (generate output) -- important to call if using MPI");
    //----------------------------------------------------------------------------------//
    tim.def("get", _as_json, "Get the storage data");
    //----------------------------------------------------------------------------------//
    tim.def(
        "init_mpip", _start_mpip,
        "Activate MPIP profiling (function name deprecated -- use start_mpip instead)");
    //----------------------------------------------------------------------------------//
    tim.def("start_mpip", _start_mpip, "Activate MPIP profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_mpip", _stop_mpip, "Deactivate MPIP profiling", py::arg("id"));
    //----------------------------------------------------------------------------------//
    tim.def("mpi_finalize", _finalize_mpi, "Finalize MPI");
    //----------------------------------------------------------------------------------//
    tim.def("mpi_init", _init_mpi, "Initialize MPI");
    //----------------------------------------------------------------------------------//
    tim.def("init_ompt", _init_ompt, "Activate OMPT (OpenMP tools) profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_ompt", _stop_ompt, "Deactivate OMPT (OpenMP tools)  profiling",
            py::arg("id"));

    //==================================================================================//
    //
    //                          SETTINGS
    //
    //==================================================================================//

#define SETTING_PROPERTY(TYPE, FUNC)                                                     \
    settings.def_property_static(TIMEMORY_STRINGIZE(FUNC),                               \
                                 [](py::object) { return tim::settings::FUNC(); },       \
                                 [](py::object, TYPE v) { tim::settings::FUNC() = v; })

    settings.def(py::init<>(), "Dummy");

    using strvector_t = std::vector<std::string>;

    SETTING_PROPERTY(bool, suppress_parsing);
    SETTING_PROPERTY(bool, enabled);
    SETTING_PROPERTY(bool, auto_output);
    SETTING_PROPERTY(bool, cout_output);
    SETTING_PROPERTY(bool, file_output);
    SETTING_PROPERTY(bool, text_output);
    SETTING_PROPERTY(bool, json_output);
    SETTING_PROPERTY(bool, dart_output);
    SETTING_PROPERTY(bool, time_output);
    SETTING_PROPERTY(bool, plot_output);
    SETTING_PROPERTY(bool, diff_output);
    SETTING_PROPERTY(bool, flamegraph_output);
    SETTING_PROPERTY(int, verbose);
    SETTING_PROPERTY(bool, debug);
    SETTING_PROPERTY(bool, banner);
    SETTING_PROPERTY(bool, flat_profile);
    SETTING_PROPERTY(bool, timeline_profile);
    SETTING_PROPERTY(bool, collapse_threads);
    SETTING_PROPERTY(bool, destructor_report);
    SETTING_PROPERTY(uint16_t, max_depth);
    SETTING_PROPERTY(string_t, time_format);
    SETTING_PROPERTY(string_t, python_exe);
    SETTING_PROPERTY(strvector_t, command_line);
    // width/precision
    SETTING_PROPERTY(int16_t, precision);
    SETTING_PROPERTY(int16_t, width);
    SETTING_PROPERTY(bool, scientific);
    SETTING_PROPERTY(int16_t, timing_precision);
    SETTING_PROPERTY(int16_t, timing_width);
    SETTING_PROPERTY(string_t, timing_units);
    SETTING_PROPERTY(bool, timing_scientific);
    SETTING_PROPERTY(int16_t, memory_precision);
    SETTING_PROPERTY(int16_t, memory_width);
    SETTING_PROPERTY(string_t, memory_units);
    SETTING_PROPERTY(bool, memory_scientific);
    // output
    SETTING_PROPERTY(string_t, output_path);
    SETTING_PROPERTY(string_t, output_prefix);
    // dart
    SETTING_PROPERTY(string_t, dart_type);
    SETTING_PROPERTY(uint64_t, dart_count);
    SETTING_PROPERTY(bool, dart_label);
    // parallelism
    SETTING_PROPERTY(bool, cpu_affinity);
    SETTING_PROPERTY(bool, mpi_init);
    SETTING_PROPERTY(bool, mpi_finalize);
    SETTING_PROPERTY(bool, mpi_thread);
    SETTING_PROPERTY(string_t, mpi_thread_type);
    SETTING_PROPERTY(bool, mpi_output_per_rank);
    SETTING_PROPERTY(bool, mpi_output_per_node);
    SETTING_PROPERTY(bool, upcxx_init);
    SETTING_PROPERTY(bool, upcxx_finalize);
    SETTING_PROPERTY(int32_t, node_count);
    // misc
    SETTING_PROPERTY(bool, stack_clearing);
    SETTING_PROPERTY(bool, add_secondary);
    SETTING_PROPERTY(tim::process::id_t, target_pid);
    // papi
    SETTING_PROPERTY(bool, papi_multiplexing);
    SETTING_PROPERTY(bool, papi_fail_on_error);
    SETTING_PROPERTY(bool, papi_quiet);
    SETTING_PROPERTY(string_t, papi_events);
    SETTING_PROPERTY(bool, papi_attach);
    SETTING_PROPERTY(int, papi_overflow);
    // cuda/nvtx/cupti
    SETTING_PROPERTY(uint64_t, cuda_event_batch_size);
    SETTING_PROPERTY(bool, nvtx_marker_device_sync);
    SETTING_PROPERTY(int32_t, cupti_activity_level);
    SETTING_PROPERTY(string_t, cupti_activity_kinds);
    SETTING_PROPERTY(string_t, cupti_events);
    SETTING_PROPERTY(string_t, cupti_metrics);
    SETTING_PROPERTY(int, cupti_device);
    // roofline
    SETTING_PROPERTY(string_t, roofline_mode);
    SETTING_PROPERTY(string_t, cpu_roofline_mode);
    SETTING_PROPERTY(string_t, gpu_roofline_mode);
    SETTING_PROPERTY(string_t, cpu_roofline_events);
    SETTING_PROPERTY(string_t, gpu_roofline_events);
    SETTING_PROPERTY(bool, roofline_type_labels);
    SETTING_PROPERTY(bool, roofline_type_labels_cpu);
    SETTING_PROPERTY(bool, roofline_type_labels_gpu);
    SETTING_PROPERTY(bool, instruction_roofline);
    // ert
    SETTING_PROPERTY(uint64_t, ert_num_threads);
    SETTING_PROPERTY(uint64_t, ert_num_threads_cpu);
    SETTING_PROPERTY(uint64_t, ert_num_threads_gpu);
    SETTING_PROPERTY(uint64_t, ert_num_streams);
    SETTING_PROPERTY(uint64_t, ert_grid_size);
    SETTING_PROPERTY(uint64_t, ert_block_size);
    SETTING_PROPERTY(uint64_t, ert_alignment);
    SETTING_PROPERTY(uint64_t, ert_min_working_size);
    SETTING_PROPERTY(uint64_t, ert_min_working_size_cpu);
    SETTING_PROPERTY(uint64_t, ert_min_working_size_gpu);
    SETTING_PROPERTY(uint64_t, ert_max_data_size);
    SETTING_PROPERTY(uint64_t, ert_max_data_size_cpu);
    SETTING_PROPERTY(uint64_t, ert_max_data_size_gpu);
    SETTING_PROPERTY(string_t, ert_skip_ops);
    // signals
    SETTING_PROPERTY(bool, allow_signal_handler);
    SETTING_PROPERTY(bool, enable_signal_handler);
    SETTING_PROPERTY(bool, enable_all_signals);
    SETTING_PROPERTY(bool, disable_all_signals);

    //==================================================================================//
    //
    //                          TIMER
    //
    //==================================================================================//
    timer.def(py::init(&pytim::init::timer), "Initialization", py::arg("key") = "",
              py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    timer.def("real_elapsed",
              [&](py::object pytimer) {
                  tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
                  auto&        obj    = *(_timer.get<wall_clock>());
                  return obj.get();
              },
              "Elapsed wall clock");
    //----------------------------------------------------------------------------------//
    timer.def("sys_elapsed",
              [&](py::object pytimer) {
                  tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
                  auto&        obj    = *(_timer.get<system_clock>());
                  return obj.get();
              },
              "Elapsed system clock");
    //----------------------------------------------------------------------------------//
    timer.def("user_elapsed",
              [&](py::object pytimer) {
                  tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
                  auto&        obj    = *(_timer.get<user_clock>());
                  return obj.get();
              },
              "Elapsed user time");
    //----------------------------------------------------------------------------------//
    timer.def("start", [&](py::object pytimer) { pytimer.cast<tim_timer_t*>()->start(); },
              "Start timer");
    //----------------------------------------------------------------------------------//
    timer.def("stop", [&](py::object pytimer) { pytimer.cast<tim_timer_t*>()->stop(); },
              "Stop timer");
    //----------------------------------------------------------------------------------//
    timer.def("report",
              [&](py::object pytimer) {
                  std::cout << *(pytimer.cast<tim_timer_t*>()) << std::endl;
              },
              "Report timer");
    //----------------------------------------------------------------------------------//
    timer.def("__str__",
              [&](py::object pytimer) {
                  std::stringstream ss;
                  ss << *(pytimer.cast<tim_timer_t*>());
                  return ss.str();
              },
              "Stringify timer");
    //----------------------------------------------------------------------------------//
    timer.def("reset", [&](py::object self) { self.cast<tim_timer_t*>()->reset(); },
              "Reset the timer");
    //----------------------------------------------------------------------------------//
    timer.def("get_raw",
              [&](py::object self) { return (*self.cast<tim_timer_t*>()).get(); },
              "Get the timer data");
    //----------------------------------------------------------------------------------//
    timer.def("get",
              [&](py::object self) {
                  auto&& _tup           = (*self.cast<tim_timer_t*>()).get_labeled();
                  using data_label_type = tim::decay_t<decltype(_tup)>;
                  return pytim::dict<data_label_type>::construct(_tup);
              },
              "Get the timer data");
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
                   py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    auto_timer.def("__str__",
                   [&](py::object self) {
                       std::stringstream _ss;
                       auto_timer_t*     _self = self.cast<auto_timer_t*>();
                       _ss << *_self;
                       return _ss.str();
                   },
                   "Print the auto timer");
    //----------------------------------------------------------------------------------//
    auto_timer.def("get_raw",
                   [&](py::object self) { return (*self.cast<auto_timer_t*>()).get(); },
                   "Get the component list data");
    //----------------------------------------------------------------------------------//
    auto_timer.def("get",
                   [&](py::object self) {
                       auto&& _tup = (*self.cast<auto_timer_t*>()).get_labeled();
                       using data_label_type = tim::decay_t<decltype(_tup)>;
                       return pytim::dict<data_label_type>::construct(_tup);
                   },
                   "Get the component list data");
    //----------------------------------------------------------------------------------//
    timer_decorator.def(py::init(&pytim::init::timer_decorator), "Initialization",
                        py::return_value_policy::automatic);
    //----------------------------------------------------------------------------------//

    auto configure_pybundle = [](py::list _args, bool flat_profile,
                                 bool timeline_profile) {
        std::set<TIMEMORY_COMPONENT> components;
        if(_args.empty())
            components.insert(WALL_CLOCK);

        for(auto itr : _args)
        {
            std::string        _sitr = "";
            TIMEMORY_COMPONENT _citr = TIMEMORY_COMPONENTS_END;

            try
            {
                _sitr = itr.cast<std::string>();
                if(_sitr.length() > 0)
                    _citr = tim::runtime::enumerate(_sitr);
                else
                    continue;
            } catch(...)
            {}

            if(_citr == TIMEMORY_COMPONENTS_END)
            {
                try
                {
                    _citr = itr.cast<TIMEMORY_COMPONENT>();
                } catch(...)
                {}
            }

            if(_citr != TIMEMORY_COMPONENTS_END)
                components.insert(_citr);
            else
            {
                PRINT_HERE("%s", "ignoring argument that failed casting to either "
                                 "'timemory.component' and string");
            }
        }

        size_t isize = pycomponent_bundle::size();
        if(tim::settings::debug() || tim::settings::verbose() > 3)
        {
            PRINT_HERE("%s", "configuring pybundle");
        }

        using bundle_type = typename pycomponent_bundle::type;
        tim::configure<bundle_type>(components,
                                    tim::scope::config{ flat_profile, timeline_profile });

        if(tim::settings::debug() || tim::settings::verbose() > 3)
        {
            size_t fsize = pycomponent_bundle::size();
            if((fsize - isize) < components.size())
            {
                std::stringstream ss;
                ss << "Warning: final size " << fsize << ", input size " << isize
                   << ". Difference is less than the components size: "
                   << components.size();
                PRINT_HERE("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
            PRINT_HERE("final size: %lu, input size: %lu, components size: %lu\n",
                       (unsigned long) fsize, (unsigned long) isize,
                       (unsigned long) components.size());
        }
    };

    pybundle_t::global_init(nullptr);

    //==================================================================================//
    //
    //                      Component bundle
    //
    //==================================================================================//
    comp_bundle.def(py::init(&pytim::init::component_bundle), "Initialization",
                    py::arg("func"), py::arg("file"), py::arg("line"),
                    py::arg("extra") = py::list(),
                    py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    comp_bundle.def("start", &pycomponent_bundle::start, "Start the bundle");
    //----------------------------------------------------------------------------------//
    comp_bundle.def("stop", &pycomponent_bundle::stop, "Stop the bundle");
    //----------------------------------------------------------------------------------//
    comp_bundle.def_static(
        "configure", configure_pybundle, py::arg("components") = py::list(),
        py::arg("flat_profile") = false, py::arg("timeline_profile") = false,
        "Configure the profiler types (default: 'wall_clock')");
    //----------------------------------------------------------------------------------//
    comp_bundle.def_static("reset", &pycomponent_bundle::reset,
                           "Reset the components in the bundle");

    //==================================================================================//
    //
    //                      TIMEMORY COMPONENT_TUPLE
    //
    //==================================================================================//
    comp_list.def(py::init(&pytim::init::component_list), "Initialization",
                  py::arg("components") = py::list(), py::arg("key") = "",
                  py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    comp_list.def("start",
                  [&](py::object self) { self.cast<component_list_t*>()->start(); },
                  "Start component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def("stop",
                  [&](py::object self) { self.cast<component_list_t*>()->stop(); },
                  "Stop component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def("report",
                  [&](py::object self) {
                      std::cout << *(self.cast<component_list_t*>()) << std::endl;
                  },
                  "Report component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def("__str__",
                  [&](py::object self) {
                      std::stringstream ss;
                      ss << *(self.cast<component_list_t*>());
                      return ss.str();
                  },
                  "Stringify component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def("reset",
                  [&](py::object self) { self.cast<component_list_t*>()->reset(); },
                  "Reset the component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def("__str__",
                  [&](py::object self) {
                      std::stringstream _ss;
                      component_list_t* _self = self.cast<component_list_t*>();
                      _ss << *_self;
                      return _ss.str();
                  },
                  "Print the component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "get_raw",
        [&](py::object self) { return (*self.cast<component_list_t*>()).get(); },
        "Get the component list data");
    //----------------------------------------------------------------------------------//
    comp_list.def("get",
                  [&](py::object self) {
                      auto&& _tup = (*self.cast<component_list_t*>()).get_labeled();
                      using data_label_type = tim::decay_t<decltype(_tup)>;
                      return pytim::dict<data_label_type>::construct(_tup);
                  },
                  "Get the component list data");
    //----------------------------------------------------------------------------------//
    comp_decorator.def(py::init(&pytim::init::component_decorator), "Initialization",
                       py::return_value_policy::automatic);
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                      AUTO TIMER DECORATOR
    //
    //==================================================================================//
    /*
    decorate_auto_timer.def(
        py::init(&pytim::decorators::init::auto_timer), "Initialization",
        py::arg("key") = "", py::arg("line") = pytim::get_line(1),
        py::arg("report_at_exit") = false, py::return_value_policy::take_ownership);
    decorate_auto_timer.def("__call__", &pytim::decorators::auto_timer::call,
                            "Call operator");
                            */

    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                      RSS USAGE
    //
    //==================================================================================//
    rss_usage.def(py::init(&pytim::init::rss_usage),
                  "Initialization of RSS measurement class", py::arg("key") = "",
                  py::arg("record") = false, py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    rss_usage.def("record", [&](py::object self) { self.cast<rss_usage_t*>()->record(); },
                  "Record the RSS usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def("__str__",
                  [&](py::object self) {
                      std::stringstream ss;
                      ss << *(self.cast<rss_usage_t*>());
                      return ss.str();
                  },
                  "Stringify the rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def("__iadd__",
                  [&](py::object self, py::object rhs) {
                      *(self.cast<rss_usage_t*>()) += *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Add rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def("__isub__",
                  [&](py::object self, py::object rhs) {
                      *(self.cast<rss_usage_t*>()) -= *(rhs.cast<rss_usage_t*>());
                      return self;
                  },
                  "Subtract rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def("__add__",
                  [&](py::object self, py::object rhs) {
                      rss_usage_t* _rss = new rss_usage_t(*(self.cast<rss_usage_t*>()));
                      *_rss += *(rhs.cast<rss_usage_t*>());
                      return _rss;
                  },
                  "Add rss usage", py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    rss_usage.def("__sub__",
                  [&](py::object self, py::object rhs) {
                      rss_usage_t* _rss = new rss_usage_t(*(self.cast<rss_usage_t*>()));
                      *_rss -= *(rhs.cast<rss_usage_t*>());
                      return _rss;
                  },
                  "Subtract rss usage", py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    rss_usage.def("current",
                  [&](py::object self, int64_t /*_units*/) {
                      return std::get<0>(*self.cast<rss_usage_t*>()).get_display();
                  },
                  "Return the current rss usage",
                  py::arg("units") = units.attr("megabyte"));
    //----------------------------------------------------------------------------------//
    rss_usage.def("peak",
                  [&](py::object self, int64_t /*_units*/) {
                      return std::get<1>(*self.cast<rss_usage_t*>()).get_display();
                  },
                  "Return the current rss usage",
                  py::arg("units") = units.attr("megabyte"));
    //----------------------------------------------------------------------------------//
    rss_usage.def("get_raw",
                  [&](py::object self) { return (*self.cast<rss_usage_t*>()).get(); },
                  "Return the rss usage data");
    //----------------------------------------------------------------------------------//
    rss_usage.def("get",
                  [&](py::object self) {
                      auto&& _tup           = (*self.cast<rss_usage_t*>()).get_labeled();
                      using data_label_type = tim::decay_t<decltype(_tup)>;
                      return pytim::dict<data_label_type>::construct(_tup);
                  },
                  "Return the rss usage data");

    //==================================================================================//
    //
    //      Options submodule
    //
    //==================================================================================//

    // ---------------------------------------------------------------------- //
    opts.def("safe_mkdir", &pytim::opt::safe_mkdir,
             "if [ ! -d <directory> ]; then mkdir -p <directory> ; fi");
    // ---------------------------------------------------------------------- //
    opts.def("ensure_directory_exists", &pytim::opt::ensure_directory_exists,
             "mkdir -p $(basename file_path)");
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
             "timemory args and replaces sys.argv with the unknown args (used to "
             "fix issue with unittest module)",
             py::arg("parser") = py::none(), py::arg("fpath") = ".");
    // ---------------------------------------------------------------------- //
}
