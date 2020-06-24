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

#include "libpytimemory.hpp"
<<<<<<< HEAD
#include "timemory/library.h"
#include "timemory/components/definition.hpp"
=======
#include "libpytimemory-components.hpp"
#include "timemory/components.hpp"
#include "timemory/components/extern.hpp"
>>>>>>> aee44b25a8e4f3c583469d63e17977cedde81fd6
#include "timemory/components/ompt.hpp"
#include "timemory/settings/extern.hpp"

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
<<<<<<< HEAD

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

struct pycomponents
{
    template <size_t Idx>
    static void generate(py::module& _pycomp)
    {
        using namespace tim;
        using T = typename tim::component::enumerator<Idx>::type;
        if(std::is_same<T, tim::component::placeholder<tim::component::nothing>>::value)
            return;
        using property_t = tim::component::properties<T>;
        std::string id   = property_t::enum_string();
        for(auto& itr : id)
            itr = tolower(itr);

        // define a component binding in the sub-module
        auto comp = py::class_<lightweight_tuple<T>>(_pycomp, id.c_str(), T::description().c_str());

        // define constructor
        comp.def(py::init([](std::string _name) { return new lightweight_tuple<T>(_name); }));

        // bind push
        auto _push = [](lightweight_tuple<T> *_tuple){ _tuple->push(); };
        comp.def("push", _push);

        // bind pop
        auto _pop = [](lightweight_tuple<T> *_tuple){ _tuple->pop(); };
        comp.def("pop", _pop);

        // bind start 
        auto _start = [](lightweight_tuple<T> *_tuple){ _tuple->start(); };
        comp.def("start", _start);

        // bind stop
        auto _stop = [](lightweight_tuple<T> *_tuple){ _tuple->stop(); };
        comp.def("stop", _stop);

        // bind reset
        auto _reset = [](lightweight_tuple<T> *_tuple){ _tuple->reset(); };
        comp.def("reset", _reset);

        // bind get
        auto _get = [](lightweight_tuple<T> *_tuple){ return _tuple->get(); };
        comp.def("get", _get);

        // bind get_labeled
        auto _get_labeled = [](lightweight_tuple<T> *_tuple){ return _tuple->get_labeled(); };
        comp.def("get_labeled", _get_labeled);
    }

    template <size_t... Idx>
    static void components(py::module& _pycomp,
                           std::index_sequence<Idx...>)
    {
        TIMEMORY_FOLD_EXPRESSION(pycomponents::generate<Idx>(_pycomp));
    }
};

//======================================================================================//
=======
>>>>>>> aee44b25a8e4f3c583469d63e17977cedde81fd6
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
    //      Submodules and enumerations built in another compilation unit
    //
    //==================================================================================//

<<<<<<< HEAD
    py::enum_<TIMEMORY_NATIVE_COMPONENT> components_enum(
        tim, "component", py::arithmetic(), "Components for timemory module");

    py::module components = tim.def_submodule("components", "Individual components for timemory module");

    //----------------------------------------------------------------------------------//

    pyenumeration::components(components_enum,
                              std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});

    pycomponents::components(components,
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
=======
    auto units      = pyunits::generate(tim);
    auto components = pycomponents::generate(tim);

    pyapi::generate(tim);
    pysignals::generate(tim);
    pysettings::generate(tim);
    pyauto_timer::generate(tim);
    pycomponent_list::generate(tim);
    pycomponent_bundle::generate(tim);
    pyhardware_counters::generate(tim);
    pyenumeration::generate(components);
    pyrss_usage::generate(tim, units);
>>>>>>> aee44b25a8e4f3c583469d63e17977cedde81fd6

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

    py::class_<tim::scope::destructor> destructor(
        tim, "scope_destructor",
        "An object that executes an operation when it is destroyed");

    destructor.def(py::init([]() { return new tim::scope::destructor([]() {}); }),
                   "Destructor", py::return_value_policy::move);
    destructor.def(py::init([](py::function pyfunc) {
                       return new tim::scope::destructor([=]() { pyfunc(); });
                   }),
                   "Destructor", py::return_value_policy::take_ownership);

    destructor.def_static(
        "test",
        []() {
            return tim::scope::destructor([]() { puts("I am a scoped destructor"); });
        },
        "Tests whether C++ returning a non-pointer works. If message is displayed at "
        "assignment, this is incorrect");

    //==================================================================================//
    //
    //      Helper lambdas
    //
    //==================================================================================//
    auto report = [&](std::string fname) {
        auto _path   = tim::settings::output_path();
        auto _prefix = tim::settings::output_prefix();

        using tuple_type = typename auto_list_t::tuple_type;
        tim::manager::get_storage<tuple_type>::print();

        if(fname.length() > 0)
        {
            tim::settings::output_path()   = _path;
            tim::settings::output_prefix() = _prefix;
        }
    };
    //----------------------------------------------------------------------------------//
    auto _as_json = [&]() {
        using tuple_type = typename auto_list_t::tuple_type;
        auto json_str    = manager_t::get_storage<tuple_type>::serialize();
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
    auto _init_trace = [&](const char* args, bool read_command_line, const char* cmd) {
            auto  _str    = std::string(args);
            char* _args = new char[_str.size()];
            std::strcpy(_args, _str.c_str());

            _str    = std::string(cmd);
            char* _cmd = new char[_str.size()];
            std::strcpy(_cmd, _str.c_str());

        timemory_trace_init(_args, read_command_line, _cmd);
    };
    //----------------------------------------------------------------------------------//
    auto _finalize_trace = [&](){
        timemory_trace_finalize();
    };
    //----------------------------------------------------------------------------------//
    auto _push_trace = [&](const char *name) {
        timemory_push_trace(name);
    };
    //----------------------------------------------------------------------------------//
    auto _pop_trace = [&](const char *name) {
        timemory_pop_trace(name);
    };
    //----------------------------------------------------------------------------------//
    auto _push_region = [&](const char *name) {
        timemory_push_region(name);
    };
    //----------------------------------------------------------------------------------//
    auto _pop_region = [&](const char *name) {
        timemory_pop_region(name);
    };
    //----------------------------------------------------------------------------------//
    auto _is_throttled = [&](const char *name) {
        return timemory_is_throttled(name);
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
    tim.def("has_mpi_support", [&]() { return tim::mpi::is_supported(); },
            "Return if the timemory library has MPI support");
    //----------------------------------------------------------------------------------//
    tim.def("set_rusage_children", set_rusage_child,
            "Set the rusage to record child processes");
    //----------------------------------------------------------------------------------//
    tim.def("set_rusage_self", set_rusage_self,
            "Set the rusage to record child processes");
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
    tim.def("timemory_trace_init", _init_trace, "Initialize Tracing", 
            py::arg("args") = "wall_clock", py::arg("read_command_line") = false, 
            py::arg("cmd") = "");
    //----------------------------------------------------------------------------------//
    tim.def("timemory_trace_finalize", _finalize_trace, "Finalize Tracing");
    //----------------------------------------------------------------------------------//
    tim.def("timemory_push_trace", _push_trace, "Push Trace", py::arg("name"));
    //----------------------------------------------------------------------------------//
    tim.def("timemory_pop_trace", _pop_trace, "Pop Trace", py::arg("name"));
    //----------------------------------------------------------------------------------//
    tim.def("timemory_push_region", _push_region, "Push Trace Region", 
            py::arg("name"));
    //----------------------------------------------------------------------------------//
    tim.def("timemory_pop_region", _pop_region, "Pop Trace Region",
            py::arg("name"));
    //----------------------------------------------------------------------------------//
    tim.def("timemory_is_throttled", _is_throttled, "Check if throttled", 
            py::arg("name"));
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
<<<<<<< HEAD
    //                          SETTINGS
    //
    //==================================================================================//

#define SETTING_PROPERTY(TYPE, FUNC)                                                     \
    settings.def_property_static(TIMEMORY_STRINGIZE(FUNC),                               \
                                 [](py::object) { return tim::settings::FUNC(); },       \
                                 [](py::object, TYPE v) { tim::settings::FUNC() = v; })

    settings.def(py::init<>(), "Dummy");

    // to parse changes in env vars
    settings.def("parse", &tim::settings::parse);

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
    SETTING_PROPERTY(bool, collapse_processes);
    SETTING_PROPERTY(bool, destructor_report);
    SETTING_PROPERTY(uint16_t, max_depth);
    SETTING_PROPERTY(string_t, time_format);
    SETTING_PROPERTY(string_t, python_exe);
    SETTING_PROPERTY(strvector_t, command_line);
    SETTING_PROPERTY(size_t, throttle_count);
    SETTING_PROPERTY(size_t, throttle_value);
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
    SETTING_PROPERTY(size_t, max_thread_bookmarks);
    SETTING_PROPERTY(bool, cpu_affinity);
    SETTING_PROPERTY(bool, mpi_init);
    SETTING_PROPERTY(bool, mpi_finalize);
    SETTING_PROPERTY(bool, mpi_thread);
    SETTING_PROPERTY(string_t, mpi_thread_type);
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
=======
>>>>>>> aee44b25a8e4f3c583469d63e17977cedde81fd6
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
