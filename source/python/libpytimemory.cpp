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
#include "libpytimemory-components.hpp"
#include "timemory/components/definition.hpp"
#include "timemory/components/ompt.hpp"

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

    auto enumeration = pyenumeration::generate(tim);
    auto components  = pycomponents::generate(tim);
    auto signals     = pysignals::generate(tim);
    auto settings    = pysettings::generate(tim);
    auto units       = pyunits::generate(tim);
    auto rss_usage   = pyrss_usage::generate(tim, units);

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
