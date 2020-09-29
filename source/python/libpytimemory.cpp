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
#include "timemory/components.hpp"
#include "timemory/components/extern.hpp"
#include "timemory/components/ompt.hpp"
#include "timemory/enum.h"
#include "timemory/library.h"
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

#if defined(TIMEMORY_USE_NCCLP_LIBRARY)
extern "C"
{
    extern uint64_t timemory_start_ncclp();
    extern uint64_t timemory_stop_ncclp(uint64_t);
}
#endif

//======================================================================================//

template <size_t Idx>
using enumerator_t = typename tim::component::enumerator<Idx>::type;

template <size_t Idx>
using enumerator_vt =
    tim::conditional_t<tim::trait::is_available<enumerator_t<Idx>>::value, std::true_type,
                       std::false_type>;

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

//--------------------------------------------------------------------------------------//
namespace impl
{
template <typename Tp, typename Archive,
          tim::enable_if_t<tim::trait::is_available<Tp>::value> = 0>
auto
get_json(Archive& ar, int) -> decltype(tim::storage<Tp>::instance()->dmp_get(ar), void())
{
    tim::storage<Tp>::instance()->dmp_get(ar);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Archive>
auto
get_json(Archive&, long)
{}
}  // namespace impl
//--------------------------------------------------------------------------------------//

template <typename Tp, typename Archive,
          tim::enable_if_t<tim::trait::is_available<Tp>::value> = 0>
auto
get_json(Archive& ar, int)
{
    impl::get_json<Tp>(ar, 0);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Archive,
          tim::enable_if_t<!tim::trait::is_available<Tp>::value> = 0>
auto
get_json(Archive&, ...)
{}

//--------------------------------------------------------------------------------------//

template <typename Archive, size_t... Idx>
auto
get_json(Archive& ar, std::index_sequence<Idx...>)
{
    TIMEMORY_FOLD_EXPRESSION(get_json<tim::decay_t<enumerator_t<Idx>>>(ar, 0));
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
    try
    {
        py::add_ostream_redirect(tim, "ostream_redirect");
    } catch(std::exception&)
    {}

    //==================================================================================//
    //
    //      Submodules and enumerations built in another compilation unit
    //
    //==================================================================================//

    pyapi::generate(tim);
    pysignals::generate(tim);
    pysettings::generate(tim);
    pyauto_timer::generate(tim);
    pycomponent_list::generate(tim);
    pycomponent_bundle::generate(tim);
    pyhardware_counters::generate(tim);
    auto pyunit = pyunits::generate(tim);
    auto pycomp = pycomponents::generate(tim);
    pyrss_usage::generate(tim, pyunit);
    pyenumeration::generate(pycomp);
    pyprofile::generate(tim);

    //==================================================================================//
    //
    //      Tracing submodule
    //
    //==================================================================================//

    py::module _trace = tim.def_submodule(
        "trace", "C/C++/Fortran-compatible library functions (subject to throttling)");

    _trace.def("init", &timemory_trace_init, "Initialize Tracing",
               py::arg("args") = "wall_clock", py::arg("read_command_line") = false,
               py::arg("cmd") = "");
    _trace.def("finalize", &timemory_trace_finalize, "Finalize Tracing");
    _trace.def("is_throttled", &timemory_is_throttled, "Check if key is throttled",
               py::arg("key"));
    _trace.def("push", &timemory_push_trace, "Push Trace", py::arg("key"));
    _trace.def("pop", &timemory_pop_trace, "Pop Trace", py::arg("key"));

    //==================================================================================//
    //
    //      Region submodule
    //
    //==================================================================================//

    py::module _region = tim.def_submodule(
        "region",
        "C/C++/Fortran-compatible library functions (not subject to throttling)");

    //----------------------------------------------------------------------------------//
    auto _set_default = [](py::list types) {
        std::stringstream ss;
        for(auto itr : types)
            ss << "," << itr.cast<std::string>();
        timemory_set_default(ss.str().substr(1).c_str());
    };
    //----------------------------------------------------------------------------------//
    auto _add_components = [](py::list types) {
        std::stringstream ss;
        for(auto itr : types)
            ss << "," << itr.cast<std::string>();
        timemory_add_components(ss.str().substr(1).c_str());
    };
    //----------------------------------------------------------------------------------//
    auto _remove_components = [](py::list types) {
        std::stringstream ss;
        for(auto itr : types)
            ss << "," << itr.cast<std::string>();
        timemory_remove_components(ss.str().substr(1).c_str());
    };
    //----------------------------------------------------------------------------------//
    auto _push_components = [](py::list types) {
        std::stringstream ss;
        for(auto itr : types)
            ss << "," << itr.cast<std::string>();
        timemory_push_components(ss.str().substr(1).c_str());
    };
    //----------------------------------------------------------------------------------//

    _region.def("push", &timemory_push_region, "Push Trace Region", py::arg("key"));
    _region.def("pop", &timemory_pop_region, "Pop Trace Region", py::arg("key"));
    _region.def("pause", &timemory_pause, "Pause data collection");
    _region.def("resume", &timemory_resume, "Resume data collection");
    _region.def("set_default", _set_default, "Set the default list of components");
    _region.def("add_components", _add_components,
                "Add these components to the current collection");
    _region.def("remove_components", _remove_components,
                "Remove these components from the current collection");
    _region.def("push_components", _push_components,
                "Set the current components to collect");
    _region.def("pop_components", &timemory_pop_components,
                "Pop the current set of components");
    _region.def("begin", &timemory_get_begin_record,
                "Begin recording and get an identifier", py::arg("key"));
    _region.def("end", &timemory_end_record, "End recording an identifier",
                py::arg("id"));

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
    auto report = [](std::string fname) {
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
    auto _as_json_classic = []() {
        using tuple_type = typename auto_list_t::tuple_type;
        auto json_str    = manager_t::get_storage<tuple_type>::serialize();
        if(tim::settings::debug())
            std::cout << "JSON CLASSIC:\n" << json_str << std::endl;
        return json_str;
    };
    //----------------------------------------------------------------------------------//
    auto _as_json_hierarchy = []() {
        std::stringstream ss;
        {
            using policy_type = tim::policy::output_archive_t<tim::manager>;
            auto oa           = policy_type::get(ss);
            oa->setNextName("timemory");
            oa->startNode();
            get_json(*oa, std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
            oa->finishNode();
        }
        if(tim::settings::debug())
            std::cout << "JSON HIERARCHY:\n" << ss.str() << std::endl;
        return ss.str();
    };
    //----------------------------------------------------------------------------------//
    auto _as_json = [_as_json_classic, _as_json_hierarchy](bool hierarchy) {
        auto json_str    = (hierarchy) ? _as_json_hierarchy() : _as_json_classic();
        auto json_module = py::module::import("json");
        return json_module.attr("loads")(json_str);
    };
    //----------------------------------------------------------------------------------//
    auto set_rusage_child = []() {
#if !defined(_WINDOWS)
        tim::get_rusage_type() = RUSAGE_CHILDREN;
#endif
    };
    //----------------------------------------------------------------------------------//
    auto set_rusage_self = []() {
#if !defined(_WINDOWS)
        tim::get_rusage_type() = RUSAGE_SELF;
#endif
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto _start_mpip = []() {
#if defined(TIMEMORY_USE_MPIP_LIBRARY)
        return timemory_start_mpip();
#else
        return 0;
#endif
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto _stop_mpip = [](uint64_t id) {
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
    auto _start_ompt = []() {
#if defined(TIMEMORY_USE_OMPT_LIBRARY)
        return timemory_start_ompt();
#else
        return 0;
#endif
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto _stop_ompt = [](uint64_t id) {
#if defined(TIMEMORY_USE_OMPT_LIBRARY)
        return timemory_stop_ompt(id);
#else
        tim::consume_parameters(id);
        return 0;
#endif
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto _start_ncclp = []() {
#if defined(TIMEMORY_USE_NCCLP_LIBRARY)
        return timemory_start_ncclp();
#else
        return 0;
#endif
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto _stop_ncclp = [](uint64_t id) {
#if defined(TIMEMORY_USE_NCCLP_LIBRARY)
        return timemory_stop_ncclp(id);
#else
        tim::consume_parameters(id);
        return 0;
#endif
    };
    //----------------------------------------------------------------------------------//
    auto _get_argv = []() {
        py::module sys   = py::module::import("sys");
        auto       argv  = sys.attr("argv").cast<py::list>();
        int        _argc = argv.size();
        char**     _argv = new char*[argv.size()];
        for(int i = 0; i < _argc; ++i)
        {
            auto  _str    = argv[i].cast<std::string>();
            char* _argv_i = new char[_str.size()];
            std::strcpy(_argv_i, _str.c_str());
            _argv[i] = _argv_i;
        }
        auto _argv_deleter = [](int fargc, char** fargv) {
            for(int i = 0; i < fargc; ++i)
                delete[] fargv[i];
            delete[] fargv;
        };
        return std::make_tuple(_argc, _argv, _argv_deleter);
    };
    //----------------------------------------------------------------------------------//
    auto _init = [](py::list argv, std::string _prefix, std::string _suffix) {
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
    auto _finalize = []() {
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
            auto             bt    = tim::get_demangled_backtrace<32>();
            std::set<size_t> valid = {};
            size_t           idx   = 0;
            for(const auto& itr : bt)
            {
                if(itr.length() > 0)
                    valid.insert(idx);
                ++idx;
            }
            if(!valid.empty())
            {
                std::cerr << "\nBacktrace:\n";
                for(auto itr : valid)
                    std::cerr << "[" << std::setw(2) << itr << " / " << std::setw(2)
                              << valid.size() << "] " << bt.at(itr) << '\n';
            }
            std::cerr << "\n" << std::flush;
#endif
            PRINT_HERE("ERROR: %s", e.what());
        }
    };
    //----------------------------------------------------------------------------------//
    auto _init_mpi = [_get_argv]() {
        try
        {
            auto                             _args = _get_argv();
            int                              _argc = std::get<0>(_args);
            char**                           _argv = std::get<1>(_args);
            std::function<void(int, char**)> _argd = std::get<2>(_args);
            // initialize mpi
            tim::mpi::initialize(&_argc, &_argv);
            // delete the c-arrays
            _argd(_argc, _argv);
        } catch(std::exception& e)
        {
            PRINT_HERE("ERROR: %s", e.what());
        }
    };
    //----------------------------------------------------------------------------------//
    auto _finalize_mpi = []() {
        try
        {
            tim::mpi::finalize();
        } catch(std::exception& e)
        {
            PRINT_HERE("ERROR: %s", e.what());
        }
    };
    //----------------------------------------------------------------------------------//
    auto _init_upcxx = []() {
        try
        {
            tim::upc::initialize();
        } catch(std::exception& e)
        {
            PRINT_HERE("ERROR: %s", e.what());
        }
    };
    //----------------------------------------------------------------------------------//
    auto _finalize_upcxx = []() {
        try
        {
            tim::mpi::finalize();
        } catch(std::exception& e)
        {
            PRINT_HERE("ERROR: %s", e.what());
        }
    };
    //----------------------------------------------------------------------------------//
    auto _init_dmp = [_init_mpi, _init_upcxx]() {
        _init_mpi();
        _init_upcxx();
    };
    //----------------------------------------------------------------------------------//
    auto _finalize_dmp = [_finalize_mpi, _finalize_upcxx]() {
        _finalize_mpi();
        _finalize_upcxx();
    };
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //                  MAIN libpytimemory MODULE (part 1)
    //
    //==================================================================================//
    tim.def("report", report, "Print the data", py::arg("filename") = "");
    //----------------------------------------------------------------------------------//
    tim.def("toggle", [](bool on) { tim::settings::enabled() = on; },
            "Enable/disable timemory", py::arg("on") = true);
    //----------------------------------------------------------------------------------//
    tim.def("enable", []() { tim::settings::enabled() = true; }, "Enable timemory");
    //----------------------------------------------------------------------------------//
    tim.def("disable", []() { tim::settings::enabled() = false; }, "Disable timemory");
    //----------------------------------------------------------------------------------//
    tim.def("is_enabled", []() { return tim::settings::enabled(); },
            "Return if timemory is enabled or disabled");
    //----------------------------------------------------------------------------------//
    tim.def("enabled", []() { return tim::settings::enabled(); },
            "Return if timemory is enabled or disabled");
    //----------------------------------------------------------------------------------//
    tim.def("has_mpi_support", []() { return tim::mpi::is_supported(); },
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
    tim.def("get", _as_json, "Get the storage data", py::arg("hierarchy") = false);
    //----------------------------------------------------------------------------------//
    tim.def(
        "init_mpip", _start_mpip,
        "Activate MPIP profiling (function name deprecated -- use start_mpip instead)");
    //----------------------------------------------------------------------------------//
    tim.def(
        "init_ompt", _start_ompt,
        "Activate OMPT profiling (function name deprecated -- use start_ompt instead)");
    //----------------------------------------------------------------------------------//
    tim.def("start_mpip", _start_mpip, "Activate MPIP profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_mpip", _stop_mpip, "Deactivate MPIP profiling", py::arg("id"));
    //----------------------------------------------------------------------------------//
    tim.def("mpi_init", _init_mpi, "Initialize MPI");
    //----------------------------------------------------------------------------------//
    tim.def("mpi_finalize", _finalize_mpi, "Finalize MPI");
    //----------------------------------------------------------------------------------//
    tim.def("upcxx_init", _init_upcxx, "Initialize UPC++");
    //----------------------------------------------------------------------------------//
    tim.def("upcxx_finalize", _finalize_upcxx, "Finalize UPC++");
    //----------------------------------------------------------------------------------//
    tim.def("dmp_init", _init_dmp, "Initialize MPI and/or UPC++");
    //----------------------------------------------------------------------------------//
    tim.def("dmp_finalize", _finalize_dmp, "Finalize MPI and/or UPC++");
    //----------------------------------------------------------------------------------//
    tim.def("start_ompt", _start_ompt, "Activate OMPT (OpenMP tools) profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_ompt", _stop_ompt, "Deactivate OMPT (OpenMP tools)  profiling",
            py::arg("id"));
    //----------------------------------------------------------------------------------//
    tim.def("start_ncclp", _start_ncclp, "Activate NCCL profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_ncclp", _stop_ncclp, "Deactivate NCCL profiling", py::arg("id"));

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
