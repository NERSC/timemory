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
#include "timemory/enum.h"
#include "timemory/library.h"
#include "timemory/timemory.hpp"
#include "timemory/tools/timemory-mallocp.h"
#include "timemory/tools/timemory-mpip.h"
#include "timemory/tools/timemory-ncclp.h"
#include "timemory/tools/timemory-ompt.h"
#include "timemory/utility/socket.hpp"

//======================================================================================//

template <size_t Idx>
using enumerator_vt =
    tim::conditional_t<tim::trait::is_available<enumerator_t<Idx>>::value, std::true_type,
                       std::false_type>;

//======================================================================================//
//
class manager_wrapper
{
public:
    manager_wrapper();
    ~manager_wrapper();
    std::shared_ptr<manager_t> get();

protected:
    std::shared_ptr<manager_t> m_manager = { nullptr };
};
//
manager_wrapper::manager_wrapper()
: m_manager(manager_t::instance())
{}
//
manager_wrapper::~manager_wrapper() {}
//
std::shared_ptr<manager_t>
manager_wrapper::get()
{
    return manager_t::instance();
}
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
//
template <typename Tp>
using basic_dmp_tree_t = std::vector<std::vector<tim::basic_tree<tim::node::tree<Tp>>>>;
//
template <typename Tp, typename Archive>
auto
get_json(Archive& ar, const pytim::pyenum_set_t& _types,
         tim::enable_if_t<tim::trait::is_available<Tp>::value, int>)
    -> decltype(
        tim::storage<Tp>::instance()->dmp_get(std::declval<basic_dmp_tree_t<Tp>&>()),
        void())
{
    if(_types.empty() || _types.count(tim::component::properties<Tp>{}()) > 0)
    {
        if(!tim::storage<Tp>::instance()->empty())
        {
            basic_dmp_tree_t<Tp> _obj{};
            tim::storage<Tp>::instance()->dmp_get(_obj);
            tim::operation::serialization<Tp>{}(ar, _obj);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Archive>
auto
get_json(Archive&, const pytim::pyenum_set_t&, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT = typename Tp::value_type, typename StreamT>
auto
get_stream(StreamT& strm, const pytim::pyenum_set_t& _types,
           tim::enable_if_t<tim::trait::is_available<Tp>::value &&
                                !tim::concepts::is_null_type<ValueT>::value,
                            int>)
    -> decltype(tim::storage<Tp>::instance()->dmp_get(), void())
{
    using printer_t = tim::operation::finalize::print<Tp, true>;
    using element_t = typename StreamT::element_type;

    if(_types.empty() || _types.count(tim::component::properties<Tp>{}()) > 0)
    {
        // strm.set_banner(Tp::get_description());
        auto _storage = tim::storage<Tp>::instance();

        if(_storage->empty())
            return;

        auto _printer = printer_t{ _storage };
        auto _data    = _printer.get_node_results();

        if(_data.empty() || _data.front().empty() || tim::dmp::rank() > 0)
            return;

        if(!strm)
            strm = std::make_shared<element_t>();

        _printer.write_stream(strm, _data);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename StreamT>
auto
get_stream(StreamT&, const pytim::pyenum_set_t&, long)
{}
//
}  // namespace impl

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Archive>
auto
get_json(Archive& ar, const pytim::pyenum_set_t& _types,
         tim::enable_if_t<tim::trait::is_available<Tp>::value, int> = 0)
{
    impl::get_json<Tp>(ar, _types, 0);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Archive>
auto
get_json(Archive&, const pytim::pyenum_set_t&,
         tim::enable_if_t<!tim::trait::is_available<Tp>::value, long> = 0)
{}

//--------------------------------------------------------------------------------------//

template <typename Archive, size_t... Idx>
auto
get_json(Archive& ar, const pytim::pyenum_set_t& _types, std::index_sequence<Idx...>)
{
    TIMEMORY_FOLD_EXPRESSION(get_json<tim::decay_t<enumerator_t<Idx>>>(ar, _types, 0));
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename StreamT>
auto
get_stream(StreamT& strm, const pytim::pyenum_set_t& _types,
           tim::enable_if_t<tim::trait::is_available<Tp>::value, int> = 0)
{
    return impl::get_stream<Tp>(strm, _types, 0);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename StreamT>
auto
get_stream(StreamT&, const pytim::pyenum_set_t&,
           tim::enable_if_t<!tim::trait::is_available<Tp>::value, long> = 0)
{}

//--------------------------------------------------------------------------------------//

template <size_t... Idx, size_t N = sizeof...(Idx)>
auto
get_stream(const pytim::pyenum_set_t& _types, std::index_sequence<Idx...>)
{
    using stream_t       = std::shared_ptr<tim::utility::stream>;
    using stream_array_t = std::array<stream_t, N>;

    auto strms = stream_array_t{};
    TIMEMORY_FOLD_EXPRESSION(
        get_stream<tim::decay_t<enumerator_t<Idx>>>(std::get<Idx>(strms), _types, 0));
    return strms;
}

//======================================================================================//
//  Python wrappers
//======================================================================================//

PYBIND11_MODULE(libpytimemory, tim)
{
    //----------------------------------------------------------------------------------//
    //
    auto _settings       = tim::settings::shared_instance();
    auto _master_manager = manager_t::master_instance();
    auto _worker_manager = manager_t::instance();
    if(_worker_manager != _master_manager)
    {
        printf("[%s]> tim::manager :: master != worker : %p vs. %p\n", __FUNCTION__,
               (void*) _master_manager.get(), (void*) _worker_manager.get());
    }
    //
    if(_settings)
    {
        if(_settings->get_debug() || _settings->get_verbose() > 3)
            PRINT_HERE("%s", "");
        //
        if(_settings->get_enable_signal_handler())
        {
            auto default_signals = tim::signal_settings::get_default();
            for(auto& itr : default_signals)
                tim::signal_settings::enable(itr);
            // should return default and any modifications from environment
            auto enabled_signals = tim::signal_settings::get_enabled();
            tim::enable_signal_detection(enabled_signals);
            // executes after the signal has been caught
            auto _exit_action = [=](int nsig) {
                if(_master_manager)
                {
                    std::cout << "Finalizing after signal: " << nsig << " :: "
                              << tim::signal_settings::str(
                                     static_cast<tim::sys_signal>(nsig))
                              << std::endl;
                    _master_manager->finalize();
                }
            };
            //
            tim::signal_settings::set_exit_action(_exit_action);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    using pytim::string_t;
    try
    {
        // py::add_ostream_redirect(tim, "ostream_redirect");
    } catch(std::exception&)
    {}

    //==================================================================================//
    //
    //      Submodules and enumerations built in another compilation unit
    //
    //==================================================================================//

    py::module _api =
        tim.def_submodule("api", "Direct python interfaces to various APIs");
    pyapi::generate(_api);
    pysignals::generate(tim);
    pyauto_timer::generate(tim);
    pycomponent_list::generate(tim);
    pycomponent_bundle::generate(tim);
    pyhardware_counters::generate(tim);
    pystatistics::generate(tim);
    pysettings::generate(tim);
    auto pyunit = pyunits::generate(tim);
    auto pycomp = pycomponents::generate(tim);
    pystorage::generate(tim);
    pyrss_usage::generate(tim, pyunit);
    pyenumeration::generate(pycomp);
    pyprofile::generate(tim);
    pytrace::generate(tim);

    //==================================================================================//
    //
    //      Scope submodule
    //
    //==================================================================================//

    py::module pyscope = tim.def_submodule(
        "scope", "Scoping controls how the values are updated in the call-graph");
    {
        namespace scope   = tim::scope;
        auto _config_init = [](py::object _flat, py::object _time) {
            auto _val = new scope::config{};
            if(!_flat.is_none())
            {
                try
                {
                    _val->set<scope::flat>(_flat.cast<bool>());
                } catch(py::cast_error&)
                {
                    auto _f = [_val](auto v) { _val->set(v); };
                    pytim::try_cast_seq<scope::flat, scope::timeline, scope::tree>(_f,
                                                                                   _flat);
                }
            }
            if(!_time.is_none())
            {
                try
                {
                    _val->set<scope::timeline>(_time.cast<bool>());
                } catch(py::cast_error&)
                {
                    auto _f = [_val](auto v) { _val->set(v); };
                    pytim::try_cast_seq<scope::flat, scope::timeline, scope::tree>(_f,
                                                                                   _time);
                }
            }
            return _val;
        };

        py::class_<scope::config>   _cfg(pyscope, "config", "Scope configuration object");
        py::class_<scope::tree>     _tree(pyscope, "_tree", "Hierarchy is maintained");
        py::class_<scope::flat>     _flat(pyscope, "_flat", "Every entry at 0th level");
        py::class_<scope::timeline> _time(pyscope, "_timeline", "Every entry is unique");

        pyscope.attr("TREE")     = new scope::tree{};
        pyscope.attr("FLAT")     = new scope::flat{};
        pyscope.attr("TIMELINE") = new scope::timeline{};

        _cfg.def(py::init(_config_init), "Create a scope", py::arg("flat") = py::none{},
                 py::arg("time") = py::none{});

        tim::consume_parameters(_tree, _flat, _time);
    }
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
    opts.attr("echo_dart")          = tim::settings::dart_output();
    opts.attr("ctest_notes")        = tim::settings::ctest_notes();
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
                       return new tim::scope::destructor([pyfunc]() { pyfunc(); });
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

        using tuple_type = tim::convert_t<tim::available_types_t, tim::type_list<>>;
        tim::manager::get_storage<tuple_type>::print();

        if(fname.length() > 0)
        {
            tim::settings::output_path()   = _path;
            tim::settings::output_prefix() = _prefix;
        }
    };
    //----------------------------------------------------------------------------------//
    auto _do_clear = [](py::list _list) {
        auto _types = pytim::get_enum_set(_list);
        if(_types.empty())
            _types = pytim::get_type_enums<tim::available_types_t>();
        using tuple_type = tim::convert_t<tim::available_types_t, tim::type_list<>>;
        manager_t::get_storage<tuple_type>::clear(_types);
    };
    //----------------------------------------------------------------------------------//
    auto _get_size = [](py::list _list) {
        auto _types = pytim::get_enum_set(_list);
        if(_types.empty())
            _types = pytim::get_type_enums<tim::available_types_t>();
        using tuple_type = tim::convert_t<tim::available_types_t, tim::type_list<>>;
        return manager_t::get_storage<tuple_type>::size(_types);
    };
    //----------------------------------------------------------------------------------//
    auto _as_json_classic = [](const pytim::pyenum_set_t& _types) -> std::string {
        using tuple_type = tim::convert_t<tim::available_types_t, tim::type_list<>>;
        auto json_str    = manager_t::get_storage<tuple_type>::serialize(_types);
        if(tim::settings::debug())
            std::cout << "JSON CLASSIC:\n" << json_str << std::endl;
        return json_str;
    };
    //----------------------------------------------------------------------------------//
    auto _as_json_hierarchy = [](const pytim::pyenum_set_t& _types) -> std::string {
        std::stringstream ss;
        {
            using policy_type = tim::policy::output_archive_t<tim::manager>;
            auto oa           = policy_type::get(ss);
            oa->setNextName("timemory");
            oa->startNode();
            get_json(*oa, _types, std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
            oa->finishNode();
        }
        if(tim::settings::debug())
            std::cout << "JSON HIERARCHY:\n" << ss.str() << std::endl;
        return ss.str();
    };
    //----------------------------------------------------------------------------------//
    auto _as_json = [_as_json_classic, _as_json_hierarchy](bool     hierarchy,
                                                           py::list pytypes) {
        auto types    = pytim::get_enum_set(pytypes);
        auto json_str = (hierarchy) ? _as_json_hierarchy(types) : _as_json_classic(types);
        auto json_module = py::module::import("json");
        return json_module.attr("loads")(json_str);
    };
    //----------------------------------------------------------------------------------//
    auto _as_text = [](py::list _pytypes) {
        std::stringstream ss;
        auto              strms = get_stream(pytim::get_enum_set(_pytypes),
                                std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
        for(auto& itr : strms)
        {
            if(itr)
                ss << *itr << std::flush;
        }
        return ss.str();
    };
    //----------------------------------------------------------------------------------//
    auto set_rusage_child = []() {
#if !defined(TIMEMORY_WINDOWS)
        tim::get_rusage_type() = RUSAGE_CHILDREN;
#endif
    };
    //----------------------------------------------------------------------------------//
    auto set_rusage_self = []() {
#if !defined(TIMEMORY_WINDOWS)
        tim::get_rusage_type() = RUSAGE_SELF;
#endif
    };
    //----------------------------------------------------------------------------------//
    auto _get_argv = []() {
        py::module sys = py::module::import("sys");
        try
        {
            auto   argv  = sys.attr("argv").cast<py::list>();
            int    _argc = argv.size();
            char** _argv = new char*[argv.size()];
            for(int i = 0; i < _argc; ++i)
            {
                auto  _str    = argv[i].cast<std::string>();
                char* _argv_i = new char[_str.size() + 1];
                std::strcpy(_argv_i, _str.c_str());
                _argv_i[_str.size()] = '\0';
                _argv[i]             = _argv_i;
            }
            auto _argv_deleter = [](int fargc, char** fargv) {
                for(int i = 0; i < fargc; ++i)
                    delete[] fargv[i];
                delete[] fargv;
            };
            return std::make_tuple(_argc, _argv, _argv_deleter);
        } catch(py::cast_error& e)
        {
            std::cerr << "Cast error in get_argv: " << e.what() << std::endl;
            throw;
        }
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
            char* _argv_i = new char[_str.size() + 1];
            std::strcpy(_argv_i, _str.c_str());
            _argv_i[_str.size()] = '\0';
            _argv[i]             = _argv_i;
        }
        tim::timemory_init(_argc, _argv, _prefix, _suffix);
        auto _manager = tim::manager::instance();
        if(_manager)
            _manager->update_metadata_prefix();
        for(int i = 0; i < _argc; ++i)
            delete[] _argv[i];
        delete[] _argv;
    };
    //----------------------------------------------------------------------------------//
    auto _finalize = []() {
        auto _manager = tim::manager::instance();
        if(!_manager || _manager->is_finalized() || _manager->is_finalizing())
            return;
        _manager->update_metadata_prefix();
        if(!tim::get_env("TIMEMORY_SKIP_FINALIZE", false))
        {
            try
            {
                py::module gc = py::module::import("gc");
                gc.attr("collect")();
            } catch(std::exception& e)
            {
                std::cerr << e.what() << std::endl;
                // w/o GC, may cause problems
                tim::settings::stack_clearing() = false;
            }
            //
            try
            {
                tim::timemory_finalize();
            } catch(std::exception& e)
            {
#if defined(TIMEMORY_UNIX)
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
        }
    };
    //----------------------------------------------------------------------------------//
    auto _argparse = [tim](py::object parser, py::object subparser) {
        try
        {
            py::object _pysettings = tim.attr("settings");
            _pysettings.attr("add_arguments")(parser, py::none{}, subparser);
        } catch(std::exception& e)
        {
            std::cerr << "[timemory_argparse]> Warning! " << e.what() << std::endl;
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
    //
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
    tim.def("timemory_init", _init, "Initialize timemory", py::arg("argv") = py::list{},
            py::arg("prefix") = "timemory-", py::arg("suffix") = "-output");
    //----------------------------------------------------------------------------------//
    tim.def("timemory_finalize", _finalize,
            "Finalize timemory (generate output) -- important to call if using MPI");
    //----------------------------------------------------------------------------------//
    tim.def("initialize", _init, "Initialize timemory", py::arg("argv") = py::list{},
            py::arg("prefix") = "timemory-", py::arg("suffix") = "-output");
    //----------------------------------------------------------------------------------//
    tim.def("finalize", _finalize,
            "Finalize timemory (generate output) -- important to call if using MPI");
    //----------------------------------------------------------------------------------//
    tim.def("timemory_argparse", _argparse, "Add argparse support for settings",
            py::arg("parser"), py::arg("subparser") = true);
    //----------------------------------------------------------------------------------//
    tim.def("add_arguments", _argparse, "Add argparse support for settings",
            py::arg("parser"), py::arg("subparser") = true);
    //----------------------------------------------------------------------------------//
    tim.def("get", _as_json, "Get the storage data in JSON format",
            py::arg("hierarchy") = false, py::arg("components") = py::list{});
    //----------------------------------------------------------------------------------//
    tim.def("get_text", _as_text, "Get the storage data in text format",
            py::arg("components") = py::list{});
    //----------------------------------------------------------------------------------//
    tim.def("clear", _do_clear, "Clear the storage data",
            py::arg("components") = py::list{});
    //----------------------------------------------------------------------------------//
    tim.def("size", _get_size,
            "Get the current storage size of component types. An empty list as the first "
            "argument will return the size for all available types",
            py::arg("components") = py::list{});
    //----------------------------------------------------------------------------------//
    tim.def("get_hash", [](const std::string& _id) { return tim::add_hash_id(_id); },
            "Get timemory's hash for a key (string)", py::arg("key"));
    //----------------------------------------------------------------------------------//
    tim.def("get_hash_identifier",
            py::overload_cast<tim::hash_value_t>(&tim::get_hash_identifier),
            "Get the string associated with a hash identifer", py::arg("hash_id"));
    //----------------------------------------------------------------------------------//
    tim.def("add_hash_id", [](const std::string& _id) { return tim::add_hash_id(_id); },
            "Add a key (string) to the database and return the hash for it",
            py::arg("key"));
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
    tim.def("init_mpip", &timemory_start_mpip,
            "Activate MPIP profiling (deprecated -- use start_mpip instead)");
    //----------------------------------------------------------------------------------//
    tim.def("init_ompt", &timemory_start_ompt,
            "Activate OMPT profiling (deprecated -- use start_ompt instead)");
    //----------------------------------------------------------------------------------//
    tim.def("start_mpip", &timemory_start_mpip, "Activate MPIP profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_mpip", &timemory_stop_mpip, "Deactivate MPIP profiling", py::arg("id"));
    //----------------------------------------------------------------------------------//
    tim.def("start_ompt", &timemory_start_ompt, "Activate OMPT (OpenMP tools) profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_ompt", &timemory_stop_ompt, "Deactivate OMPT (OpenMP tools)  profiling",
            py::arg("id"));
    //----------------------------------------------------------------------------------//
    tim.def("start_ncclp", &timemory_start_ncclp, "Activate NCCL profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_ncclp", &timemory_stop_ncclp, "Deactivate NCCL profiling",
            py::arg("id"));
    //----------------------------------------------------------------------------------//
    tim.def("start_mallocp", &timemory_start_mallocp,
            "Activate Memory Allocation profiling");
    //----------------------------------------------------------------------------------//
    tim.def("stop_mallocp", &timemory_stop_mallocp,
            "Deactivate Memory Allocation profiling", py::arg("id"));

    //==================================================================================//
    //
    //                          TIMING MANAGER
    //
    //==================================================================================//
    man.attr("text_files") = py::list{};
    //----------------------------------------------------------------------------------//
    man.attr("json_files") = py::list{};
    //----------------------------------------------------------------------------------//
    man.def(py::init([]() { return new manager_wrapper{}; }), "Initialization",
            py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    man.def("write_ctest_notes", &pytim::manager::write_ctest_notes,
            "Write a CTestNotes.cmake file", py::arg("directory") = ".",
            py::arg("append") = false);
    //----------------------------------------------------------------------------------//
    auto _add_metadata = [](std::string _label, py::object _data) {
        std::stringstream _msg;
        auto _f = [_label](auto _value) { tim::manager::add_metadata(_label, _value); };
        bool _success =
            pytim::try_cast_seq<std::string, size_t, double, std::vector<std::string>,
                                std::vector<size_t>, std::vector<double>>(_f, _data,
                                                                          &_msg);
        if(!_success)
            throw std::runtime_error(_msg.str());
    };
    man.def_static("add_metadata", _add_metadata, "Add metadata");
    //----------------------------------------------------------------------------------//
    auto _get_metadata = []() {
        std::stringstream _data;
        tim::manager::master_instance()->write_metadata(_data);
        auto     json_module = py::module::import("json");
        py::dict _metadata   = json_module.attr("loads")(_data.str());
        if(_metadata.contains("timemory"))
        {
            _metadata = _metadata["timemory"];
            if(_metadata.contains("metadata"))
                _metadata = _metadata["metadata"];
        }
        return _metadata;
    };
    man.def_static("get_metadata", _get_metadata, "Get the metadata dictionary");
    //----------------------------------------------------------------------------------//

    //==================================================================================//
    //
    //      Options submodule
    //
    //==================================================================================//

    opts.def("safe_mkdir", &pytim::opt::safe_mkdir,
             "if [ ! -d <directory> ]; then mkdir -p <directory> ; fi");

    opts.def("ensure_directory_exists", &pytim::opt::ensure_directory_exists,
             "mkdir -p $(basename file_path)");

    opts.def("parse_args", &pytim::opt::parse_args,
             "Parse the command-line arguments via ArgumentParser.parse_args()",
             py::arg("parser") = py::none{});

    opts.def("parse_known_args", &pytim::opt::parse_known_args,
             "Parse the command-line arguments via ArgumentParser.parse_known_args()",
             py::arg("parser") = py::none{});

    static auto _add_arguments = [](py::object parser, py::object subparser) {
        auto locals = py::dict("parser"_a = parser, "subparser"_a = subparser);
        py::exec(R"(
        import argparse
        from timemory import settings

        if parser is None:
            parser = argparse.ArgumentParser()

        settings.add_argparse(parser, subparser=subparser)
        parser.add_argument('--timemory-echo-dart', required=False,
                            action='store_true', help="Echo dart tags for CDash")
        parser.add_argument('--timemory-mpl-backend', required=False,
                            default="default", type=str, help="Matplotlib backend")
        )",
                 py::globals(), locals);
        return locals["parser"].cast<py::object>();
    };

    auto _add_args_and_parse = [](py::object parser, py::object subparser) {
        _add_arguments(parser, subparser);
        return pytim::opt::parse_args(parser);
    };

    auto _add_args_and_parse_known = [](py::object parser, py::object subparser) {
        _add_arguments(parser, subparser);
        return pytim::opt::parse_known_args(parser);
    };

    opts.def("add_arguments", _add_arguments, "Function to add command-line arguments",
             py::arg("parser") = py::none{}, py::arg("subparser") = true);

    opts.def("add_args_and_parse", _add_args_and_parse,
             "Combination of timing.add_arguments and timing.parse_args but returns",
             py::arg("parser") = py::none{}, py::arg("subparser") = true);

    opts.def("add_args_and_parse_known", _add_args_and_parse_known,
             "Combination of add_arguments and parse_args. Returns "
             "timemory args and replaces sys.argv with the unknown args (used to "
             "fix issue with unittest module)",
             py::arg("parser") = py::none{}, py::arg("subparser") = true);

#if !defined(TIMEMORY_WINDOWS) || defined(TIMEMORY_USE_WINSOCK)
    py::module _socket = tim.def_submodule("socket", "Socket communication API");

    using socket_manager_t = std::unique_ptr<tim::socket::manager>;
    auto _socket_manager   = []() -> socket_manager_t& {
        static auto _instance = std::make_unique<tim::socket::manager>();
        return _instance;
    };

    auto _socket_connect = [_socket_manager](const std::string& _name,
                                             const std::string& _addr, int _port) {
        return _socket_manager()->connect(_name, _addr, _port);
    };

    auto _socket_send = [_socket_manager](const std::string& _name,
                                          const std::string& _message) {
        return _socket_manager()->send(_name, _message);
    };

    auto _socket_close = [_socket_manager](const std::string& _name) {
        return _socket_manager()->close(_name);
    };

    auto _socket_listen = [](const std::string& _name, int _port, int _max_packets) {
        std::vector<std::string> _results;

        auto _handle_data = [&](std::string str) {
            if(tim::settings::debug() || tim::settings::verbose() > 2)
                std::cout << "[timemory-socket][server]> received: " << str << std::endl;
            _results.emplace_back(std::move(str));
        };

        if(tim::settings::debug() || tim::settings::verbose() > 2)
            std::cout << "[timemory-socket][server]> started listening..." << std::endl;

        tim::socket::manager{}.listen(_name, _port, _handle_data, _max_packets);

        if(tim::settings::debug() || tim::settings::verbose() > 2)
            std::cout << "[timemory-socket][server]> stopped listening..." << std::endl;

        return _results;
    };

    _socket.def("connect", _socket_connect, "Connect to a socket (client)",
                py::arg("name"), py::arg("address"), py::arg("port"));
    _socket.def("close", _socket_close, "Close a socket", py::arg("name"));
    _socket.def("send", _socket_send, "Send a message over the socket", py::arg("name"),
                py::arg("message"));
    _socket.def("listen", _socket_listen, "Listen on a socket (server)", py::arg("name"),
                py::arg("port"), py::arg("max_packets") = 0);
#endif
}
