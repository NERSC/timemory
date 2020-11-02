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

#pragma once

#include "timemory/backends/process.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/manager/macros.hpp"
#include "timemory/manager/types.hpp"
#include "timemory/utility/signals.hpp"

//--------------------------------------------------------------------------------------//
//
//          if not using extern and not compiling manager library, everything
//          in this file gets excluded by the pre-processor
//
//--------------------------------------------------------------------------------------//
//
#if(!defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_MANAGER_EXTERN)) ||           \
    defined(TIMEMORY_MANAGER_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
#    include "timemory/api.hpp"
#    include "timemory/backends/threading.hpp"
#    include "timemory/mpl/policy.hpp"
#    include "timemory/mpl/type_traits.hpp"
#    include "timemory/operations/types/finalize/ctest_notes.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/utility/macros.hpp"

#    include <algorithm>
#    include <atomic>
#    include <fstream>
#    include <iosfwd>
#    include <memory>
#    include <string>
#    include <utility>
#    include <vector>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              manager
//
//----------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_MANAGER_LINKAGE_API)
#        if defined(TIMEMORY_MANAGER_SOURCE)
#            define TIMEMORY_MANAGER_LINKAGE_API
#        else
#            define TIMEMORY_MANAGER_LINKAGE_API inline
#        endif
#    endif

TIMEMORY_MANAGER_LINKAGE_API
manager::manager()
: m_is_finalizing(false)
, m_is_finalized(false)
, m_write_metadata(0)
, m_instance_count(f_manager_instance_count()++)
, m_rank(dmp::rank())
, m_num_entries(0)
, m_thread_index(threading::get_id())
, m_thread_id(std::this_thread::get_id())
, m_metadata_prefix("")
, m_lock(std::make_shared<auto_lock_t>(m_mutex, std::defer_lock))
, m_hash_ids(get_hash_ids())
, m_hash_aliases(get_hash_aliases())
{
    f_thread_counter()++;
    static std::atomic<int> _once(0);

    bool _first = (_once++ == 0);

    if(_first)
    {
        settings::parse();
        // papi::init();
        // std::atexit(manager::exit_hook);
    }

#    if !defined(TIMEMORY_DISABLE_BANNER)
    if(_first && m_settings->get_banner())
        printf("#------------------------- tim::manager initialized "
               "[id=%i][pid=%i] "
               "-------------------------#\n",
               m_instance_count, process::get_id());
#    endif

    if(settings::cpu_affinity())
        threading::affinity::set();
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE_API
manager::~manager()
{
    auto _remain = --f_manager_instance_count();
    bool _last   = (get_shared_ptr_pair<this_type, TIMEMORY_API>().second == nullptr ||
                  _remain == 0 || m_instance_count == 0);

    if(_last)
    {
        f_thread_counter().store(0, std::memory_order_relaxed);
    }

#    if !defined(TIMEMORY_DISABLE_BANNER)
    if(_last && m_settings && m_settings->get_banner())
    {
        printf("#---------------------- tim::manager destroyed "
               "[rank=%i][id=%i][pid=%i] "
               "----------------------#\n",
               m_rank, m_instance_count, process::get_id());
    }
#    endif
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::cleanup(const std::string& key)
{
    if(f_debug())
        PRINT_HERE("cleaning %s", key.c_str());

    auto itr = m_finalizer_cleanups.begin();
    for(; itr != m_finalizer_cleanups.end(); ++itr)
    {
        if(itr->first == key)
            break;
    }

    if(itr != m_finalizer_cleanups.end())
    {
        itr->second();
        m_finalizer_cleanups.erase(itr);
    }

    if(f_debug())
        PRINT_HERE("%s [size: %i]", "cleaned", (int) m_finalizer_cleanups.size());
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::cleanup()
{
    m_is_finalizing = true;
    if(f_debug())
        PRINT_HERE("%s [size: %i]", "cleaning", (int) m_finalizer_cleanups.size());

    auto _cleanup = [](finalizer_list_t& _functors) {
        // reverse to delete the most recent additions first
        std::reverse(_functors.begin(), _functors.end());
        // invoke all the functions
        for(auto& itr : _functors)
            itr.second();
        // remove all these functors
        _functors.clear();
    };

    _cleanup(m_finalizer_cleanups);

    if(f_debug())
        PRINT_HERE("%s [size: %i]", "cleaned", (int) m_finalizer_cleanups.size());
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::finalize()
{
    m_is_finalizing = true;
    m_rank          = std::max<int32_t>(m_rank, dmp::rank());
    if(f_debug())
        PRINT_HERE("%s [master: %i/%i, worker: %i/%i, other: %i]", "finalizing",
                   (int) m_master_cleanup.size(), (int) m_master_finalizers.size(),
                   (int) m_worker_cleanup.size(), (int) m_worker_finalizers.size(),
                   (int) m_pointer_fini.size());

    cleanup();

    auto _finalize = [](finalizer_list_t& _functors) {
        // reverse to delete the most recent additions first
        std::reverse(_functors.begin(), _functors.end());
        // invoke all the functions
        for(auto& itr : _functors)
            itr.second();
        // remove all these finalizers
        _functors.clear();
    };

    if(f_debug())
        PRINT_HERE("%s [master: %i/%i, worker: %i/%i, other: %i]", "finalizing",
                   (int) m_master_cleanup.size(), (int) m_master_finalizers.size(),
                   (int) m_worker_cleanup.size(), (int) m_worker_finalizers.size(),
                   (int) m_pointer_fini.size());

    //
    //  ideally, only one of these will be populated
    //  these clear the stack before outputting
    //
    // finalize workers first
    _finalize(m_worker_cleanup);
    // finalize masters second
    _finalize(m_master_cleanup);

    if(f_debug())
        PRINT_HERE("%s [master: %i/%i, worker: %i/%i, other: %i]", "finalizing",
                   (int) m_master_cleanup.size(), (int) m_master_finalizers.size(),
                   (int) m_worker_cleanup.size(), (int) m_worker_finalizers.size(),
                   (int) m_pointer_fini.size());

    //
    //  ideally, only one of these will be populated
    //
    // finalize workers first
    _finalize(m_worker_finalizers);
    // finalize masters second
    _finalize(m_master_finalizers);

    if(f_debug())
        PRINT_HERE("%s [master: %i/%i, worker: %i/%i, other: %i]", "finalizing",
                   (int) m_master_cleanup.size(), (int) m_master_finalizers.size(),
                   (int) m_worker_cleanup.size(), (int) m_worker_finalizers.size(),
                   (int) m_pointer_fini.size());

    for(auto& itr : m_pointer_fini)
        itr.second();

    m_pointer_fini.clear();

    m_is_finalizing = false;

    if(f_debug())
        PRINT_HERE("%s [master: %i/%i, worker: %i/%i, other: %i]", "finalized",
                   (int) m_master_cleanup.size(), (int) m_master_finalizers.size(),
                   (int) m_worker_cleanup.size(), (int) m_worker_finalizers.size(),
                   (int) m_pointer_fini.size());

    if(m_instance_count == 0 && m_rank == 0)
    {
        operation::finalize::ctest_notes<manager>::get_notes().reset();
        write_metadata("manager::finalize");
    }

    m_is_finalized = true;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::exit_hook()
{
    if(f_debug())
        PRINT_HERE("%s", "finalizing...");

    if(!manager::f_use_exit_hook())
        return;

    try
    {
        auto master_count = f_manager_instance_count().load();
        if(master_count > 0)
        {
            auto master_manager =
                get_shared_ptr_pair_master_instance<manager, TIMEMORY_API>();
            if(master_manager)
            {
                master_manager->write_metadata("manager::exit_hook");
                master_manager.reset();
            }
        }
    } catch(...)
    {}

    if(f_debug())
        PRINT_HERE("%s", "finalizing...");
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::update_metadata_prefix()
{
    m_rank = std::max<int32_t>(m_rank, dmp::rank());
    if(m_write_metadata < 0)
        return;
    auto _settings = f_settings();
    if(!_settings)
        return;
    auto _outp_prefix = _settings->get_output_prefix();
    m_metadata_prefix = _outp_prefix;
    if(f_debug())
        PRINT_HERE("[rank=%i][id=%i] metadata prefix: '%s'", m_rank, m_instance_count,
                   m_metadata_prefix.c_str());
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::write_metadata(const char* context)
{
    if(m_rank != 0)
    {
        if(f_debug())
            PRINT_HERE("[%s]> metadata disabled for rank %i", context, (int) m_rank);
        return;
    }

    if(m_num_entries < 1)
    {
        if(f_debug())
            PRINT_HERE("[%s]> No components generated output. Skipping metadata",
                       context);
        return;
    }

    if(tim::get_env<bool>("TIMEMORY_CXX_PLOT_MODE", false))
    {
        if(f_debug())
            PRINT_HERE("[%s]> plot mode enabled. Skipping metadata", context);
        return;
    }
    auto _settings = f_settings();
    if(!_settings)
    {
        if(f_debug())
            PRINT_HERE("[%s]> Null pointer to settings", context);
        return;
    }

    bool _banner      = _settings->get_banner();
    bool _auto_output = _settings->get_auto_output();
    bool _file_output = _settings->get_file_output();
    auto _outp_prefix = _settings->get_global_output_prefix();

    static bool written = false;
    if(written || m_write_metadata < 1)
    {
        if(f_debug() && written)
            PRINT_HERE("[%s]> metadata already written", context);
        if(f_debug() && m_write_metadata)
            PRINT_HERE("[%s]> metadata disabled: %i", context, (int) m_write_metadata);
        return;
    }

    if(!_auto_output || !_file_output)
    {
        if(f_debug() && !_auto_output)
            PRINT_HERE("[%s]> metadata disabled because auto output disabled", context);
        if(f_debug() && !_file_output)
            PRINT_HERE("[%s]> metadata disabled because file output disabled", context);
        return;
    }

    written          = true;
    m_write_metadata = -1;

    if(f_debug())
        PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    // get the output prefix if not already set
    if(m_metadata_prefix.empty())
        m_metadata_prefix = _outp_prefix;

    if(f_debug())
        PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    // remove any non-ascii characters
    auto only_ascii = [](char c) { return !isascii(c); };
    m_metadata_prefix.erase(
        std::remove_if(m_metadata_prefix.begin(), m_metadata_prefix.end(), only_ascii),
        m_metadata_prefix.end());

    if(f_debug())
        PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    // if empty, set to default
    if(m_metadata_prefix.empty())
        m_metadata_prefix = "timemory-output/";

    if(f_debug())
        PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    // if first char is a control character, the statics probably got deleted
    if(iscntrl(m_metadata_prefix[0]))
        m_metadata_prefix = "timemory-output/";

    if(f_debug())
        PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    auto fname = settings::compose_output_filename("metadata", "json", false, -1, false,
                                                   m_metadata_prefix);
    consume_parameters(fname);

    if(f_verbose() > 0 || _banner || f_debug())
        printf("\n[metadata::%s]> Outputting '%s'...\n", context, fname.c_str());

    std::ofstream ofs(fname.c_str());
    if(ofs)
    {
        // ensure json write final block during destruction before the file is closed
        using policy_type = policy::output_archive_t<manager>;
        auto oa           = policy_type::get(ofs);
        oa->setNextName("timemory");
        oa->startNode();
        {
            oa->setNextName("metadata");
            oa->startNode();
            // settings
            {
                settings::serialize_settings(*oa, *(_settings.get()));
            }
            // output
            {
                oa->setNextName("output");
                oa->startNode();
                for(const auto& itr : m_output_files)
                    (*oa)(cereal::make_nvp(itr.first.c_str(), itr.second));
                oa->finishNode();
            }
            // environment
            {
                env_settings::serialize_environment(*oa);
            }
            //
            oa->finishNode();
        }
        oa->finishNode();
    }
    if(ofs)
        ofs << std::endl;
    else
        printf("[manager]> Warning! Error opening '%s'...\n", fname.c_str());

    ofs.close();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::add_file_output(const string_t& _category, const string_t& _label,
                         const string_t& _file)
{
    m_output_files[_category][_label].insert(_file);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::add_text_output(const string_t& _label, const string_t& _file)
{
    add_file_output("text", _label, _file);
    auto _settings = f_settings();
    if(_settings && _settings->get_ctest_notes())
        operation::finalize::ctest_notes<manager>::get_notes()->insert(_file);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::add_json_output(const string_t& _label, const string_t& _file)
{
    add_file_output("json", _label, _file);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::remove_cleanup(const std::string& _key)
{
    auto _remove_functor = [&](finalizer_list_t& _functors) {
        for(auto itr = _functors.begin(); itr != _functors.end(); ++itr)
        {
            if(itr->first == _key)
            {
                _functors.erase(itr);
                return;
            }
        }
    };

    _remove_functor(m_finalizer_cleanups);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::remove_finalizer(const std::string& _key)
{
    auto _remove_finalizer = [&](finalizer_list_t& _functors) {
        for(auto itr = _functors.begin(); itr != _functors.end(); ++itr)
        {
            if(itr->first == _key)
            {
                _functors.erase(itr);
                return;
            }
        }
    };

    _remove_finalizer(m_master_cleanup);
    _remove_finalizer(m_worker_cleanup);
    _remove_finalizer(m_master_finalizers);
    _remove_finalizer(m_worker_finalizers);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(manager::comm_group_t)
manager::get_communicator_group()
{
    int32_t max_concurrency = std::thread::hardware_concurrency();
    // We want on-node communication only
    int32_t nthreads         = f_thread_counter().load();
    int32_t max_processes    = max_concurrency / nthreads;
    int32_t mpi_node_default = mpi::size() / max_processes;
    if(mpi_node_default < 1)
        mpi_node_default = 1;
    int32_t mpi_node_count = get_env<int32_t>("TIMEMORY_NODE_COUNT", mpi_node_default);
    int32_t mpi_split_size = mpi::rank() / (mpi::size() / mpi_node_count);

    // Split the communicator based on the number of nodes and use the
    // original rank for ordering
    mpi::comm_t local_mpi_comm;
    mpi::comm_split(mpi::comm_world_v, mpi_split_size, mpi::rank(), &local_mpi_comm);

#    if defined(DEBUG)
    if(f_verbose() > 1 || f_debug())
    {
        int32_t local_mpi_rank = mpi::rank(local_mpi_comm);
        int32_t local_mpi_size = mpi::size(local_mpi_comm);
        int32_t local_mpi_file = mpi::rank() / local_mpi_size;

        std::stringstream _info;
        _info << "\t" << mpi::rank() << " Rank      : " << mpi::rank() << std::endl;
        _info << "\t" << mpi::rank() << " Size      : " << mpi::size() << std::endl;
        _info << "\t" << mpi::rank() << " Node      : " << mpi_node_count << std::endl;
        _info << "\t" << mpi::rank() << " Local Size: " << local_mpi_size << std::endl;
        _info << "\t" << mpi::rank() << " Local Rank: " << local_mpi_rank << std::endl;
        _info << "\t" << mpi::rank() << " Local File: " << local_mpi_file << std::endl;
        std::cout << "tim::manager::" << __FUNCTION__ << "\n" << _info.str();
    }
#    endif

    auto local_rank = mpi::rank() / mpi::size(local_mpi_comm);
    // check
    assert(local_rank == mpi::get_node_index());

    return comm_group_t(local_mpi_comm, local_rank);
}
//
//----------------------------------------------------------------------------------//
//
// persistent data for instance counting, threading counting, and exit-hook control
//
TIMEMORY_MANAGER_LINKAGE(manager::persistent_data&)
manager::f_manager_persistent_data()
{
    static persistent_data _instance{};
    return _instance;
}
//
//----------------------------------------------------------------------------------//
//
// get either master or thread-local instance
//
TIMEMORY_MANAGER_LINKAGE(manager::pointer_t)
manager::instance()
{
    static thread_local auto _inst =
        get_shared_ptr_pair_instance<manager, TIMEMORY_API>();
    return _inst;
}
//
//----------------------------------------------------------------------------------//
//
// get master instance
//
TIMEMORY_MANAGER_LINKAGE(manager::pointer_t)
manager::master_instance()
{
    static auto _pinst = get_shared_ptr_pair_master_instance<manager, TIMEMORY_API>();
    manager::f_manager_persistent_data().master_instance = _pinst;
    return _pinst;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(manager*)
timemory_manager_master_instance()
{
    static auto _pinst = tim::get_shared_ptr_pair<manager, TIMEMORY_API>();
    manager::set_persistent_master(_pinst.first);
    return _pinst.first.get();
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
timemory_library_constructor()
{
    static auto _preloaded = []() {
        auto library_ctor = tim::get_env<bool>("TIMEMORY_LIBRARY_CTOR", true);
        if(!library_ctor)
            return true;

        auto       ld_preload   = tim::get_env<std::string>("LD_PRELOAD", "");
        auto       dyld_preload = tim::get_env<std::string>("DYLD_INSERT_LIBRARIES", "");
        std::regex lib_regex("libtimemory");
        if(std::regex_search(ld_preload, lib_regex) ||
           std::regex_search(dyld_preload, lib_regex))
        {
            tim::set_env("TIMEMORY_LIBRARY_CTOR", "OFF", 1);
            return true;
        }
        return false;
    }();

    if(_preloaded)
        return;

    static thread_local bool _once = false;
    if(_once)
        return;
    _once = true;

    auto _debug   = tim::settings::debug();
    auto _verbose = tim::settings::verbose();

    if(_debug || _verbose > 3)
        printf("[%s]> initializing manager...\n", __FUNCTION__);

    auto                     _inst   = timemory_manager_master_instance();
    static auto              _master = manager::master_instance();
    static thread_local auto _worker = manager::instance();

    if(!_master && _inst)
        _master.reset(_inst);
    else if(!_master)
        _master = manager::master_instance();

    if(_worker == _master)
    {
        // this will create a recursive-dynamic-library load situation
        // since the timemory-config library depends on the manager library
        // std::atexit(tim::timemory_finalize);
    }
    else
    {
        printf("[%s]> manager :: master != worker : %p vs. %p. TLS behavior is abnormal. "
               "Report any issues to https://github.com/NERSC/timemory/issues\n",
               __FUNCTION__, (void*) _master.get(), (void*) _worker.get());
        if(!signal_settings::is_active())
        {
            enable_signal_detection({ sys_signal::SegFault, sys_signal::Bus });
            auto _exit_action = [](int nsig) {
                auto _manager = manager::instance();
                if(_manager)
                {
                    std::cout << "Finalizing after signal: " << nsig << std::endl;
                    _manager->finalize();
                }
            };
            signal_settings::set_exit_action(_exit_action);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
#    if defined(TIMEMORY_MANGER_INLINE)
namespace
{
static auto timemory_library_is_constructed = (timemory_library_constructor(), true);
}
#    endif
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#endif
