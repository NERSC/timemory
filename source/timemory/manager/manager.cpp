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

// maybe included directly in header-only mode but pragma once will cause warnings
#ifndef TIMEMORY_MANAGER_MANAGER_CPP_
#define TIMEMORY_MANAGER_MANAGER_CPP_ 1

#include "timemory/manager/manager.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/manager/macros.hpp"
#include "timemory/manager/types.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/utility/signals.hpp"

//--------------------------------------------------------------------------------------//
//
//          if not using extern and not compiling manager library, everything
//          in this file gets excluded by the pre-processor
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_MANAGER_SOURCE) || defined(TIMEMORY_MANAGER_INLINE)
//
//--------------------------------------------------------------------------------------//
//
#    include "timemory/api.hpp"
#    include "timemory/backends/cpu.hpp"
#    include "timemory/backends/threading.hpp"
#    include "timemory/mpl/policy.hpp"
#    include "timemory/mpl/type_traits.hpp"
#    include "timemory/operations/types/finalize/ctest_notes.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/utility/macros.hpp"
#    include "timemory/version.h"

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

    static bool _once = [=] {
        auto _launch_date =
            get_local_datetime("%D", settings::get_launch_time(TIMEMORY_API{}));
        auto _launch_time =
            get_local_datetime("%H:%M", settings::get_launch_time(TIMEMORY_API{}));
        auto _cpu_info = cpu::get_info();
        auto _user     = get_env<std::string>("USER", "nobody");

        for(auto&& itr : { "SHELL", "HOME", "PWD" })
        {
            auto _var = get_env<std::string>(itr, "");
            if(!_var.empty())
                add_metadata(itr, _var);
        }

        add_metadata("USER", _user);
        add_metadata("LAUNCH_DATE", _launch_date);
        add_metadata("LAUNCH_TIME", _launch_time);
        add_metadata("TIMEMORY_API", demangle<TIMEMORY_API>());

#    if defined(TIMEMORY_VERSION_STRING)
        add_metadata("TIMEMORY_VERSION", TIMEMORY_VERSION_STRING);
#    endif

#    if defined(TIMEMORY_GIT_DESCRIBE)
        add_metadata("TIMEMORY_GIT_DESCRIBE", TIMEMORY_GIT_DESCRIBE);
#    endif

#    if defined(TIMEMORY_GIT_REVISION)
        add_metadata("TIMEMORY_GIT_REVISION", TIMEMORY_GIT_REVISION);
#    endif

        add_metadata("CPU_MODEL", _cpu_info.model);
        add_metadata("CPU_VENDOR", _cpu_info.vendor);
        add_metadata("CPU_FEATURES", _cpu_info.features);
        add_metadata("CPU_FREQUENCY", _cpu_info.frequency);
        add_metadata("HW_CONCURRENCY", threading::affinity::hw_concurrency());
        add_metadata("HW_PHYSICAL_CPU", threading::affinity::hw_physicalcpu());

        for(int i = 0; i < 3; ++i)
        {
            auto              _cache_size = cpu::cache_size::get(i + 1);
            std::stringstream _cache_lvl;
            _cache_lvl << "HW_L" << (i + 1) << "_CACHE_SIZE";
            add_metadata(_cache_lvl.str(), _cache_size);
        }

#    if !defined(TIMEMORY_DISABLE_BANNER)
        if(m_settings->get_banner())
            printf("#------------------------- tim::manager initialized "
                   "[pid=%i] -------------------------#\n",
                   process::get_id());
#    endif
        return true;
    }();
    consume_parameters(_once);

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
               "[rank=%i][pid=%i] "
               "----------------------#\n",
               m_rank, process::get_id());
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

    auto _cleanup = [&](auto& _functors) {
        auto _orig = _functors.size();
        auto itr   = _functors.begin();
        for(; itr != _functors.end(); ++itr)
        {
            if(itr->first == key)
                break;
        }

        if(itr != _functors.end())
        {
            itr->second();
            _functors.erase(itr);
        }

        if(f_debug())
            PRINT_HERE("%s [size: %i]", "cleaned", (int) (_orig - _functors.size()));
    };

    _cleanup(m_finalizer_cleanups);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::cleanup()
{
    m_is_finalizing = true;
    auto _orig_sz   = m_finalizer_cleanups.size();

    if(f_debug())
    {
        PRINT_HERE("%s [size: %i]", "cleaning", (int) m_finalizer_cleanups.size());
    }

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
    {
        PRINT_HERE("%s [size: %i]", "cleaned",
                   (int) (_orig_sz - m_finalizer_cleanups.size()));
    }
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::finalize()
{
    auto _print_size = [&](const char* _msg) {
        if(f_debug())
        {
            auto _get_sz = [](auto& _mdata) {
                int _val = 0;
                for(auto& itr : _mdata)
                    _val += itr.second.size();
                return _val;
            };
            PRINT_HERE("%s [master: %i/%i, worker: %i/%i, other: %i]", _msg,
                       _get_sz(m_master_cleanup), _get_sz(m_master_finalizers),
                       _get_sz(m_worker_cleanup), _get_sz(m_worker_finalizers),
                       (int) m_pointer_fini.size());
        }
    };

    auto _finalize = [](finalizer_pmap_t& _pdata) {
        for(auto& _functors : _pdata)
        {
            // reverse to delete the most recent additions first
            std::reverse(_functors.second.begin(), _functors.second.end());
            // invoke all the functions
            for(auto& itr : _functors.second)
                itr.second();
            // remove all these finalizers
            _functors.second.clear();
        }
    };

    m_is_finalizing = true;
    m_rank          = std::max<int32_t>(m_rank, dmp::rank());

    _print_size("finalizing");

    cleanup();

    _print_size("finalizing (pre-storage-cleanup)");

    //
    //  ideally, only one of these will be populated
    //  these clear the stack before outputting
    //
    // finalize workers first
    _finalize(m_worker_cleanup);
    // finalize masters second
    _finalize(m_master_cleanup);

    _print_size("finalizing (pre-storage-finalization)");

    //
    //  ideally, only one of these will be populated
    //
    // finalize workers first
    _finalize(m_worker_finalizers);
    // finalize masters second
    _finalize(m_master_finalizers);

    _print_size("finalizing (pre-pointer-finalization)");

    for(auto& itr : m_pointer_fini)
        itr.second();

    m_pointer_fini.clear();

    m_is_finalizing = false;

    _print_size("finalized");

    if(m_instance_count == 0 && m_rank == 0)
    {
        operation::finalize::ctest_notes<manager>::get_notes().reset();
        internal_write_metadata("manager::finalize");
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
                get_shared_ptr_pair_main_instance<manager, TIMEMORY_API>();
            if(master_manager)
            {
                master_manager->internal_write_metadata("manager::exit_hook");
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
    auto _outp_prefix = _settings->get_global_output_prefix();
    m_metadata_prefix = _outp_prefix;
    if(f_debug())
        PRINT_HERE("[rank=%i][id=%i] metadata prefix: '%s'", m_rank, m_instance_count,
                   m_metadata_prefix.c_str());
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(std::ostream&)
manager::write_metadata(std::ostream& ofs)
{
    auto_lock_t _lk(type_mutex<manager>(), std::defer_lock);

    if(!_lk.owns_lock())
        _lk.lock();

    auto _settings = f_settings();

    // ensure json write final block during destruction before the file is closed
    using policy_type = policy::output_archive_t<manager>;
    auto oa           = policy_type::get(ofs);
    oa->setNextName("timemory");
    oa->startNode();
    {
        oa->setNextName("metadata");
        oa->startNode();
        // user
        {
            auto& _info_metadata = f_manager_persistent_data().info_metadata;
            auto& _func_metadata = f_manager_persistent_data().func_metadata;
            oa->setNextName("info");
            oa->startNode();
            for(const auto& itr : _info_metadata)
                (*oa)(cereal::make_nvp(itr.first.c_str(), itr.second));
            for(const auto& itr : _func_metadata)
                itr(static_cast<void*>(oa.get()));
            oa->finishNode();
        }
        // settings
        if(_settings)
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
    return ofs;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::write_metadata(const std::string& _output_dir, const char* context)
{
    if(m_rank != 0)
    {
        if(f_debug())
            PRINT_HERE("[%s]> metadata disabled for rank %i", context, (int) m_rank);
        return;
    }

    if(tim::get_env<bool>("TIMEMORY_CXX_PLOT_MODE", false))
    {
        if(f_debug())
            PRINT_HERE("[%s]> plot mode enabled. Skipping metadata", context);
        return;
    }

    auto fname = settings::compose_output_filename("metadata", "json", false, -1, false,
                                                   _output_dir);
    auto hname = settings::compose_output_filename("functions", "json", false, -1, false,
                                                   _output_dir);

    auto _settings = f_settings();
    auto _banner   = (_settings) ? _settings->get_banner() : false;
    if(f_verbose() > 0 || _banner || f_debug())
        printf("\n[metadata::%s]> Outputting '%s' and '%s'...\n", context, fname.c_str(),
               hname.c_str());

    auto_lock_t _lk(type_mutex<manager>());

    std::ofstream ofs(fname.c_str());
    if(ofs)
    {
        write_metadata(ofs);
    }
    if(ofs)
        ofs << std::endl;
    else
        printf("[manager]> Warning! Error opening '%s'...\n", fname.c_str());
    ofs.close();

    std::map<std::string, std::set<size_t>> _hashes{};
    if(m_hash_ids && m_hash_aliases)
    {
        for(const auto& itr : (*m_hash_aliases))
        {
            auto hitr = m_hash_ids->find(itr.second);
            if(hitr != m_hash_ids->end())
            {
                _hashes[operation::decode<TIMEMORY_API>{}(hitr->second)].insert(
                    itr.first);
                _hashes[operation::decode<TIMEMORY_API>{}(hitr->second)].insert(
                    hitr->first);
            }
        }
        for(const auto& itr : (*m_hash_ids))
            _hashes[operation::decode<TIMEMORY_API>{}(itr.second)].insert(itr.first);
    }
    if(_hashes.empty())
        return;

    std::ofstream hfs(hname.c_str());
    if(hfs)
    {
        // ensure json write final block during destruction before the file is closed
        using policy_type = policy::output_archive_t<manager>;
        auto oa           = policy_type::get(hfs);
        oa->setNextName("timemory");
        oa->startNode();
        {
            oa->setNextName("functions");
            oa->startNode();
            // hash-keys
            {
                for(const auto& itr : _hashes)
                    (*oa)(cereal::make_nvp(itr.first.c_str(), itr.second));
            }
            //
            oa->finishNode();
        }
        oa->finishNode();
    }
    if(hfs)
        hfs << std::endl;
    else
        printf("[manager]> Warning! Error opening '%s'...\n", hname.c_str());

    hfs.close();
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::internal_write_metadata(const char* context)
{
    if(m_rank != 0)
    {
        if(f_debug())
            PRINT_HERE("[%s]> metadata disabled for rank %i", context, (int) m_rank);
        return;
    }

    if(m_num_entries < 1 && m_write_metadata < 1)
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

    if((!_auto_output || !_file_output) && m_write_metadata < 1)
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

    write_metadata(m_metadata_prefix, context);
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
manager::remove_cleanup(void* _key)
{
    if(m_pointer_fini.find(_key) != m_pointer_fini.end())
        m_pointer_fini.erase(_key);
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
    auto _remove_finalizer = [&](finalizer_pmap_t& _pdata) {
        for(auto& _functors : _pdata)
        {
            for(auto itr = _functors.second.begin(); itr != _functors.second.end(); ++itr)
            {
                if(itr->first == _key)
                {
                    _functors.second.erase(itr);
                    return;
                }
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
TIMEMORY_MANAGER_LINKAGE(void)
manager::add_metadata(const std::string& _key, const char* _value)
{
    auto_lock_t _lk(type_mutex<manager>());
    f_manager_persistent_data().info_metadata.insert({ _key, std::string{ _value } });
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::add_metadata(const std::string& _key, const std::string& _value)
{
    auto_lock_t _lk(type_mutex<manager>());
    f_manager_persistent_data().info_metadata.insert({ _key, _value });
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::add_synchronization(const std::string& _key, int64_t _id,
                             std::function<void()> _func)
{
    m_mutex.lock();
    m_synchronize[_key].emplace(_id, std::move(_func));
    m_mutex.unlock();
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::remove_synchronization(const std::string& _key, int64_t _id)
{
    m_mutex.lock();
    if(m_synchronize[_key].find(_id) != m_synchronize[_key].end())
        m_synchronize[_key].erase(_id);
    m_mutex.unlock();
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(void)
manager::synchronize()
{
    for(auto& itr : m_synchronize)
    {
        for(auto& fitr : itr.second)
        {
            fitr.second();
        }
    }
}
//
//----------------------------------------------------------------------------------//
//
/*TIMEMORY_MANAGER_LINKAGE(manager::comm_group_t)
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
}*/
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
    static auto _pinst = get_shared_ptr_pair_main_instance<manager, TIMEMORY_API>();
    manager::f_manager_persistent_data().master_instance = _pinst;
    return _pinst;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_LINKAGE(bool)
manager::get_is_main_thread()
{
    if(!master_instance())
        return true;
    return (std::this_thread::get_id() == master_instance()->m_thread_id);
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

    auto _settings = tim::settings::shared_instance();
    auto _debug    = (_settings) ? _settings->get_debug() : false;
    auto _verbose  = (_settings) ? _settings->get_verbose() : 0;

    static thread_local bool _once = false;
    if(_once)
        return;
    _once = true;

    auto _inst = timemory_manager_master_instance();
    if(_settings)
    {
        static auto _dir         = _settings->get_output_path();
        static auto _prefix      = _settings->get_output_prefix();
        static auto _time_output = _settings->get_time_output();
        static auto _time_format = _settings->get_time_format();
        tim::consume_parameters(_dir, _prefix, _time_output, _time_format);
    }

    if(_debug || _verbose > 3)
        printf("[%s]> initializing manager...\n", __FUNCTION__);

    auto _master = manager::master_instance();
    auto _worker = manager::instance();

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
            auto default_signals = signal_settings::get_default();
            for(auto& itr : default_signals)
                signal_settings::enable(itr);
            // should return default and any modifications from environment
            auto enabled_signals = signal_settings::get_enabled();
            enable_signal_detection(enabled_signals);
            auto _exit_action = [=](int nsig) {
                if(_master)
                {
                    std::cout << "Finalizing after signal: " << nsig << " :: "
                              << signal_settings::str(static_cast<sys_signal>(nsig))
                              << std::endl;
                    _master->finalize();
                }
            };
            signal_settings::set_exit_action(_exit_action);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
#    if defined(TIMEMORY_MANAGER_INLINE)
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

// formerly in manager/extern.cpp
#if defined(TIMEMORY_MANAGER_SOURCE)
#    include "timemory/manager/extern.hpp"
#    include "timemory/manager/manager.hpp"
#    include "timemory/manager/types.hpp"
//
#    include "timemory/environment.hpp"
#    include "timemory/hash.hpp"
#    include "timemory/plotting.hpp"
#    include "timemory/settings.hpp"
#endif

#endif  // TIMEMORY_MANAGER_MANAGER_CPP_
