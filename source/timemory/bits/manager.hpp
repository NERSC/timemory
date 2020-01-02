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

/** \file manager.hpp
 * \headerfile manager.hpp "timemory/details/manager.hpp"
 * Provides inline implementation of manager functions
 *
 */

#include "timemory/backends/papi.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/general/hash.hpp"
#include "timemory/general/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

//======================================================================================//

namespace tim
{
//======================================================================================//
//
#if !defined(TIMEMORY_EXTERN_INIT)

inline std::atomic<int32_t>&
manager::f_manager_instance_count()
{
    static std::atomic<int32_t> instance(0);
    return instance;
}

//======================================================================================//
// number of threads counter
//
inline std::atomic<int32_t>&
manager::f_thread_counter()
{
    static std::atomic<int32_t> _instance(0);
    return _instance;
}

//======================================================================================//
// get either master or thread-local instance
//
inline manager::pointer_t
manager::instance()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<manager>();
    return _inst;
}

//======================================================================================//
// get master instance
//
inline manager::pointer_t
manager::master_instance()
{
    static auto _pinst = get_shared_ptr_pair_master_instance<manager>();
    return _pinst;
}

#endif
//
//======================================================================================//

inline manager::manager()
: m_is_finalizing(false)
, m_instance_count(f_manager_instance_count()++)
, m_rank(dmp::rank())
, m_metadata_fname(settings::compose_output_filename("metadata", "json"))
, m_thread_id(std::this_thread::get_id())
, m_hash_ids(get_hash_ids())
, m_hash_aliases(get_hash_aliases())
, m_lock(new auto_lock_t(m_mutex, std::defer_lock))
{
    f_thread_counter()++;
    static std::atomic<int> _once(0);

    bool _first = (_once++ == 0);

    if(_first)
    {
        settings::parse();
        papi::init();
        std::atexit(manager::exit_hook);
    }

    // if(settings::banner())
    if(_first && settings::banner())
        printf("#--------------------- tim::manager initialized [%i][%i] "
               "---------------------#\n\n",
               m_rank, m_instance_count);

    if(settings::cpu_affinity())
        threading::affinity::set();
}

//======================================================================================//

inline manager::~manager()
{
    auto _remain = --f_manager_instance_count();

    bool _last = (get_shared_ptr_pair<this_type>().second == nullptr || _remain == 0 ||
                  m_instance_count == 0);

    if(m_rank == 0 && _remain == 0)
        write_metadata("manager::~manager");

    if(_last)
    {
        // papi::shutdown();
        // dmp::finalize();
        f_thread_counter().store(0, std::memory_order_relaxed);
    }

    // if(settings::banner())
    if(_last && settings::banner())
        printf("\n\n#---------------------- tim::manager destroyed [%i][%i] "
               "----------------------#\n",
               m_rank, m_instance_count);

    delete m_lock;
}

//======================================================================================//

template <typename _Func>
inline void
manager::add_finalizer(const std::string& _key, _Func&& _func, bool _is_master)
{
    // ensure there are no duplicates
    remove_finalizer(_key);

    if(m_lock && !m_lock->owns_lock())
        m_lock->lock();

    m_metadata_fname = settings::compose_output_filename("metadata", "json");

    auto _entry = finalizer_pair_t{ _key, std::forward<_Func>(_func) };

    if(_is_master)
        m_master_finalizers.push_back(_entry);
    else
        m_worker_finalizers.push_back(_entry);
}

//======================================================================================//

inline void
manager::remove_finalizer(const std::string& _key)
{
    // if(m_is_finalizing)
    //    return;

    if(m_lock && !m_lock->owns_lock())
        m_lock->lock();

    auto _remove_finalizer = [&](finalizer_list_t& _finalizers) {
        for(auto itr = _finalizers.begin(); itr != _finalizers.end(); ++itr)
        {
            if(itr->first == _key)
            {
                _finalizers.erase(itr);
                return;
            }
        }
    };

    _remove_finalizer(m_master_finalizers);
    _remove_finalizer(m_worker_finalizers);
}

//======================================================================================//

inline void
manager::finalize()
{
    if(m_lock && !m_lock->owns_lock())
        m_lock->lock();

    m_is_finalizing = true;
    if(settings::debug())
        PRINT_HERE("%s [master: %i, worker: %i]", "finalizing",
                   (int) m_master_finalizers.size(), (int) m_worker_finalizers.size());

    auto _finalize = [](finalizer_list_t& _finalizers) {
        // reverse to delete the most recent additions first
        std::reverse(_finalizers.begin(), _finalizers.end());
        // invoke all the functions
        for(auto& itr : _finalizers)
            itr.second();
        // remove all these finalizers
        _finalizers.clear();
    };

    //
    //  ideally, only one of these will be populated
    //
    // finalize workers first
    _finalize(m_worker_finalizers);
    // finalize masters second
    _finalize(m_master_finalizers);

    m_is_finalizing = false;

    if(m_instance_count == 0)
        write_metadata("manager::finalize");

    if(settings::debug())
        PRINT_HERE("%s [master: %i, worker: %i]", "finalizing",
                   (int) m_master_finalizers.size(), (int) m_worker_finalizers.size());
}

//======================================================================================//

inline void
manager::exit_hook()
{
    if(settings::debug())
        PRINT_HERE("%s", "finalizing...");

    try
    {
        auto master_count = f_manager_instance_count().load();
        if(master_count > 0)
        {
            auto master_manager = get_shared_ptr_pair_master_instance<manager>();
            master_manager->write_metadata("manager::exit_hook");
            master_manager.reset();
        }
        else
        {
            // papi::shutdown();
            // dmp::finalize();
        }
        // papi::shutdown();
    } catch(...)
    {}

    if(settings::debug())
        PRINT_HERE("%s", "finalizing...");
}

//======================================================================================//
// metadata
//
inline void
manager::write_metadata(const char* context)
{
    if(m_rank != 0)
        return;

    static bool written = false;
    if(written)
        return;

    written          = true;
    m_metadata_fname = settings::compose_output_filename("metadata", "json");
    auto fname       = m_metadata_fname;

    if(settings::verbose() > 0 || settings::banner() || settings::debug())
        printf("\n[metadata::%s]> Outputting '%s'...\n", context, fname.c_str());

    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
    std::ofstream         ofs(fname.c_str());
    if(ofs)
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 2);
        cereal::JSONOutputArchive          oa(ofs, opts);
        oa.setNextName("timemory");
        oa.startNode();
        {
            oa.setNextName("metadata");
            oa.startNode();
            {
                oa.setNextName("settings");
                oa.startNode();
                settings::serialize_settings(oa);
                oa.finishNode();
            }
            {
                oa.setNextName("output");
                oa.startNode();
                oa(cereal::make_nvp("text", m_text_files),
                   cereal::make_nvp("json", m_json_files));
                oa.finishNode();
            }
            auto _env = env_settings::instance()->get();
            oa(cereal::make_nvp("environment", _env));
            oa.finishNode();
        }
        oa.finishNode();
    }
    if(ofs)
        ofs << std::endl;
    else
        printf("[manager]> Warning! Error opening '%s'...\n", fname.c_str());

    ofs.close();
}

//======================================================================================//
// static function
//
inline manager::comm_group_t
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

#if defined(DEBUG)
    if(settings::verbose() > 1 || settings::debug())
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
#endif

    auto local_rank = mpi::rank() / mpi::size(local_mpi_comm);
    // check
    assert(local_rank == mpi::get_node_index());

    return comm_group_t(local_mpi_comm, local_rank);
}

//======================================================================================//

}  // namespace tim

//======================================================================================//

#include "timemory/config.hpp"
#include "timemory/settings.hpp"
#include "timemory/types.hpp"
#include "timemory/utility/storage.hpp"

//======================================================================================//
//  non-template version
//
inline void
tim::settings::initialize_storage()
{
    //
    // THIS CAUSES SUPER-LONG COMPILE TIMES BECAUSE IT ALWAYS GETS INSTANTIATED
    //

    // using _Tuple = available_tuple<tim::complete_tuple_t>;
    // manager::get_storage<_Tuple>::initialize();

    throw std::runtime_error(
        "tim::settings::initialize_storage() without tuple of types has been disabled "
        "because it causes extremely long compile times!");
}

//--------------------------------------------------------------------------------------//
//  template version
//
template <typename _Tuple>
void
tim::settings::initialize_storage()
{
    manager::get_storage<_Tuple>::initialize();
}

//--------------------------------------------------------------------------------------//

inline void
tim::base::storage::free_shared_manager()
{
    if(m_manager)
        m_manager->remove_finalizer(m_label);
}

//======================================================================================//
