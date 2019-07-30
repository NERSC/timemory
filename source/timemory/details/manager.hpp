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

/** \file manager.hpp
 * \headerfile manager.hpp "timemory/details/manager.hpp"
 * Provides inline implementation of manager functions
 *
 */

#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#    define TIMEMORY_DEFAULT_ENABLED true
#endif

//======================================================================================//

namespace tim
{
//======================================================================================//
#if !defined(TIMEMORY_EXTERN_INIT)
inline std::atomic<int32_t>&
manager::f_manager_instance_count()
{
    static std::atomic<int32_t> instance;
    return instance;
}

//======================================================================================//
// get either master or thread-local instance
//
inline manager::pointer
manager::instance()
{
    return details::manager_singleton().instance();
}

//======================================================================================//
// get master instance
//
inline manager::pointer
manager::master_instance()
{
    return details::manager_singleton().master_instance();
}

//======================================================================================//
// static function
inline manager::pointer
manager::noninit_instance()
{
    return details::manager_singleton().instance_ptr();
}

//======================================================================================//
// static function
inline manager::pointer
manager::noninit_master_instance()
{
    return details::manager_singleton().master_instance_ptr();
}
#endif
//======================================================================================//

inline manager::manager()
: m_instance_count(f_manager_instance_count()++)
{
    f_thread_counter()++;
    static std::atomic<int> _once(0);

    if(_once++ == 0)
    {
        tim::papi::init();
        tim::cupti::initialize();
        tim::settings::parse();
        std::atexit(&exit_hook);
    }
    else
    {
        if(m_instance_count == 0)
            tim::papi::register_thread();
    }

    if(m_instance_count == 0)
    {
        if(get_env("TIMEMORY_BANNER", true))
            printf(
                "#--------------------- tim::manager initialized [%i] "
                "---------------------#\n\n",
                m_instance_count);
    }

    if(singleton_t::master_instance_ptr() && singleton_t::instance_ptr())
    {
        std::ostringstream errss;
        errss << "manager singleton has already been created";
        throw std::runtime_error(errss.str().c_str());
    }
}

//======================================================================================//

inline manager::~manager()
{
    if(m_instance_count > 0)
    {
        tim::papi::unregister_thread();
        f_thread_counter().store(0, std::memory_order_relaxed);
    }

    --f_manager_instance_count();
}

//======================================================================================//

inline void
manager::insert(const int64_t& _hash_id, const string_t& _prefix, const string_t& _data)
{
    using sibling_itr = typename graph_t::sibling_iterator;
    graph_node node(_hash_id, _prefix, _data);

    auto _update = [&](iterator itr) {
        m_data.current() = itr;
        *m_data.current() += node;
    };

    // lambda for inserting child
    auto _insert_child = [&]() {
        auto itr = m_data.append_child(node);
        m_node_ids.insert(std::make_pair(_hash_id, itr));
    };

    if(m_node_ids.find(_hash_id) != m_node_ids.end())
    {
        _update(m_node_ids.find(_hash_id)->second);
    }

    // if first instance
    if(m_data.depth() < 0)
    {
        if(this == master_instance())
        {
            m_data.depth()   = 0;
            m_data.head()    = m_data.graph().set_head(node);
            m_data.current() = m_data.head();
        }
        else
        {
            return;
        }
    }
    else
    {
        auto current = m_data.current();

        if(_hash_id == current->id())
        {
            return;
        }
        else if(m_data.graph().is_valid(current))
        {
            // check parent if not head
            if(!m_data.graph().is_head(current))
            {
                auto parent = graph_t::parent(current);
                for(sibling_itr itr = parent.begin(); itr != parent.end(); ++itr)
                {
                    // check hash id's
                    if(_hash_id == itr->id())
                    {
                        _update(itr);
                    }
                }
            }

            // check siblings
            for(sibling_itr itr = current.begin(); itr != current.end(); ++itr)
            {
                // skip if current
                if(itr == current)
                    continue;
                // check hash id's
                if(_hash_id == itr->id())
                {
                    _update(itr);
                }
            }

            // check children
            auto nchildren = graph_t::number_of_children(current);
            if(nchildren == 0)
            {
                _insert_child();
            }
            else
            {
                bool exists = false;
                auto fchild = graph_t::child(current, 0);
                for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
                {
                    if(_hash_id == itr->id())
                    {
                        exists = true;
                        _update(itr);
                        break;
                    }
                }
                if(!exists)
                    _insert_child();
            }
        }
    }
    return _insert_child();
}

//======================================================================================//
/*
inline void
manager::clear()
{
#if defined(DEBUG)
    if(tim::settings::verbose() > 1)
        std::cout << "tim::manager::" << __FUNCTION__ << " Clearing " << instance()
                  << "..." << std::endl;
#endif

    // if(this == singleton_t::master_instance_ptr())
    // tim::format::timer::default_width(8);

    // m_laps += compute_total_laps();
    // timer_data.graph().clear();
    // timer_data.current() = nullptr;
    // timer_data.map().clear();
    for(auto& itr : m_daughters)
        if(itr != this && itr)
            itr->clear();

    ofstream_t* m_fos = get_ofstream(m_report);

    if(m_fos)
    {
        for(int32_t i = 0; i < mpi_size(); ++i)
        {
            mpi_barrier(MPI_COMM_WORLD);
            if(mpi_rank() != i)
                continue;

            if(m_fos->good() && m_fos->is_open())
            {
                if(mpi_rank() + 1 >= mpi_size())
                {
                    m_fos->flush();
                    m_fos->close();
                    delete m_fos;
                }
                else
                {
                    m_fos->flush();
                    m_fos->close();
                    delete m_fos;
                }
            }
        }
    }

    m_report = &std::cout;

}
*/

//======================================================================================//
/*
inline void
manager::report(bool ign_cutoff, bool endline) const
{
    const_cast<this_type*>(this)->merge();

    int32_t _default = (mpi_is_initialized()) ? 1 : 0;
    int32_t _verbose = tim::get_env<int32_t>("TIMEMORY_VERBOSE", _default);

    if(mpi_rank() == 0 && _verbose > 0)
    {
        std::stringstream _info;
        if(mpi_is_initialized())
            _info << "[" << mpi_rank() << "] ";
        _info << "Reporting timing output..." << std::endl;
        std::cout << _info.str();
    }

    int nitr = std::max(mpi_size(), 1);
    for(int32_t i = 0; i < nitr; ++i)
    {
        // MPI blocking
        if(mpi_is_initialized())
        {
            mpi_barrier(MPI_COMM_WORLD);
            // only 1 at a time
            if(i != mpi_rank())
                continue;
        }
        report(m_report, ign_cutoff, endline);
    }
}
*/

//======================================================================================//
/*
inline void
manager::set_output_stream(const path_t& fname)
{
    if(fname.find(fname.os()) != std::string::npos)
        tim::makedir(fname.substr(0, fname.find_last_of(fname.os())));

    auto ostreamop = [&](ostream_t*& m_os, const string_t& _fname) {
        if(m_os != &std::cout)
            delete(ofstream_t*) m_os;

        ofstream_t* _fos = new ofstream_t;
        for(int32_t i = 0; i < mpi_size(); ++i)
        {
            mpi_barrier(MPI_COMM_WORLD);
            if(mpi_rank() != i)
                continue;

            if(mpi_rank() == 0)
                _fos->open(_fname);
            else
                _fos->open(_fname, std::ios_base::out | std::ios_base::app);
        }

        if(_fos->is_open() && _fos->good())
            m_os = _fos;
        else
        {
#if defined(DEBUG)
            if(tim::settings::verbose() > 2)
            {
                tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
                std::cerr << "Warning! Unable to open file " << _fname << ". "
                          << "Redirecting to stdout..." << std::endl;
            }
#endif
            _fos->close();
            delete _fos;
            m_os = &std::cout;
        }
    };

    ostreamop(m_report, fname);
}
*/

//======================================================================================//

inline void
manager::merge(pointer itr)
{
    if(itr == this)
        return;

    // create lock but don't immediately lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);

    // lock if not already owned
    if(!l.owns_lock())
        l.lock();

    auto _this_beg = graph().begin();
    auto _this_end = graph().end();

    bool _merged = false;
    for(auto _this_itr = _this_beg; _this_itr != _this_end; ++_this_itr)
    {
        if(_this_itr == itr->data().head())
        {
            auto _iter_beg = itr->graph().begin();
            auto _iter_end = itr->graph().end();
            graph().merge(_this_itr, _this_end, _iter_beg, _iter_end, false, true);
            _merged = true;
            break;
        }
    }

    if(_merged)
    {
        _this_beg = graph().begin();
        _this_end = graph().end();
        graph().reduce(_this_beg, _this_end, _this_beg, _this_end);
    }
    else
    {
        auto_lock_t lerr(type_mutex<decltype(std::cerr)>());
        std::cerr << "Failure to merge graphs!" << std::endl;
        auto g = graph();
        graph().insert_subgraph_after(m_data.current(), itr->data().head());
    }
}

//======================================================================================//
// static function
inline manager::comm_group_t
manager::get_communicator_group()
{
    int32_t max_concurrency = std::thread::hardware_concurrency();
    // We want on-node communication only
    int32_t nthreads         = f_thread_counter().load();
    int32_t max_processes    = max_concurrency / nthreads;
    int32_t mpi_node_default = mpi_size() / max_processes;
    if(mpi_node_default < 1)
        mpi_node_default = 1;
    int32_t mpi_node_count =
        tim::get_env<int32_t>("TIMEMORY_NODE_COUNT", mpi_node_default);
    int32_t mpi_split_size = mpi_rank() / (mpi_size() / mpi_node_count);

    // Split the communicator based on the number of nodes and use the
    // original rank for ordering
    MPI_Comm local_mpi_comm;
    MPI_Comm_split(MPI_COMM_WORLD, mpi_split_size, mpi_rank(), &local_mpi_comm);

#if defined(DEBUG)
    if(tim::settings::verbose() > 1)
    {
        int32_t local_mpi_rank = mpi_rank(local_mpi_comm);
        int32_t local_mpi_size = mpi_size(local_mpi_comm);
        int32_t local_mpi_file = mpi_rank() / local_mpi_size;

        std::stringstream _info;
        _info << "\t" << mpi_rank() << " Rank      : " << mpi_rank() << std::endl;
        _info << "\t" << mpi_rank() << " Size      : " << mpi_size() << std::endl;
        _info << "\t" << mpi_rank() << " Node      : " << mpi_node_count << std::endl;
        _info << "\t" << mpi_rank() << " Local Size: " << local_mpi_size << std::endl;
        _info << "\t" << mpi_rank() << " Local Rank: " << local_mpi_rank << std::endl;
        _info << "\t" << mpi_rank() << " Local File: " << local_mpi_file << std::endl;
        std::cout << "tim::manager::" << __FUNCTION__ << "\n" << _info.str();
    }
#endif

    return comm_group_t(local_mpi_comm, mpi_rank() / mpi_size(local_mpi_comm));
}

//======================================================================================//

}  // namespace tim

//======================================================================================//

#include "timemory/settings.hpp"
#include "timemory/utility/storage.hpp"
#include "timemory/variadic/component_tuple.hpp"

//======================================================================================//

template <typename Head, typename... Tail>
inline void
tim::manager::print(const tim::component_tuple<Head, Tail...>&)
{
    auto storage = tim::storage<Head>::instance();
    if(storage && !storage->empty())
        storage->print();
    using tail_obj_t = PopFront<tim::component_tuple<Head, Tail...>>;
    print(tail_obj_t());
}

//--------------------------------------------------------------------------------------//

inline void
tim::manager::print(bool /*ign_cutoff*/, bool /*endline*/)
{
}

//======================================================================================//
