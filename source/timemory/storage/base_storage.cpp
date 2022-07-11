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

#ifndef TIMEMORY_STORAGE_BASE_STORAGE_CPP_
#define TIMEMORY_STORAGE_BASE_STORAGE_CPP_ 1

#include "timemory/storage/macros.hpp"

#if !defined(TIMEMORY_STORAGE_HIDE_DEFINITION)

#    include "timemory/backends/gperftools.hpp"
#    include "timemory/hash/types.hpp"
#    include "timemory/manager/manager.hpp"
#    include "timemory/storage/base_storage.hpp"

#    include <atomic>
#    include <cstdint>
#    include <functional>
#    include <string>

namespace tim
{
namespace base
{
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_INLINE
storage::storage(bool _is_master, int64_t _instance_id, std::string _label)
: m_is_master(_is_master)
, m_instance_id(_instance_id)
, m_label(std::move(_label))
, m_manager(::tim::manager::instance())
, m_settings(::tim::settings::shared_instance())
{
    if(m_is_master && m_instance_id > 0)
    {
        int _id = m_instance_id;
        TIMEMORY_PRINT_HERE("%s: %i (%s)",
                            "Error! base::storage is master but is not zero instance",
                            _id, m_label.c_str());
        if(m_instance_id > 10)
        {
            // at this point we have a recursive loop
            TIMEMORY_EXCEPTION("Duplication!")
        }
    }

    if(!m_is_master && m_instance_id == 0)
    {
        int _id = m_instance_id;
        TIMEMORY_PRINT_HERE("%s: %i (%s)",
                            "Warning! base::storage is not master but is zero instance",
                            _id, m_label.c_str());
    }

    TIMEMORY_CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "%s: %i (%s)",
                                    "base::storage instance created", (int) m_instance_id,
                                    m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_INLINE storage::~storage()
{
    TIMEMORY_CONDITIONAL_PRINT_HERE(m_settings->get_debug(),
                                    "base::storage instance %i deleted for %s",
                                    (int) m_instance_id, m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_INLINE std::atomic<int>&
                        storage::storage_once_flag()
{
    static std::atomic<int> _instance(0);
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_INLINE void
storage::stop_profiler()
{
    // disable gperf if profiling
#    if defined(TIMEMORY_USE_GPERFTOOLS) || defined(TIMEMORY_USE_GPERFTOOLS_PROFILER) || \
        defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
    try
    {
        if(storage_once_flag()++ == 0)
            gperftools::profiler_stop();
    } catch(std::exception& e)
    {
        std::cerr << "Error calling gperftools::profiler_stop(): " << e.what()
                  << ". Continuing..." << std::endl;
    }
#    endif
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_INLINE void
storage::add_hash_id(uint64_t _lhs, uint64_t _rhs)
{
    ::tim::add_hash_id(m_hash_aliases, _lhs, _rhs);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_INLINE hash_value_t
storage::add_hash_id(const std::string& _prefix)
{
    return ::tim::add_hash_id(m_hash_ids, _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_INLINE void
storage::add_file_output(const std::string& _category, const std::string& _label,
                         const std::string& _file)
{
    if(m_manager)
        m_manager->add_file_output(_category, _label, _file);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_INLINE void
storage::free_shared_manager()
{
    if(m_manager)
        m_manager->remove_finalizer(m_label);
}
}  // namespace base
}  // namespace tim

#endif  // defined(TIMEMORY_STORAGE_HIDE_DEFINITION)
#endif
