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

/**
 * \file timemory/operations/types/finalize_get.hpp
 * \brief Definition for various functions for finalize_get in operations
 */

#pragma once

//======================================================================================//
//
#include "timemory/operations/macros.hpp"
//
#include "timemory/operations/types.hpp"
//
#include "timemory/operations/declaration.hpp"
//
//======================================================================================//

namespace tim
{
namespace operation
{
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct dmp_get<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    dmp_get(storage_type&, distrib_type&);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct dmp_get<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    dmp_get(storage_type&) {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
dmp_get<Type, true>::dmp_get(storage_type& data, distrib_type& results)
{
    auto fallback_get = [&]() { return distrib_type(1, data.get()); };

#if defined(TIMEMORY_USE_UPCXX) && defined(TIMEMORY_USE_MPI)
    results = (mpi::is_initialized())
                  ? data.mpi_get()
                  : ((upc::is_initialized()) ? data.upc_get() : fallback_get());
#elif defined(TIMEMORY_USE_UPCXX)
    results = (upc::is_initialized()) ? data.upc_get() : fallback_get();
#elif defined(TIMEMORY_USE_MPI)
    results = (mpi::is_initialized()) ? data.mpi_get() : fallback_get();
#else
    results = fallback_get();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
