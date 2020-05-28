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
struct mpi_get<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    mpi_get(storage_type&, distrib_type&);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct mpi_get<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    mpi_get(storage_type&) {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
mpi_get<Type, true>::mpi_get(storage_type& data, distrib_type& results)
{
#if !defined(TIMEMORY_USE_MPI)
    if(settings::debug())
        PRINT_HERE("%s", "timemory not using MPI");

    results = distrib_type(1, data.get());
#else
    if(settings::debug())
        PRINT_HERE("%s", "timemory using MPI");

    // not yet implemented
    // auto comm =
    //    (settings::mpi_output_per_node()) ? mpi::get_node_comm() : mpi::comm_world_v;
    auto comm = mpi::comm_world_v;
    mpi::barrier(comm);

    int mpi_rank = mpi::rank(comm);
    int mpi_size = mpi::size(comm);

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [&](const result_type& src) {
        std::stringstream ss;
        {
            auto oa = policy::output_archive<cereal::MinimalJSONOutputArchive,
                                             api::native_tag>::get(ss);
            (*oa)(cereal::make_nvp("data", src));
        }
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //  Used to convert the serialization to a result
    //
    auto recv_serialize = [&](const std::string& src) {
        result_type       ret;
        std::stringstream ss;
        ss << src;
        {
            auto ia =
                policy::input_archive<cereal::JSONInputArchive, api::native_tag>::get(ss);
            (*ia)(cereal::make_nvp("data", ret));
            if(settings::debug())
                printf("[RECV: %i]> data size: %lli\n", mpi_rank,
                       (long long int) ret.size());
        }
        return ret;
    };

    //------------------------------------------------------------------------------//
    //  Calculate the total number of measurement records
    //
    auto get_num_records = [&](const auto& _inp) {
        int _sz = 0;
        for(const auto& itr : _inp)
            _sz += itr.size();
        return _sz;
    };

    results = distrib_type(mpi_size);

    auto ret     = data.get();
    auto str_ret = send_serialize(ret);

    if(mpi_rank == 0)
    {
        //
        //  The root rank receives data from all non-root ranks and reports all data
        //
        for(int i = 1; i < mpi_size; ++i)
        {
            std::string str;
            if(settings::debug())
                printf("[RECV: %i]> starting %i\n", mpi_rank, i);
            mpi::recv(str, i, 0, comm);
            if(settings::debug())
                printf("[RECV: %i]> completed %i\n", mpi_rank, i);
            results[i] = recv_serialize(str);
        }
        results[mpi_rank] = ret;
    }
    else
    {
        //
        //  The non-root rank sends its data to the root rank and only reports own data
        //
        if(settings::debug())
            printf("[SEND: %i]> starting\n", mpi_rank);
        mpi::send(str_ret, 0, 0, comm);
        if(settings::debug())
            printf("[SEND: %i]> completed\n", mpi_rank);
        results = distrib_type(1, ret);
    }

    // collapse into a single result
    if(settings::collapse_processes() && mpi_rank == 0)
    {
        auto init_size = get_num_records(results);
        if(settings::debug() || settings::verbose() > 3)
        {
            PRINT_HERE("[%s][pid=%i][rank=%i]> collapsing %i records from %i ranks",
                       demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                       mpi_rank, init_size, mpi_size);
        }

        auto _collapsed = distrib_type{};
        // so we can pop off back
        std::reverse(results.begin(), results.end());
        while(!results.empty())
        {
            if(_collapsed.empty())
                _collapsed.emplace_back(results.back());
            else
                operation::finalize::merge<Type, true>(_collapsed.front(),
                                                       results.back());
            results.pop_back();
        }

        // assign results to collapsed entry
        results = _collapsed;

        if(settings::debug() || settings::verbose() > 3)
        {
            auto fini_size = get_num_records(results);
            PRINT_HERE("[%s][pid=%i][rank=%i]> collapsed %i records into %i records "
                       "from %i ranks",
                       demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                       mpi_rank, init_size, fini_size, mpi_size);
        }
    }
    else if(settings::node_count() > 0 && mpi_rank == 0)
    {
        // calculate some size parameters
        int32_t nmod  = mpi_size % settings::node_count();
        int32_t bins  = mpi_size / settings::node_count() + ((nmod == 0) ? 0 : 1);
        int32_t bsize = mpi_size / bins;

        if(settings::debug() || settings::verbose() > 3)
            PRINT_HERE("[%s][pid=%i][rank=%i]> node_count = %i, mpi_size = %i, bins = "
                       "%i, bin size = %i",
                       demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                       mpi_rank, settings::node_count(), mpi_size, bins, bsize);

        // generate a map of the ranks to the node ids
        int32_t                              ncnt = 0;  // current count
        int32_t                              midx = 0;  // current bin map index
        std::map<int32_t, std::set<int32_t>> binmap;
        for(int32_t i = 0; i < mpi_size; ++i)
        {
            if(settings::debug())
                PRINT_HERE("[%s][pid=%i][rank=%i]> adding rank %i to bin %i",
                           demangle<mpi_get<Type, true>>().c_str(),
                           (int) process::get_id(), mpi_rank, i, midx);

            binmap[midx].insert(i);
            // check to see if we reached the bin size
            if(++ncnt == bsize)
            {
                // set counter to zero and advance the node
                ncnt = 0;
                ++midx;
            }
        }

        auto init_size = get_num_records(results);
        if(settings::debug() || settings::verbose() > 3)
            PRINT_HERE(
                "[%s][pid=%i][rank=%i]> collapsing %i records from %i ranks into %i bins",
                demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                mpi_rank, init_size, mpi_size, (int) binmap.size());

        assert(binmap.size() <= settings::node_count());

        // the collapsed data
        auto _collapsed = distrib_type(binmap.size());
        // loop over the node indexes
        for(const auto& itr : binmap)
        {
            // target the node index
            auto& _dst = _collapsed.at(itr.first);
            for(const auto& bitr : itr.second)
            {
                // combine the node index entry with all of the ranks in that node
                auto& _src = results.at(bitr);
                operation::finalize::merge<Type, true>(_dst, _src);
            }
        }

        // assign results to collapsed entry
        results = _collapsed;

        if(settings::debug() || settings::verbose() > 3)
        {
            auto fini_size = get_num_records(results);
            PRINT_HERE("[%s][pid=%i][rank=%i]> collapsed %i records into %i records "
                       "and %i bins",
                       demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                       mpi_rank, init_size, fini_size, (int) results.size());
        }
    }

    if(settings::debug() || settings::verbose() > 1)
    {
        auto ret_size = get_num_records(results);
        PRINT_HERE("[%s][pid=%i]> %i total records on rank %i of %i",
                   demangle<mpi_get<Type, true>>().c_str(), (int) process::get_id(),
                   ret_size, mpi_rank, mpi_size);
    }

#endif
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
