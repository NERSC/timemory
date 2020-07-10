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

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

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
struct upc_get<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    upc_get(storage_type&, distrib_type&);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct upc_get<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    upc_get(storage_type&) {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
upc_get<Type, true>::upc_get(storage_type& data, distrib_type& results)
{
#if !defined(TIMEMORY_USE_UPCXX)
    if(settings::debug())
        PRINT_HERE("%s", "timemory not using UPC++");

    results = distrib_type(1, data.get());
#else
    if(settings::debug())
        PRINT_HERE("%s", "timemory using UPC++");

    upc::barrier();

    int comm_rank = upc::rank();
    int comm_size = upc::size();

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [=](const result_type& src) {
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
    auto recv_serialize = [=](const std::string& src) {
        result_type       ret;
        std::stringstream ss;
        ss << src;
        {
            auto ia =
                policy::input_archive<cereal::JSONInputArchive, api::native_tag>::get(ss);
            (*ia)(cereal::make_nvp("data", ret));
        }
        return ret;
    };

    //------------------------------------------------------------------------------//
    //  Function executed on remote node
    //
    auto remote_serialize = [=]() {
        return send_serialize(storage_type::master_instance()->get());
    };

    results.resize(comm_size);

    //------------------------------------------------------------------------------//
    //  Combine on master rank
    //
    if(comm_rank == 0)
    {
        for(int i = 1; i < comm_size; ++i)
        {
            upcxx::future<std::string> fut = upcxx::rpc(i, remote_serialize);
            while(!fut.ready())
                upcxx::progress();
            fut.wait();
            results[i] = recv_serialize(fut.result());
        }
        results[comm_rank] = data.get();
    }

    upcxx::barrier(upcxx::world());

    if(comm_rank != 0)
        results = distrib_type(1, data.get());

    // collapse into a single result
    if(settings::collapse_processes() && comm_rank == 0)
    {
        auto init_size = get_num_records(results);
        if(settings::debug() || settings::verbose() > 3)
        {
            PRINT_HERE("[%s][pid=%i][rank=%i]> collapsing %i records from %i ranks",
                       demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                       comm_rank, init_size, comm_size);
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
                       demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                       comm_rank, init_size, fini_size, comm_size);
        }
    }
    else if(settings::node_count() > 0 && comm_rank == 0)
    {
        // calculate some size parameters
        int32_t nmod  = comm_size % settings::node_count();
        int32_t bins  = comm_size / settings::node_count() + ((nmod == 0) ? 0 : 1);
        int32_t bsize = comm_size / bins;

        if(settings::debug() || settings::verbose() > 3)
            PRINT_HERE("[%s][pid=%i][rank=%i]> node_count = %i, comm_size = %i, bins = "
                       "%i, bin size = %i",
                       demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                       comm_rank, settings::node_count(), comm_size, bins, bsize);

        // generate a map of the ranks to the node ids
        int32_t                              ncnt = 0;  // current count
        int32_t                              midx = 0;  // current bin map index
        std::map<int32_t, std::set<int32_t>> binmap;
        for(int32_t i = 0; i < comm_size; ++i)
        {
            if(settings::debug())
                PRINT_HERE("[%s][pid=%i][rank=%i]> adding rank %i to bin %i",
                           demangle<upc_get<Type, true>>().c_str(),
                           (int) process::get_id(), comm_rank, i, midx);

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
                demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                comm_rank, init_size, comm_size, (int) binmap.size());

        assert((int32_t) binmap.size() <= (int32_t) settings::node_count());

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
                       demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                       comm_rank, init_size, fini_size, (int) results.size());
        }
    }

    if(settings::debug() || settings::verbose() > 1)
    {
        auto ret_size = get_num_records(results);
        PRINT_HERE("[%s][pid=%i]> %i total records on rank %i of %i",
                   demangle<upc_get<Type, true>>().c_str(), (int) process::get_id(),
                   ret_size, comm_rank, comm_size);
    }

#endif
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
