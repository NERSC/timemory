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
#include "timemory/backends/threading.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/serializer.hpp"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace node
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct data
{
    using string_t     = std::string;
    using strvector_t  = std::vector<string_t>;
    using uintvector_t = std::vector<uint64_t>;

    using type         = typename trait::statistics<Tp>::type;
    using stats_policy = policy::record_statistics<Tp, type>;
    using stats_type   = typename stats_policy::statistics_type;
    using node_type   = std::tuple<uint64_t, Tp, int64_t, stats_type, uint16_t, uint16_t>;
    using result_type = std::tuple<uint64_t, Tp, string_t, int64_t, uint64_t,
                                   uintvector_t, stats_type, uint16_t, uint16_t>;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct graph : public data<Tp>::node_type
{
    using this_type       = graph;
    using base_type       = typename data<Tp>::node_type;
    using data_value_type = typename Tp::value_type;
    using data_base_type  = typename Tp::base_type;
    using stats_type      = typename data<Tp>::stats_type;
    using string_t        = std::string;

    uint64_t&   id() { return std::get<0>(*this); }
    Tp&         obj() { return std::get<1>(*this); }
    int64_t&    depth() { return std::get<2>(*this); }
    stats_type& stats() { return std::get<3>(*this); }
    uint16_t&   tid() { return std::get<4>(*this); }
    uint16_t&   pid() { return std::get<5>(*this); }

    const uint64_t&   id() const { return std::get<0>(*this); }
    const Tp&         obj() const { return std::get<1>(*this); }
    const int64_t&    depth() const { return std::get<2>(*this); }
    const stats_type& stats() const { return std::get<3>(*this); }
    const uint16_t&   tid() const { return std::get<4>(*this); }
    const uint16_t&   pid() const { return std::get<5>(*this); }

    graph();
    explicit graph(base_type&& _base);
    graph(uint64_t _id, const Tp& _obj, int64_t _depth, uint16_t _tid,
          uint16_t _pid = process::get_id());
    ~graph() = default;

    bool      operator==(const graph& rhs) const;
    bool      operator!=(const graph& rhs) const;
    static Tp get_dummy();
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct result : public data<Tp>::result_type
{
    using uintvector_t = std::vector<uint64_t>;
    using base_type    = typename data<Tp>::result_type;
    using stats_type   = typename data<Tp>::stats_type;
    using this_type    = result<Tp>;

    result()              = default;
    ~result()             = default;
    result(const result&) = default;
    result(result&&)      = default;
    result& operator=(const result&) = default;
    result& operator=(result&&) = default;

    result(base_type&& _base)
    : base_type(std::forward<base_type>(_base))
    {}

    result(uint64_t _hash, const Tp& _data, const string_t& _prefix, int64_t _depth,
           uint64_t _rolling, const uintvector_t& _hierarchy, const stats_type& _stats,
           uint16_t _tid, uint16_t _pid);

    uint64_t&     hash() { return std::get<0>(*this); }
    Tp&           data() { return std::get<1>(*this); }
    string_t&     prefix() { return std::get<2>(*this); }
    int64_t&      depth() { return std::get<3>(*this); }
    uint64_t&     rolling_hash() { return std::get<4>(*this); }
    uintvector_t& hierarchy() { return std::get<5>(*this); }
    stats_type&   stats() { return std::get<6>(*this); }
    uint16_t&     tid() { return std::get<7>(*this); }
    uint16_t&     pid() { return std::get<8>(*this); }

    const uint64_t&     hash() const { return std::get<0>(*this); }
    const Tp&           data() const { return std::get<1>(*this); }
    const string_t&     prefix() const { return std::get<2>(*this); }
    const int64_t&      depth() const { return std::get<3>(*this); }
    const uint64_t&     rolling_hash() const { return std::get<4>(*this); }
    const uintvector_t& hierarchy() const { return std::get<5>(*this); }
    const stats_type&   stats() const { return std::get<6>(*this); }
    const uint16_t&     tid() const { return std::get<7>(*this); }
    const uint16_t&     pid() const { return std::get<8>(*this); }

    uint64_t&       id() { return std::get<0>(*this); }
    const uint64_t& id() const { return std::get<0>(*this); }

    Tp&       obj() { return std::get<1>(*this); }
    const Tp& obj() const { return std::get<1>(*this); }

    bool operator==(const this_type& rhs) const
    {
        return (hash() == rhs.hash() && prefix() == rhs.prefix() &&
                depth() == rhs.depth() && rolling_hash() == rhs.rolling_hash());
    }

    bool operator!=(const this_type& rhs) const { return !(*this == rhs); }

    this_type& operator-=(const this_type& rhs)
    {
        data() -= rhs.data();
        stats() -= rhs.stats();
        return *this;
    }
};
//
//--------------------------------------------------------------------------------------//
//
//                              Definitions
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
graph<Tp>::graph()
: base_type(0, Tp{}, 0, stats_type{}, threading::get_id(), process::get_id())
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
graph<Tp>::graph(base_type&& _base)
: base_type(std::forward<base_type>(_base))
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
graph<Tp>::graph(uint64_t _id, const Tp& _obj, int64_t _depth, uint16_t _tid,
                 uint16_t _pid)
: base_type(_id, _obj, _depth, stats_type{}, _tid, _pid)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
bool
graph<Tp>::operator==(const graph& rhs) const
{
    return (id() == rhs.id() && depth() == rhs.depth());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
bool
graph<Tp>::operator!=(const graph& rhs) const
{
    return !(*this == rhs);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
graph<Tp>::get_dummy()
{
    using object_base = typename Tp::base_type;
    return object_base::dummy();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
result<Tp>::result(uint64_t _hash, const Tp& _data, const string_t& _prefix,
                   int64_t _depth, uint64_t _rolling, const uintvector_t& _hierarchy,
                   const stats_type& _stats, uint16_t _tid, uint16_t _pid)
: base_type(_hash, _data, _prefix, _depth, _rolling, _hierarchy, _stats, _tid, _pid)
{}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace node
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
namespace cereal
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const tim::node::result<Tp>& r)
{
    // clang-format off
    ar(cereal::make_nvp("hash", r.hash()),
       cereal::make_nvp("prefix", r.prefix()),
       cereal::make_nvp("depth", r.depth()),
       cereal::make_nvp("entry", r.data()),
       cereal::make_nvp("stats", r.stats()),
       cereal::make_nvp("rolling_hash", r.rolling_hash()));
    // clang-format on
    // ar(cereal::make_nvp("hierarchy", r.hierarchy()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, tim::node::result<Tp>& r)
{
    // clang-format off
    ar(cereal::make_nvp("hash", r.hash()),
       cereal::make_nvp("prefix", r.prefix()),
       cereal::make_nvp("depth", r.depth()),
       cereal::make_nvp("entry", r.data()),
       cereal::make_nvp("stats", r.stats()),
       cereal::make_nvp("rolling_hash", r.rolling_hash()));
    // clang-format on
    // ar(cereal::make_nvp("hierarchy", r.hierarchy()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const std::vector<tim::node::result<Tp>>& result_nodes)
{
    ar(cereal::make_nvp("graph_size", result_nodes.size()));
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(const auto& itr : result_nodes)
    {
        ar.startNode();
        save(ar, itr);
        ar.finishNode();
    }
    ar.finishNode();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, std::vector<tim::node::result<Tp>>& result_nodes)
{
    size_t nnodes = 0;
    ar(cereal::make_nvp("graph_size", nnodes));
    result_nodes.resize(nnodes, tim::node::result<Tp>{});

    ar.setNextName("graph");
    ar.startNode();
    for(auto& itr : result_nodes)
    {
        ar.startNode();
        load(ar, itr);
        ar.finishNode();
    }
    ar.finishNode();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
struct specialize<Archive, tim::node::result<Tp>,
                  cereal::specialization::non_member_load_save>
{};
//
}  // namespace cereal
//
//--------------------------------------------------------------------------------------//
