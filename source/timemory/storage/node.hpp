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
#include "timemory/data/statistics.hpp"
#include "timemory/hash.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/archives.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

//--------------------------------------------------------------------------------------//
//
namespace tim
{
//
namespace operation
{
template <typename Tp>
struct dummy;
}
//
namespace node
{
//
//--------------------------------------------------------------------------------------//
/// \struct tim::node::entry
/// \tparam Tp Component type
/// \tparam StatT Statistics type
///
/// \brief This data type is used in \ref tim::node::tree for inclusive and exclusive
/// values.
template <typename Tp, typename StatT>
struct entry : std::tuple<Tp, StatT>
{
    using base_type = std::tuple<Tp, StatT>;
    using this_type = entry;

    TIMEMORY_DEFAULT_OBJECT(entry)

    template <typename... Args>
    explicit entry(Args&&... args)
    : base_type(std::forward<Args>(args)...)
    {}
    explicit entry(const base_type& _obj)
    : base_type(_obj)
    {}
    explicit entry(base_type&& _obj) noexcept
    : base_type(std::forward<base_type>(_obj))
    {}

    /// component object with either inclusive or exclusive values
    Tp& data() { return std::get<0>(*this); }

    /// statistics data with either inclusive or exclusive values
    StatT& stats() { return std::get<1>(*this); }

    /// component object with either inclusive or exclusive values
    const Tp& data() const { return std::get<0>(*this); }

    /// statistics data with either inclusive or exclusive values
    const StatT& stats() const { return std::get<1>(*this); }

    this_type& operator+=(const this_type& rhs)
    {
        data() += rhs.data();
        stats() += rhs.stats();
        return *this;
    }

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
template <typename Tp>
struct data
{
    using string_t     = std::string;
    using strvector_t  = std::vector<string_t>;
    using uintvector_t = std::vector<uint64_t>;

    using type         = typename trait::statistics<Tp>::type;
    using stats_policy = policy::record_statistics<Tp, type>;
    using stats_type   = typename stats_policy::statistics_type;
    using node_type =
        std::tuple<bool, uint32_t, uint32_t, uint64_t, int64_t, Tp, stats_type>;
    using result_type = std::tuple<uint32_t, uint32_t, int64_t, uint64_t, uint64_t,
                                   string_t, uintvector_t, Tp, stats_type>;
    using idset_type  = std::set<int64_t>;
    using entry_type  = entry<Tp, stats_type>;
    using tree_type   = std::tuple<bool, uint64_t, int64_t, idset_type, idset_type,
                                 entry_type, entry_type>;
};
//
//--------------------------------------------------------------------------------------//
/// \struct tim::node::graph
/// \tparam Tp Component type
///
/// \brief This is the compact representation of a measurement in the call-graph.
template <typename Tp>
struct graph : private data<Tp>::node_type
{
    using this_type       = graph;
    using base_type       = typename data<Tp>::node_type;
    using data_value_type = typename Tp::value_type;
    using data_base_type  = typename Tp::base_type;
    using stats_type      = typename data<Tp>::stats_type;
    using string_t        = std::string;

public:
    // ctor, dtor
    graph();
    graph(uint64_t _id, const Tp& _obj, int64_t _depth, uint32_t _tid,
          uint32_t _pid = process::get_id(), bool _is_dummy = false);

    ~graph()                = default;
    graph(const graph&)     = default;
    graph(graph&&) noexcept = default;

    graph& operator=(const graph&) = default;
    graph& operator=(graph&&) noexcept = default;

public:
    static Tp  get_dummy();
    bool       operator==(const graph& rhs) const;
    bool       operator!=(const graph& rhs) const;
    this_type& operator+=(const this_type& rhs)
    {
        obj() += rhs.obj();
        stats() += rhs.stats();
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        obj() -= rhs.obj();
        stats() -= rhs.stats();
        return *this;
    }

public:
    /// denotes this is a placeholder for synchronization
    bool& is_dummy() { return std::get<0>(*this); }

    /// thread identifier
    uint32_t& tid() { return std::get<1>(*this); }

    /// process identifier
    uint32_t& pid() { return std::get<2>(*this); }

    /// hash identifer
    uint64_t& id() { return std::get<3>(*this); }

    /// depth in call-graph
    int64_t& depth() { return std::get<4>(*this); }

    /// this is the instance that gets updated in call-graph
    Tp& obj() { return std::get<5>(*this); }

    /// statistics data for entry in call-graph
    stats_type& stats() { return std::get<6>(*this); }

    const bool&       is_dummy() const { return std::get<0>(*this); }
    const uint32_t&   tid() const { return std::get<1>(*this); }
    const uint32_t&   pid() const { return std::get<2>(*this); }
    const uint64_t&   id() const { return std::get<3>(*this); }
    const int64_t&    depth() const { return std::get<4>(*this); }
    const Tp&         obj() const { return std::get<5>(*this); }
    const stats_type& stats() const { return std::get<6>(*this); }

    auto&       data() { return this->obj(); }
    auto&       hash() { return this->id(); }
    const auto& data() const { return this->obj(); }
    const auto& hash() const { return this->id(); }
};
//
//--------------------------------------------------------------------------------------//
/// \struct tim::node::result
/// \tparam Tp Component type
///
/// \brief This data type is used when rendering the flat representation (i.e.
/// loop-iterable) representation of the calling-context. The prefix here will be
/// identical to the prefix in the text output.
template <typename Tp>
struct result : public data<Tp>::result_type
{
    using uintvector_t = std::vector<uint64_t>;
    using base_type    = typename data<Tp>::result_type;
    using stats_type   = typename data<Tp>::stats_type;
    using this_type    = result<Tp>;

    result()                  = default;
    ~result()                 = default;
    result(const result&)     = default;
    result(result&&) noexcept = default;
    result& operator=(const result&) = default;
    result& operator=(result&&) noexcept = default;

    result(base_type&& _base) noexcept
    : base_type(std::forward<base_type>(_base))
    {}

    result(uint64_t _hash, const Tp& _data, const string_t& _prefix, int64_t _depth,
           uint64_t _rolling, const uintvector_t& _hierarchy, const stats_type& _stats,
           uint32_t _tid, uint32_t _pid);

    /// measurement thread. May be `std::numeric_limits<uint16_t>::max()` (i.e. 65536) if
    /// this entry is a combination of multiple threads
    uint32_t& tid() { return std::get<0>(*this); }

    /// the process identifier of the reporting process, if multiple process data is
    /// combined, or the process identifier of the collecting process
    uint32_t& pid() { return std::get<1>(*this); }

    /// depth of the node in the calling-context
    int64_t& depth() { return std::get<2>(*this); }

    /// hash identifer of the node
    uint64_t& hash() { return std::get<3>(*this); }

    /// the summation of this hash and it's parent hashes
    uint64_t& rolling_hash() { return std::get<4>(*this); }

    /// the associated string with the hash + indentation and other decoration
    string_t& prefix() { return std::get<5>(*this); }

    /// an array of the hash value + each parent hash (not serialized)
    uintvector_t& hierarchy() { return std::get<6>(*this); }

    /// reference to the component
    Tp& data() { return std::get<7>(*this); }

    /// reference to the associate statistical accumulation of the data (if any)
    stats_type& stats() { return std::get<8>(*this); }

    /// alias for `hash()`
    uint64_t& id() { return std::get<3>(*this); }

    /// alias for `data()`
    Tp& obj() { return std::get<7>(*this); }

    const uint32_t&     tid() const { return std::get<0>(*this); }
    const uint32_t&     pid() const { return std::get<1>(*this); }
    const int64_t&      depth() const { return std::get<2>(*this); }
    const uint64_t&     hash() const { return std::get<3>(*this); }
    const uint64_t&     rolling_hash() const { return std::get<4>(*this); }
    const string_t&     prefix() const { return std::get<5>(*this); }
    const uintvector_t& hierarchy() const { return std::get<6>(*this); }
    const Tp&           data() const { return std::get<7>(*this); }
    const stats_type&   stats() const { return std::get<8>(*this); }
    const uint64_t&     id() const { return std::get<3>(*this); }
    const Tp&           obj() const { return std::get<7>(*this); }

    bool operator==(const this_type& rhs) const
    {
        return (depth() == rhs.depth() && hash() == rhs.hash() &&
                rolling_hash() == rhs.rolling_hash() && prefix() == rhs.prefix());
    }

    bool operator!=(const this_type& rhs) const { return !(*this == rhs); }

    this_type& operator+=(const this_type& rhs)
    {
        data() += rhs.data();
        stats() += rhs.stats();
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        data() -= rhs.data();
        stats() -= rhs.stats();
        return *this;
    }
};
//
//--------------------------------------------------------------------------------------//
/// \struct tim::node::tree
/// \tparam Tp Generally `tim::basic_tree<ComponentT>`
///
/// \brief This data type is used when rendering the hierarchical representation (i.e.
/// requires recursion) representation of the calling-context. The prefix here has no
/// decoration.
template <typename Tp>
struct tree : private data<Tp>::tree_type
{
    using this_type       = tree;
    using base_type       = typename data<Tp>::tree_type;
    using data_value_type = typename Tp::value_type;
    using data_base_type  = typename Tp::base_type;
    using stats_type      = typename data<Tp>::stats_type;
    using entry_type      = typename data<Tp>::entry_type;
    using idset_type      = typename data<Tp>::idset_type;
    using string_t        = std::string;

public:
    // ctor, dtor
    tree();
    explicit tree(base_type&& _base) noexcept;

    tree(const graph<Tp>&);
    tree& operator=(const graph<Tp>&);

    ~tree()               = default;
    tree(const tree&)     = default;
    tree(tree&&) noexcept = default;
    tree(bool _is_dummy, uint32_t _tid, uint32_t _pid, uint64_t _hash, int64_t _depth,
         const Tp& _obj);

public:
    static Tp  get_dummy();
    bool       operator==(const tree& rhs) const;
    bool       operator!=(const tree& rhs) const;
    tree&      operator=(const tree&) = default;
    tree&      operator=(tree&&) noexcept = default;
    this_type& operator+=(const this_type& rhs)
    {
        inclusive() += rhs.inclusive();
        exclusive() += rhs.exclusive();
        for(const auto& itr : rhs.tid())
            tid().insert(itr);
        for(const auto& itr : rhs.pid())
            pid().insert(itr);
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        inclusive() -= rhs.inclusive();
        exclusive() -= rhs.exclusive();
        return *this;
    }

public:
    /// returns whether or not this node is a synchronization point and, if so, should be
    /// ignored
    bool& is_dummy() { return std::get<0>(*this); }

    /// returns the hash identifier for the associated string identifier
    uint64_t& hash() { return std::get<1>(*this); }

    /// returns the depth of the node in the tree. NOTE: this value may be relative to
    /// dummy nodes
    int64_t& depth() { return std::get<2>(*this); }

    /// the set of thread ids this data was collected from
    idset_type& tid() { return std::get<3>(*this); }

    /// the set of process ids this data was collected from
    idset_type& pid() { return std::get<4>(*this); }

    /// the inclusive data + statistics
    entry_type& inclusive() { return std::get<5>(*this); }

    /// the exclusive data + statistics
    entry_type& exclusive() { return std::get<6>(*this); }

    const bool&       is_dummy() const { return std::get<0>(*this); }
    const uint64_t&   hash() const { return std::get<1>(*this); }
    const int64_t&    depth() const { return std::get<2>(*this); }
    const idset_type& tid() const { return std::get<3>(*this); }
    const idset_type& pid() const { return std::get<4>(*this); }
    const entry_type& inclusive() const { return std::get<5>(*this); }
    const entry_type& exclusive() const { return std::get<6>(*this); }
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
: base_type(false, threading::get_id(), process::get_id(), 0, 0, Tp{}, stats_type{})
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
graph<Tp>::graph(uint64_t _id, const Tp& _obj, int64_t _depth, uint32_t _tid,
                 uint32_t _pid, bool _is_dummy)
: base_type(_is_dummy, _tid, _pid, _id, _depth, _obj, stats_type{})
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
    return operation::dummy<Tp>{}();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
result<Tp>::result(uint64_t _id, const Tp& _data, const string_t& _prefix, int64_t _depth,
                   uint64_t _rolling, const uintvector_t& _hierarchy,
                   const stats_type& _stats, uint32_t _tid, uint32_t _pid)
: base_type(_tid, _pid, _depth, _id, _rolling, _prefix, _hierarchy, _data, _stats)
{}
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
tree<Tp>::tree()
: base_type(false, 0, 0, idset_type{ threading::get_id() },
            idset_type{ process::get_id() }, entry_type{}, entry_type{})
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
tree<Tp>::tree(const graph<Tp>& rhs)
: base_type(rhs.is_dummy(), rhs.hash(), rhs.depth(), idset_type{ rhs.tid() },
            idset_type{ rhs.pid() }, entry_type{ rhs.data(), rhs.stats() },
            entry_type{ rhs.data(), rhs.stats() })
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
tree<Tp>&
tree<Tp>::operator=(const graph<Tp>& rhs)
{
    is_dummy()  = rhs.is_dummy();
    hash()      = rhs.hash();
    depth()     = rhs.depth();
    tid()       = { rhs.tid() };
    pid()       = { rhs.pid() };
    inclusive() = entry_type{ rhs.data(), rhs.stats() };
    exclusive() = entry_type{ rhs.data(), rhs.stats() };
    return *this;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace node
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace cereal
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const tim::node::graph<Tp>& d)
{
    auto _prefix = operation::decode<TIMEMORY_API>{}(::tim::get_hash_identifier(d.id()));
    ar(cereal::make_nvp("hash", d.id()), cereal::make_nvp("prefix", _prefix),
       cereal::make_nvp("entry", d.obj()), cereal::make_nvp("depth", d.depth()),
       cereal::make_nvp("stats", d.stats()), cereal::make_nvp("tid", d.tid()),
       cereal::make_nvp("pid", d.pid()), cereal::make_nvp("dummy", d.is_dummy()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, tim::node::graph<Tp>& d)
{
    std::string _prefix{};
    ar(cereal::make_nvp("hash", d.id()), cereal::make_nvp("prefix", _prefix),
       cereal::make_nvp("entry", d.obj()), cereal::make_nvp("depth", d.depth()),
       cereal::make_nvp("stats", d.stats()), cereal::make_nvp("tid", d.tid()),
       cereal::make_nvp("pid", d.pid()), cereal::make_nvp("dummy", d.is_dummy()));
    auto _id = tim::add_hash_id(_prefix);
    if(_id != d.id())
        tim::add_hash_id(_id, d.id());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
struct specialize<Archive, tim::node::graph<Tp>,
                  cereal::specialization::non_member_load_save>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp, typename StatT>
void
save(Archive& ar, const tim::node::entry<Tp, StatT>& e)
{
    ar(cereal::make_nvp("entry", e.data()));
    ar(cereal::make_nvp("stats", e.stats()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp, typename StatT>
void
load(Archive& ar, tim::node::entry<Tp, StatT>& e)
{
    ar(cereal::make_nvp("entry", e.data()));
    ar(cereal::make_nvp("stats", e.stats()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp, typename StatT>
struct specialize<Archive, tim::node::entry<Tp, StatT>,
                  cereal::specialization::non_member_load_save>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const tim::node::tree<Tp>& t)
{
    auto _prefix =
        operation::decode<TIMEMORY_API>{}(::tim::get_hash_identifier(t.hash()));
    ar(cereal::make_nvp("hash", t.hash()), cereal::make_nvp("prefix", _prefix),
       cereal::make_nvp("tid", t.tid()), cereal::make_nvp("pid", t.pid()),
       cereal::make_nvp("depth", t.depth()), cereal::make_nvp("is_dummy", t.is_dummy()));
    ar(cereal::make_nvp("inclusive", t.inclusive()));
    ar(cereal::make_nvp("exclusive", t.exclusive()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, tim::node::tree<Tp>& t)
{
    std::string _prefix{};
    ar(cereal::make_nvp("hash", t.hash()), cereal::make_nvp("prefix", _prefix),
       cereal::make_nvp("tid", t.tid()), cereal::make_nvp("pid", t.pid()),
       cereal::make_nvp("depth", t.depth()), cereal::make_nvp("is_dummy", t.is_dummy()));
    ar(cereal::make_nvp("inclusive", t.inclusive()));
    ar(cereal::make_nvp("exclusive", t.exclusive()));
    auto _id = tim::add_hash_id(_prefix);
    if(_id != t.hash())
        tim::add_hash_id(_id, t.hash());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
struct specialize<Archive, tim::node::tree<Tp>,
                  cereal::specialization::non_member_load_save>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
save(Archive& ar, const tim::node::result<Tp>& r)
{
    ar(cereal::make_nvp("hash", r.hash()), cereal::make_nvp("prefix", r.prefix()),
       cereal::make_nvp("depth", r.depth()), cereal::make_nvp("entry", r.data()),
       cereal::make_nvp("stats", r.stats()),
       cereal::make_nvp("rolling_hash", r.rolling_hash()));
    // ar(cereal::make_nvp("hierarchy", r.hierarchy()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
load(Archive& ar, tim::node::result<Tp>& r)
{
    ar(cereal::make_nvp("hash", r.hash()), cereal::make_nvp("prefix", r.prefix()),
       cereal::make_nvp("depth", r.depth()), cereal::make_nvp("entry", r.data()),
       cereal::make_nvp("stats", r.stats()),
       cereal::make_nvp("rolling_hash", r.rolling_hash()));
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
    result_nodes.resize(nnodes);

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
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
