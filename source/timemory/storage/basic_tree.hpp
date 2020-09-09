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

#include "timemory/utility/serializer.hpp"
#include "timemory/utility/types.hpp"

#include <iterator>
#include <vector>

namespace tim
{
/// \struct basic_tree
/// \brief Basic hierarchical tree implementation. Expects population from
/// \ref tim::graph
template <typename Tp>
struct basic_tree
{
    using this_type     = basic_tree<Tp>;
    using value_type    = Tp;
    using child_type    = this_type;
    using children_type = std::vector<child_type>;

    TIMEMORY_DEFAULT_OBJECT(basic_tree)

    template <typename GraphT, typename ItrT>
    this_type& operator()(GraphT g, ItrT root)
    {
        m_inclusive          = *root;
        m_exclusive          = *root;
        using iterator_t     = typename GraphT::sibling_iterator;
        iterator_t _begin    = g.begin(root);
        iterator_t _end      = g.end(root);
        auto       nchildren = std::distance(_begin, _end);
        if(nchildren > 0)
        {
            m_children.reserve(nchildren);
            for(auto itr = _begin; itr != _end; ++itr)
            {
                m_exclusive.data() -= itr->data();
                m_exclusive.stats() -= itr->stats();
                m_children.push_back(child_type{}(g, itr));
            }
        }
        return *this;
    }

    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        ar(cereal::make_nvp("inclusive", m_inclusive),
           cereal::make_nvp("exclusive", m_exclusive),
           cereal::make_nvp("children", m_children));
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("inclusive", m_inclusive),
           cereal::make_nvp("exclusive", m_exclusive),
           cereal::make_nvp("children", m_children));
    }

    auto& get_inclusive() { return m_inclusive; }
    auto& get_exclusive() { return m_exclusive; }
    auto& get_children() { return m_children; }

    const auto& get_inclusive() const { return m_inclusive; }
    const auto& get_exclusive() const { return m_exclusive; }
    const auto& get_children() const { return m_children; }

private:
    value_type    m_inclusive = {};
    value_type    m_exclusive = {};
    children_type m_children  = {};
};

}  // namespace tim
