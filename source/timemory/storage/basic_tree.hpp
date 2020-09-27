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

#include "timemory/hash/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
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
        using iterator_t = typename GraphT::sibling_iterator;

        m_value              = *root;
        iterator_t _begin    = g.begin(root);
        iterator_t _end      = g.end(root);
        auto       nchildren = std::distance(_begin, _end);
        if(nchildren > 0)
        {
            m_children.reserve(nchildren);
            for(auto itr = _begin; itr != _end; ++itr)
            {
                if(!itr->is_dummy())
                {
                    m_value.exclusive().data() -= itr->data();
                    m_value.exclusive().stats() -= itr->stats();
                    m_children.push_back(child_type{}(g, itr));
                }
                else
                {
                    iterator_t _dbegin = g.begin(itr);
                    iterator_t _dend   = g.end(itr);
                    for(auto ditr = _dbegin; ditr != _dend; ++ditr)
                    {
                        if(!ditr->is_dummy())
                            m_children.push_back(child_type{}(g, ditr));
                    }
                }
            }
        }
        return *this;
    }

    friend bool operator==(const this_type& lhs, const this_type& rhs)
    {
        auto _lhash = get_hash_id(get_hash_aliases(), lhs.m_value.hash());
        auto _rhash = get_hash_id(get_hash_aliases(), rhs.m_value.hash());
        return (_lhash == _rhash);
    }

    friend bool operator!=(const this_type& lhs, const this_type& rhs)
    {
        return !(lhs == rhs);
    }

    this_type& operator+=(const this_type& rhs)
    {
        if(*this == rhs)
        {
            m_value += rhs.m_value;
        }

        for(auto& ritr : rhs.m_children)
        {
            bool found = false;
            for(auto& itr : m_children)
            {
                if(itr == ritr)
                {
                    found = true;
                    itr += ritr;
                    break;
                }
            }
            if(!found)
                m_children.insert(m_children.end(), ritr);
        }
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        if(*this == rhs)
        {
            m_value -= rhs.m_value;
        }

        for(auto& ritr : rhs.m_children)
        {
            bool found = false;
            for(auto& itr : m_children)
            {
                if(itr == ritr)
                {
                    found = true;
                    itr -= ritr;
                }
            }
            if(!found)
                m_children.insert(m_children.end(), ritr);
        }
        return *this;
    }

    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        ar(cereal::make_nvp("node", m_value), cereal::make_nvp("children", m_children));
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("node", m_value), cereal::make_nvp("children", m_children));
    }

    auto& get_value() { return m_value; }
    auto& get_children() { return m_children; }

    const auto& get_value() const { return m_value; }
    const auto& get_children() const { return m_children; }

private:
    value_type    m_value    = {};
    children_type m_children = {};
};

}  // namespace tim
