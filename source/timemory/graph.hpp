//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file graph.hpp
 * \headerfile graph.hpp "timemory/graph.hpp"
 * Arbitrary Graph / Tree (i.e. binary-tree but not binary)
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>

//--------------------------------------------------------------------------------------//

namespace tim
{
//======================================================================================//

/// A node in the graph, combining links to other nodes as well as the actual
/// data.
template <typename T>
class tgraph_node
{
    // size: 5*4=20 bytes (on 32 bit arch), can be reduced by 8.
public:
    tgraph_node();
    explicit tgraph_node(const T&);
    explicit tgraph_node(T&&);

    tgraph_node(const tgraph_node&) = delete;
    tgraph_node(tgraph_node&&)      = default;

    tgraph_node& operator=(const tgraph_node&) = delete;
    tgraph_node& operator=(tgraph_node&&) = default;

    tgraph_node<T>* parent       = nullptr;
    tgraph_node<T>* first_child  = nullptr;
    tgraph_node<T>* last_child   = nullptr;
    tgraph_node<T>* prev_sibling = nullptr;
    tgraph_node<T>* next_sibling = nullptr;
    T               data         = T();
};

//======================================================================================//

template <typename T>
tgraph_node<T>::tgraph_node()
: parent(nullptr)
, first_child(nullptr)
, last_child(nullptr)
, prev_sibling(nullptr)
, next_sibling(nullptr)
{
}

//--------------------------------------------------------------------------------------//

template <typename T>
tgraph_node<T>::tgraph_node(const T& val)
: data(val)
{
}

//--------------------------------------------------------------------------------------//

template <typename T>
tgraph_node<T>::tgraph_node(T&& val)
: data(std::move(val))
{
}

//======================================================================================//

template <typename T, typename AllocatorT = std::allocator<tgraph_node<T>>>
class graph
{
protected:
    using graph_node = tgraph_node<T>;

public:
    /// Value of the data stored at a node.
    using value_type = T;

    class iterator_base;
    class pre_order_iterator;
    class sibling_iterator;

    graph();          // empty constructor
    graph(const T&);  // constructor setting given element as head
    graph(const iterator_base&);
    graph(const graph<T, AllocatorT>&);  // copy constructor
    graph(graph<T, AllocatorT>&&);       // move constructor
    ~graph();
    graph<T, AllocatorT>& operator=(const graph<T, AllocatorT>&);  // copy assignment
    graph<T, AllocatorT>& operator=(graph<T, AllocatorT>&&);       // move assignment

    /// Base class for iterators, only pointers stored, no traversal logic.
    class iterator_base
    {
    public:
        typedef T                               value_type;
        typedef T*                              pointer;
        typedef T&                              reference;
        typedef size_t                          size_type;
        typedef ptrdiff_t                       difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

        iterator_base();
        iterator_base(graph_node*);

        T& operator*() const;
        T* operator->() const;

        /// When called, the next increment/decrement skips children of this
        /// node.
        void skip_children();
        void skip_children(bool skip);
        /// Number of children of the node pointed to by the iterator.
        unsigned int number_of_children() const;

        sibling_iterator begin() const;
        sibling_iterator end() const;

        operator bool() const { return node != nullptr; }

        graph_node* node = nullptr;

    protected:
        bool m_skip_current_children = false;
    };

    /// Depth-first iterator, first accessing the node, then its children.
    class pre_order_iterator : public iterator_base
    {
    public:
        pre_order_iterator();
        pre_order_iterator(graph_node*);
        pre_order_iterator(const iterator_base&);
        pre_order_iterator(const sibling_iterator&);

        bool                operator==(const pre_order_iterator&) const;
        bool                operator!=(const pre_order_iterator&) const;
        pre_order_iterator& operator++();
        pre_order_iterator& operator--();
        pre_order_iterator  operator++(int);
        pre_order_iterator  operator--(int);
        pre_order_iterator& operator+=(unsigned int);
        pre_order_iterator& operator-=(unsigned int);
        pre_order_iterator  operator+(unsigned int);

        pre_order_iterator& next_skip_children();
    };

    /// The default iterator types throughout the graph class.
    typedef pre_order_iterator                 iterator;
    typedef const iterator                     const_iterator;
    typedef typename iterator::difference_type difference_type;

    /// Iterator which traverses only the nodes which are siblings of each
    /// other.
    class sibling_iterator : public iterator_base
    {
    public:
        sibling_iterator();
        sibling_iterator(graph_node*);
        sibling_iterator(const sibling_iterator&);
        sibling_iterator(const iterator_base&);

        bool              operator==(const sibling_iterator&) const;
        bool              operator!=(const sibling_iterator&) const;
        sibling_iterator& operator++();
        sibling_iterator& operator--();
        sibling_iterator  operator++(int);
        sibling_iterator  operator--(int);
        sibling_iterator& operator+=(unsigned int);
        sibling_iterator& operator-=(unsigned int);
        sibling_iterator  operator+(unsigned int);

        graph_node* range_first() const;
        graph_node* range_last() const;
        graph_node* m_parent;

    private:
        void m_set_parent();
    };

    /// Return iterator to the beginning of the graph.
    inline pre_order_iterator begin() const;
    /// Return iterator to the end of the graph.
    inline pre_order_iterator end() const;
    /// Return sibling iterator to the first child of given node.
    static sibling_iterator begin(const iterator_base&);
    /// Return sibling end iterator for children of given node.
    static sibling_iterator end(const iterator_base&);

    /// Return iterator to the parent of a node.
    template <typename iter>
    static iter parent(iter);
    /// Return iterator to the previous sibling of a node.
    template <typename iter>
    static iter previous_sibling(iter);
    /// Return iterator to the next sibling of a node.
    template <typename iter>
    static iter next_sibling(iter);

    /// Erase all nodes of the graph.
    inline void clear();
    /// Erase element at position pointed to by iterator, return incremented
    /// iterator.
    template <typename iter>
    inline iter erase(iter);
    /// Erase all children of the node pointed to by iterator.
    inline void erase_children(const iterator_base&);
    /// Erase all siblings to the right of the iterator.
    inline void erase_right_siblings(const iterator_base&);
    /// Erase all siblings to the left of the iterator.
    inline void erase_left_siblings(const iterator_base&);

    /// Insert empty node as last/first child of node pointed to by position.
    template <typename iter>
    inline iter append_child(iter position);
    template <typename iter>
    inline iter prepend_child(iter position);
    /// Insert node as last/first child of node pointed to by position.
    template <typename iter>
    inline iter append_child(iter position, const T& x);
    template <typename iter>
    inline iter append_child(iter position, T&& x);
    template <typename iter>
    inline iter prepend_child(iter position, const T& x);
    template <typename iter>
    inline iter prepend_child(iter position, T&& x);
    /// Append the node (plus its children) at other_position as last/first
    /// child of position.
    template <typename iter>
    inline iter append_child(iter position, iter other_position);
    template <typename iter>
    inline iter prepend_child(iter position, iter other_position);
    /// Append the nodes in the from-to range (plus their children) as
    /// last/first children of position.
    template <typename iter>
    inline iter append_children(iter position, sibling_iterator from,
                                const sibling_iterator& to);
    template <typename iter>
    inline iter prepend_children(iter position, sibling_iterator from,
                                 sibling_iterator to);

    /// Short-hand to insert topmost node in otherwise empty graph.
    inline pre_order_iterator set_head(const T& x);
    inline pre_order_iterator set_head(T&& x);
    /// Insert node as previous sibling of node pointed to by position.
    template <typename iter>
    inline iter insert(iter position, const T& x);
    template <typename iter>
    inline iter insert(iter position, T&& x);
    /// Specialisation of previous member.
    inline sibling_iterator insert(sibling_iterator position, const T& x);
    /// Insert node (with children) pointed to by subgraph as previous sibling
    /// of node pointed to by position. Does not change the subgraph itself (use
    /// move_in or move_in_below for that).
    template <typename iter>
    inline iter insert_subgraph(iter position, const iterator_base& subgraph);
    /// Insert node as next sibling of node pointed to by position.
    template <typename iter>
    inline iter insert_after(iter position, const T& x);
    template <typename iter>
    inline iter insert_after(iter position, T&& x);
    /// Insert node (with children) pointed to by subgraph as next sibling of
    /// node pointed to by position.
    template <typename iter>
    inline iter insert_subgraph_after(iter position, const iterator_base& subgraph);

    /// Replace node at 'position' with other node (keeping same children);
    /// 'position' becomes invalid.
    template <typename iter>
    inline iter replace(iter position, const T& x);
    /// Replace node at 'position' with subgraph starting at 'from' (do not
    /// erase subgraph at 'from'); see above.
    template <typename iter>
    inline iter replace(iter position, const iterator_base& from);
    /// Replace string of siblings (plus their children) with copy of a new
    /// string (with children); see above
    inline sibling_iterator replace(sibling_iterator        orig_begin,
                                    const sibling_iterator& orig_end,
                                    sibling_iterator        new_begin,
                                    const sibling_iterator& new_end);

    /// Move all children of node at 'position' to be siblings, returns
    /// position.
    template <typename iter>
    inline iter flatten(iter position);
    /// Move nodes in range to be children of 'position'.
    template <typename iter>
    inline iter reparent(iter position, sibling_iterator begin,
                         const sibling_iterator& end);
    /// Move all child nodes of 'from' to be children of 'position'.
    template <typename iter>
    inline iter reparent(iter position, iter from);

    /// Replace node with a new node, making the old node (plus subgraph) a
    /// child of the new node.
    template <typename iter>
    inline iter wrap(iter position, const T& x);
    /// Replace the range of sibling nodes (plus subgraphs), making these
    /// children of the new node.
    template <typename iter>
    inline iter wrap(iter from, iter to, const T& x);

    /// Move 'source' node (plus its children) to become the next sibling of
    /// 'target'.
    template <typename iter>
    inline iter move_after(iter target, iter source);
    /// Move 'source' node (plus its children) to become the previous sibling of
    /// 'target'.
    template <typename iter>
    inline iter             move_before(iter target, iter source);
    inline sibling_iterator move_before(sibling_iterator target, sibling_iterator source);
    /// Move 'source' node (plus its children) to become the node at 'target'
    /// (erasing the node at 'target').
    template <typename iter>
    inline iter move_ontop(iter target, iter source);

    /// Extract the subgraph starting at the indicated node, removing it from
    /// the original graph.
    inline graph move_out(iterator);
    /// Inverse of take_out: inserts the given graph as previous sibling of
    /// indicated node by a move operation, that is, the given graph becomes
    /// empty. Returns iterator to the top node.
    template <typename iter>
    inline iter move_in(iter, graph&);
    /// As above, but now make the graph a child of the indicated node.
    template <typename iter>
    inline iter move_in_below(iter, graph&);
    /// As above, but now make the graph the nth child of the indicated node (if
    /// possible).
    template <typename iter>
    inline iter move_in_as_nth_child(iter, size_t, graph&);

    /// Merge with other graph, creating new branches and leaves only if they
    /// are not already present.
    inline void merge(const sibling_iterator&, const sibling_iterator&, sibling_iterator,
                      const sibling_iterator&, bool duplicate_leaves = false,
                      bool first = false);
    /// Reduce duplicate nodes
    template <typename Predicate>
    inline void reduce(const sibling_iterator&, const sibling_iterator&,
                       const sibling_iterator&, sibling_iterator, const Predicate&);
    /// Sort (std::sort only moves values of nodes, this one moves children as
    /// well).
    inline void sort(const sibling_iterator& from, const sibling_iterator& to,
                     bool deep = false);
    template <class StrictWeakOrdering>
    inline void sort(sibling_iterator from, const sibling_iterator& to,
                     StrictWeakOrdering comp, bool deep = false);
    /// Compare two ranges of nodes (compares nodes as well as graph structure).
    template <typename iter>
    inline bool equal(const iter& one, const iter& two, const iter& three) const;
    template <typename iter, class BinaryPredicate>
    inline bool equal(const iter& one, const iter& two, const iter& three,
                      BinaryPredicate) const;
    template <typename iter>
    inline bool equal_subgraph(const iter& one, const iter& two) const;
    template <typename iter, class BinaryPredicate>
    inline bool equal_subgraph(const iter& one, const iter& two, BinaryPredicate) const;
    /// Extract a new graph formed by the range of siblings plus all their
    /// children.
    inline graph subgraph(sibling_iterator from, sibling_iterator to) const;
    inline void  subgraph(graph&, sibling_iterator from, sibling_iterator to) const;
    /// Exchange the node (plus subgraph) with its sibling node (do nothing if
    /// no sibling present).
    inline void swap(sibling_iterator it);
    /// Exchange two nodes (plus subgraphs). The iterators will remain valid and
    /// keep pointing to the same nodes, which now sit at different locations in
    /// the graph.
    inline void swap(iterator, iterator);

    /// Count the total number of nodes.
    inline size_t size() const;
    /// Count the total number of nodes below the indicated node (plus one).
    inline size_t size(const iterator_base&) const;
    /// Check if graph is empty.
    inline bool empty() const;
    /// Compute the depth to the root or to a fixed other iterator.
    static int depth(const iterator_base&);
    static int depth(const iterator_base&, const iterator_base&);
    /// Determine the maximal depth of the graph. An empty graph has
    /// max_depth=-1.
    inline int max_depth() const;
    /// Determine the maximal depth of the graph with top node at the given
    /// position.
    inline int max_depth(const iterator_base&) const;
    /// Count the number of children of node at position.
    static unsigned int number_of_children(const iterator_base&);
    /// Count the number of siblings (left and right) of node at iterator. Total
    /// nodes at this level is +1.
    inline unsigned int number_of_siblings(const iterator_base&) const;
    /// Determine whether node at position is in the subgraphs with root in the
    /// range.
    inline bool is_in_subgraph(const iterator_base& position, const iterator_base& begin,
                               const iterator_base& end) const;
    /// Determine whether the iterator is an 'end' iterator and thus not
    /// actually pointing to a node.
    inline bool is_valid(const iterator_base&) const;
    /// Determine whether the iterator is one of the 'head' nodes at the top
    /// level, i.e. has no parent.
    static bool is_head(const iterator_base&);
    /// Find the lowest common ancestor of two nodes, that is, the deepest node
    /// such that both nodes are descendants of it.
    inline iterator lowest_common_ancestor(const iterator_base&,
                                           const iterator_base&) const;

    /// Determine the index of a node in the range of siblings to which it
    /// belongs.
    inline unsigned int index(sibling_iterator it) const;
    /// Inverse of 'index': return the n-th child of the node at position.
    static sibling_iterator child(const iterator_base& position, unsigned int);
    /// Return iterator to the sibling indicated by index
    inline sibling_iterator sibling(const iterator_base& position, unsigned int) const;

    /// For debugging only: verify internal consistency by inspecting all
    /// pointers in the graph (which will also trigger a valgrind error in case
    /// something got corrupted).
    inline void debug_verify_consistency() const;

    /// Comparator class for iterators (compares pointer values; why doesn't
    /// this work automatically?)
    class iterator_base_less
    {
    public:
        bool operator()(const typename graph<T, AllocatorT>::iterator_base& one,
                        const typename graph<T, AllocatorT>::iterator_base& two) const
        {
            return one.node < two.node;
        }
    };

    graph_node* head;  // head/feet are always dummy; if an iterator
    graph_node* feet;  // points to them it is invalid

private:
    AllocatorT  m_alloc;
    inline void m_head_initialize();
    inline void m_copy(const graph<T, AllocatorT>& other);

    /// Comparator class for two nodes of a graph (used for sorting and searching).
    template <class StrictWeakOrdering>
    class compare_nodes
    {
    public:
        explicit compare_nodes(StrictWeakOrdering comp)
        : m_comp(comp)
        {
        }

        bool operator()(const graph_node* a, const graph_node* b)
        {
            return m_comp(a->data, b->data);
        }

    private:
        StrictWeakOrdering m_comp;
    };
};

//======================================================================================//
// Graph
//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::graph()
{
    m_head_initialize();
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::graph(const T& x)
{
    m_head_initialize();
    set_head(x);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::graph(graph<T, AllocatorT>&& x)
{
    m_head_initialize();
    if(x.head->next_sibling != x.feet)
    {  // move graph if non-empty only
        head->next_sibling                 = x.head->next_sibling;
        feet->prev_sibling                 = x.head->prev_sibling;
        x.head->next_sibling->prev_sibling = head;
        x.feet->prev_sibling->next_sibling = feet;
        x.head->next_sibling               = x.feet;
        x.feet->prev_sibling               = x.head;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::graph(const iterator_base& other)
{
    m_head_initialize();
    set_head((*other));
    replace(begin(), other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::~graph()
{
    clear();
    m_alloc.destroy(head);
    m_alloc.destroy(feet);
    m_alloc.deallocate(head, 1);
    m_alloc.deallocate(feet, 1);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::m_head_initialize()
{
    head = m_alloc.allocate(1, 0);  // MSVC does not have default second argument
    feet = m_alloc.allocate(1, 0);
    m_alloc.construct(head, std::move(tgraph_node<T>()));
    m_alloc.construct(feet, std::move(tgraph_node<T>()));

    head->parent       = 0;
    head->first_child  = 0;
    head->last_child   = 0;
    head->prev_sibling = 0;     // head;
    head->next_sibling = feet;  // head;

    feet->parent       = 0;
    feet->first_child  = 0;
    feet->last_child   = 0;
    feet->prev_sibling = head;
    feet->next_sibling = 0;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>&
graph<T, AllocatorT>::operator=(const graph<T, AllocatorT>& other)
{
    if(this != &other)
        m_copy(other);
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>&
graph<T, AllocatorT>::operator=(graph<T, AllocatorT>&& x)
{
    if(this != &x)
    {
        head->next_sibling                 = x.head->next_sibling;
        feet->prev_sibling                 = x.head->prev_sibling;
        x.head->next_sibling->prev_sibling = head;
        x.feet->prev_sibling->next_sibling = feet;
        x.head->next_sibling               = x.feet;
        x.feet->prev_sibling               = x.head;
    }
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::graph(const graph<T, AllocatorT>& other)
{
    m_head_initialize();
    m_copy(other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::m_copy(const graph<T, AllocatorT>& other)
{
    clear();
    pre_order_iterator it = other.begin(), to = begin();
    while(it != other.end())
    {
        to = insert(to, (*it));
        it.skip_children();
        ++it;
    }
    to = begin();
    it = other.begin();
    while(it != other.end())
    {
        to = replace(to, it);
        to.skip_children();
        it.skip_children();
        ++to;
        ++it;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::clear()
{
    if(head)
        while(head->next_sibling != feet)
            erase(pre_order_iterator(head->next_sibling));
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::erase_children(const iterator_base& it)
{
    //	std::cout << "erase_children " << it.node << std::endl;
    if(it.node == 0)
        return;

    graph_node* cur = it.node->first_child;

    while(cur != 0)
    {
        graph_node* prev = 0;
        cur              = cur->next_sibling;
        erase_children(pre_order_iterator(prev));
        m_alloc.destroy(prev);
        m_alloc.deallocate(prev, 1);
    }
    it.node->first_child = 0;
    it.node->last_child  = 0;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::erase(iter it)
{
    graph_node* cur = it.node;
    assert(cur != head);
    iter ret = it;
    ret.skip_children();
    ++ret;
    erase_children(it);
    if(cur->prev_sibling == 0)
    {
        cur->parent->first_child = cur->next_sibling;
    }
    else
    {
        cur->prev_sibling->next_sibling = cur->next_sibling;
    }
    if(cur->next_sibling == 0)
    {
        cur->parent->last_child = cur->prev_sibling;
    }
    else
    {
        cur->next_sibling->prev_sibling = cur->prev_sibling;
    }

    //	kp::destructor(&cur->data);
    m_alloc.destroy(cur);
    m_alloc.deallocate(cur, 1);
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::begin() const
{
    return pre_order_iterator(head->next_sibling);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::end() const
{
    return pre_order_iterator(feet);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::begin(const iterator_base& pos)
{
    assert(pos.node != 0);
    if(pos.node->first_child == 0)
    {
        return end(pos);
    }
    return pos.node->first_child;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::end(const iterator_base& pos)
{
    sibling_iterator ret(nullptr);
    ret.m_parent = pos.node;
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::parent(iter position)
{
    assert(position.node != 0);
    return iter(position.node->parent);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::previous_sibling(iter position)
{
    assert(position.node != 0);
    iter ret(position);
    ret.node = position.node->prev_sibling;
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::next_sibling(iter position)
{
    assert(position.node != 0);
    iter ret(position);
    ret.node = position.node->next_sibling;
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::append_child(iter position)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp, tgraph_node<T>());
    //	kp::constructor(&tmp->data);
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent = position.node;
    if(position.node->last_child != 0)
    {
        position.node->last_child->next_sibling = tmp;
    }
    else
    {
        position.node->first_child = tmp;
    }
    tmp->prev_sibling         = position.node->last_child;
    position.node->last_child = tmp;
    tmp->next_sibling         = 0;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::prepend_child(iter position)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp, tgraph_node<T>());
    //	kp::constructor(&tmp->data);
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent = position.node;
    if(position.node->first_child != 0)
    {
        position.node->first_child->prev_sibling = tmp;
    }
    else
    {
        position.node->last_child = tmp;
    }
    tmp->next_sibling         = position.node->first_child;
    position.node->prev_child = tmp;
    tmp->prev_sibling         = 0;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::append_child(iter position, const T& x)
{
    // If your program fails here you probably used 'append_child' to add the
    // top node to an empty graph. From version 1.45 the top element should be
    // added using 'insert'. See the documentation for further information, and
    // sorry about the API change.
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp, x);
    //	kp::constructor(&tmp->data, x);
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent = position.node;
    if(position.node->last_child != 0)
    {
        position.node->last_child->next_sibling = tmp;
    }
    else
    {
        position.node->first_child = tmp;
    }
    tmp->prev_sibling         = position.node->last_child;
    position.node->last_child = tmp;
    tmp->next_sibling         = 0;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::append_child(iter position, T&& x)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp);  // Here is where the move semantics kick in
    std::swap(tmp->data, std::move(x));

    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent = position.node;
    if(position.node->last_child != 0)
    {
        position.node->last_child->next_sibling = tmp;
    }
    else
    {
        position.node->first_child = tmp;
    }
    tmp->prev_sibling         = position.node->last_child;
    position.node->last_child = tmp;
    tmp->next_sibling         = 0;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::prepend_child(iter position, const T& x)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp, x);
    //	kp::constructor(&tmp->data, x);
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent = position.node;
    if(position.node->first_child != 0)
    {
        position.node->first_child->prev_sibling = tmp;
    }
    else
    {
        position.node->last_child = tmp;
    }
    tmp->next_sibling          = position.node->first_child;
    position.node->first_child = tmp;
    tmp->prev_sibling          = 0;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::prepend_child(iter position, T&& x)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp);
    std::swap(tmp->data, std::move(x));

    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent = position.node;
    if(position.node->first_child != 0)
    {
        position.node->first_child->prev_sibling = tmp;
    }
    else
    {
        position.node->last_child = tmp;
    }
    tmp->next_sibling          = position.node->first_child;
    position.node->first_child = tmp;
    tmp->prev_sibling          = 0;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::append_child(iter position, iter other)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    sibling_iterator aargh = append_child(position, value_type());
    return replace(aargh, other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::prepend_child(iter position, iter other)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    sibling_iterator aargh = prepend_child(position, value_type());
    return replace(aargh, other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::append_children(iter position, sibling_iterator from,
                                      const sibling_iterator& to)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    iter ret = from;

    while(from != to)
    {
        insert_subgraph(position.end(), from);
        ++from;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::prepend_children(iter position, sibling_iterator from,
                                       sibling_iterator to)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    if(from == to)
        return from;  // should return end of graph?

    iter ret;
    do
    {
        --to;
        ret = insert_subgraph(position.begin(), to);
    } while(to != from);

    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::set_head(const T& x)
{
    assert(head->next_sibling == feet);
    return insert(iterator(feet), x);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::set_head(T&& x)
{
    assert(head->next_sibling == feet);
    return insert(iterator(feet), std::move(x));
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::insert(iter position, const T& x)
{
    if(position.node == 0)
    {
        position.node = feet;  // Backward compatibility: when calling insert on
                               // a null node, insert before the feet.
    }
    assert(position.node != head);  // Cannot insert before head.

    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp, x);
    //	kp::constructor(&tmp->data, x);
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent                 = position.node->parent;
    tmp->next_sibling           = position.node;
    tmp->prev_sibling           = position.node->prev_sibling;
    position.node->prev_sibling = tmp;

    if(tmp->prev_sibling == 0)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->first_child = tmp;
    }
    else
        tmp->prev_sibling->next_sibling = tmp;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::insert(iter position, T&& x)
{
    if(position.node == 0)
    {
        position.node = feet;  // Backward compatibility: when calling insert on
                               // a null node, insert before the feet.
    }
    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp);
    std::swap(tmp->data, x);  // Move semantics
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent                 = position.node->parent;
    tmp->next_sibling           = position.node;
    tmp->prev_sibling           = position.node->prev_sibling;
    position.node->prev_sibling = tmp;

    if(tmp->prev_sibling == 0)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->first_child = tmp;
    }
    else
        tmp->prev_sibling->next_sibling = tmp;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::insert(sibling_iterator position, const T& x)
{
    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp, x);
    //	kp::constructor(&tmp->data, x);
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->next_sibling = position.node;
    if(position.node == 0)
    {  // iterator points to end of a subgraph
        tmp->parent             = position.m_parent;
        tmp->prev_sibling       = position.range_last();
        tmp->parent->last_child = tmp;
    }
    else
    {
        tmp->parent                 = position.node->parent;
        tmp->prev_sibling           = position.node->prev_sibling;
        position.node->prev_sibling = tmp;
    }

    if(tmp->prev_sibling == 0)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->first_child = tmp;
    }
    else
        tmp->prev_sibling->next_sibling = tmp;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::insert_after(iter position, const T& x)
{
    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp, x);
    //	kp::constructor(&tmp->data, x);
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent                 = position.node->parent;
    tmp->prev_sibling           = position.node;
    tmp->next_sibling           = position.node->next_sibling;
    position.node->next_sibling = tmp;

    if(tmp->next_sibling == 0)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->last_child = tmp;
    }
    else
    {
        tmp->next_sibling->prev_sibling = tmp;
    }
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::insert_after(iter position, T&& x)
{
    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp);
    std::swap(tmp->data, x);  // move semantics
                              //	kp::constructor(&tmp->data, x);
    tmp->first_child = 0;
    tmp->last_child  = 0;

    tmp->parent                 = position.node->parent;
    tmp->prev_sibling           = position.node;
    tmp->next_sibling           = position.node->next_sibling;
    position.node->next_sibling = tmp;

    if(tmp->next_sibling == 0)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->last_child = tmp;
    }
    else
    {
        tmp->next_sibling->prev_sibling = tmp;
    }
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::insert_subgraph(iter position, const iterator_base& _subgraph)
{
    // insert dummy
    iter it = insert(position, value_type());
    // replace dummy with subgraph
    return replace(it, _subgraph);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::insert_subgraph_after(iter position, const iterator_base& _subgraph)
{
    // insert dummy
    iter it = insert_after(position, value_type());
    // replace dummy with subgraph
    return replace(it, _subgraph);
}

//--------------------------------------------------------------------------------------//

// template <typename T, typename AllocatorT>
// template <class iter>
// iter graph<T, AllocatorT>::insert_subgraph(sibling_iterator
// position, iter subgraph)
// 	{
// 	// insert dummy
// 	iter it(insert(position, value_type()));
// 	// replace dummy with subgraph
// 	return replace(it, subgraph);
// 	}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::replace(iter position, const T& x)
{
    //	kp::destructor(&position.node->data);
    //	kp::constructor(&position.node->data, x);
    position.node->data = x;
    //	m_alloc.destroy(position.node);
    //	m_alloc.construct(position.node, x);
    return position;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class iter>
iter
graph<T, AllocatorT>::replace(iter position, const iterator_base& from)
{
    assert(position.node != head);
    graph_node* current_from = from.node;
    graph_node* start_from   = from.node;
    graph_node* current_to   = position.node;

    // replace the node at position with head of the replacement graph at from
    //	std::cout << "warning!" << position.node << std::endl;
    erase_children(position);
    //	std::cout << "no warning!" << std::endl;
    graph_node* tmp = m_alloc.allocate(1, 0);
    m_alloc.construct(tmp, (*from));
    //	kp::constructor(&tmp->data, (*from));
    tmp->first_child = 0;
    tmp->last_child  = 0;
    if(current_to->prev_sibling == 0)
    {
        if(current_to->parent != 0)
            current_to->parent->first_child = tmp;
    }
    else
    {
        current_to->prev_sibling->next_sibling = tmp;
    }
    tmp->prev_sibling = current_to->prev_sibling;
    if(current_to->next_sibling == 0)
    {
        if(current_to->parent != 0)
            current_to->parent->last_child = tmp;
    }
    else
    {
        current_to->next_sibling->prev_sibling = tmp;
    }
    tmp->next_sibling = current_to->next_sibling;
    tmp->parent       = current_to->parent;
    //	kp::destructor(&current_to->data);
    m_alloc.destroy(current_to);
    m_alloc.deallocate(current_to, 1);
    current_to = tmp;

    // only at this stage can we fix 'last'
    graph_node* last = from.node->next_sibling;

    pre_order_iterator toit = tmp;
    // copy all children
    do
    {
        assert(current_from != 0);
        if(current_from->first_child != 0)
        {
            current_from = current_from->first_child;
            toit         = append_child(toit, current_from->data);
        }
        else
        {
            while(current_from->next_sibling == 0 && current_from != start_from)
            {
                current_from = current_from->parent;
                toit         = parent(toit);
                assert(current_from != 0);
            }
            current_from = current_from->next_sibling;
            if(current_from != last)
            {
                toit = append_child(parent(toit), current_from->data);
            }
        }
    } while(current_from != last);

    return current_to;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::replace(sibling_iterator        orig_begin,
                              const sibling_iterator& orig_end,
                              sibling_iterator new_begin, const sibling_iterator& new_end)
{
    graph_node* orig_first = orig_begin.node;
    graph_node* new_first  = new_begin.node;
    graph_node* orig_last  = orig_first;
    while((++orig_begin) != orig_end)
        orig_last = orig_last->next_sibling;
    graph_node* new_last = new_first;
    while((++new_begin) != new_end)
        new_last = new_last->next_sibling;

    // insert all siblings in new_first..new_last before orig_first
    bool               first = true;
    pre_order_iterator ret;
    while(true)
    {
        pre_order_iterator tt = insert_subgraph(pre_order_iterator(orig_first),
                                                pre_order_iterator(new_first));
        if(first)
        {
            ret   = tt;
            first = false;
        }
        if(new_first == new_last)
            break;
        new_first = new_first->next_sibling;
    }

    // erase old range of siblings
    bool        last = false;
    graph_node* next = orig_first;
    while(true)
    {
        if(next == orig_last)
            last = true;
        next = next->next_sibling;
        erase((pre_order_iterator) orig_first);
        if(last)
            break;
        orig_first = next;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::flatten(iter position)
{
    if(position.node->first_child == 0)
        return position;

    graph_node* tmp = position.node->first_child;
    while(tmp)
    {
        tmp->parent = position.node->parent;
        tmp         = tmp->next_sibling;
    }
    if(position.node->next_sibling)
    {
        position.node->last_child->next_sibling   = position.node->next_sibling;
        position.node->next_sibling->prev_sibling = position.node->last_child;
    }
    else
    {
        position.node->parent->last_child = position.node->last_child;
    }
    position.node->next_sibling               = position.node->first_child;
    position.node->next_sibling->prev_sibling = position.node;
    position.node->first_child                = 0;
    position.node->last_child                 = 0;

    return position;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::reparent(iter position, sibling_iterator _begin,
                               const sibling_iterator& _end)
{
    graph_node* first = _begin.node;
    graph_node* last  = first;

    assert(first != position.node);

    if(_begin == _end)
        return _begin;
    // determine last node
    while((++_begin) != _end)
    {
        last = last->next_sibling;
    }
    // move subgraph
    if(first->prev_sibling == 0)
    {
        first->parent->first_child = last->next_sibling;
    }
    else
    {
        first->prev_sibling->next_sibling = last->next_sibling;
    }
    if(last->next_sibling == 0)
    {
        last->parent->last_child = first->prev_sibling;
    }
    else
    {
        last->next_sibling->prev_sibling = first->prev_sibling;
    }
    if(position.node->first_child == 0)
    {
        position.node->first_child = first;
        position.node->last_child  = last;
        first->prev_sibling        = 0;
    }
    else
    {
        position.node->last_child->next_sibling = first;
        first->prev_sibling                     = position.node->last_child;
        position.node->last_child               = last;
    }
    last->next_sibling = 0;

    graph_node* pos = first;
    for(;;)
    {
        pos->parent = position.node;
        if(pos == last)
            break;
        pos = pos->next_sibling;
    }

    return first;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::reparent(iter position, iter from)
{
    if(from.node->first_child == 0)
        return position;
    return reparent(position, from.node->first_child, end(from));
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::wrap(iter position, const T& x)
{
    assert(position.node != 0);
    sibling_iterator fr = position, to = position;
    ++to;
    iter ret = insert(position, x);
    reparent(ret, fr, to);
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::wrap(iter from, iter to, const T& x)
{
    assert(from.node != 0);
    iter ret = insert(from, x);
    reparent(ret, from, to);
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::move_after(iter target, iter source)
{
    graph_node* dst = target.node;
    graph_node* src = source.node;
    assert(dst);
    assert(src);

    if(dst == src)
        return source;
    if(dst->next_sibling)
        if(dst->next_sibling == src)  // already in the right spot
            return source;

    // take src out of the graph
    if(src->prev_sibling != 0)
        src->prev_sibling->next_sibling = src->next_sibling;
    else
        src->parent->first_child = src->next_sibling;
    if(src->next_sibling != 0)
        src->next_sibling->prev_sibling = src->prev_sibling;
    else
        src->parent->last_child = src->prev_sibling;

    // connect it to the new point
    if(dst->next_sibling != 0)
        dst->next_sibling->prev_sibling = src;
    else
        dst->parent->last_child = src;
    src->next_sibling = dst->next_sibling;
    dst->next_sibling = src;
    src->prev_sibling = dst;
    src->parent       = dst->parent;
    return src;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::move_before(iter target, iter source)
{
    graph_node* dst = target.node;
    graph_node* src = source.node;
    assert(dst);
    assert(src);

    if(dst == src)
        return source;
    if(dst->prev_sibling)
        if(dst->prev_sibling == src)  // already in the right spot
            return source;

    // take src out of the graph
    if(src->prev_sibling != 0)
        src->prev_sibling->next_sibling = src->next_sibling;
    else
        src->parent->first_child = src->next_sibling;
    if(src->next_sibling != 0)
        src->next_sibling->prev_sibling = src->prev_sibling;
    else
        src->parent->last_child = src->prev_sibling;

    // connect it to the new point
    if(dst->prev_sibling != 0)
        dst->prev_sibling->next_sibling = src;
    else
        dst->parent->first_child = src;
    src->prev_sibling = dst->prev_sibling;
    dst->prev_sibling = src;
    src->next_sibling = dst;
    src->parent       = dst->parent;
    return src;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::move_ontop(iter target, iter source)
{
    graph_node* dst = target.node;
    graph_node* src = source.node;
    assert(dst);
    assert(src);

    if(dst == src)
        return source;

    //	if(dst==src->prev_sibling) {
    //
    //		}

    // remember connection points
    graph_node* b_prev_sibling = dst->prev_sibling;
    graph_node* b_next_sibling = dst->next_sibling;
    graph_node* b_parent       = dst->parent;

    // remove target
    erase(target);

    // take src out of the graph
    if(src->prev_sibling != 0)
        src->prev_sibling->next_sibling = src->next_sibling;
    else
        src->parent->first_child = src->next_sibling;
    if(src->next_sibling != 0)
        src->next_sibling->prev_sibling = src->prev_sibling;
    else
        src->parent->last_child = src->prev_sibling;

    // connect it to the new point
    if(b_prev_sibling != 0)
        b_prev_sibling->next_sibling = src;
    else
        b_parent->first_child = src;
    if(b_next_sibling != 0)
        b_next_sibling->prev_sibling = src;
    else
        b_parent->last_child = src;
    src->prev_sibling = b_prev_sibling;
    src->next_sibling = b_next_sibling;
    src->parent       = b_parent;
    return src;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>
graph<T, AllocatorT>::move_out(iterator source)
{
    graph ret;

    // Move source node into the 'ret' graph.
    ret.head->next_sibling = source.node;
    ret.feet->prev_sibling = source.node;
    source.node->parent    = 0;

    // Close the links in the current graph.
    if(source.node->prev_sibling != 0)
        source.node->prev_sibling->next_sibling = source.node->next_sibling;

    if(source.node->next_sibling != 0)
        source.node->next_sibling->prev_sibling = source.node->prev_sibling;

    // Fix source prev/next links.
    source.node->prev_sibling = ret.head;
    source.node->next_sibling = ret.feet;

    return ret;  // A good compiler will move this, not copy.
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::move_in(iter loc, graph& other)
{
    if(other.head->next_sibling == other.feet)
        return loc;  // other graph is empty

    graph_node* other_first_head = other.head->next_sibling;
    graph_node* other_last_head  = other.feet->prev_sibling;

    sibling_iterator prev(loc);
    --prev;

    prev.node->next_sibling        = other_first_head;
    loc.node->prev_sibling         = other_last_head;
    other_first_head->prev_sibling = prev.node;
    other_last_head->next_sibling  = loc.node;

    // Adjust parent pointers.
    graph_node* walk = other_first_head;
    while(true)
    {
        walk->parent = loc.node->parent;
        if(walk == other_last_head)
            break;
        walk = walk->next_sibling;
    }

    // Close other graph.
    other.head->next_sibling = other.feet;
    other.feet->prev_sibling = other.head;

    return other_first_head;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
iter
graph<T, AllocatorT>::move_in_as_nth_child(iter loc, size_t n, graph& other)
{
    if(other.head->next_sibling == other.feet)
        return loc;  // other graph is empty

    graph_node* other_first_head = other.head->next_sibling;
    graph_node* other_last_head  = other.feet->prev_sibling;

    if(n == 0)
    {
        if(loc.node->first_child == 0)
        {
            loc.node->first_child          = other_first_head;
            loc.node->last_child           = other_last_head;
            other_last_head->next_sibling  = 0;
            other_first_head->prev_sibling = 0;
        }
        else
        {
            loc.node->first_child->prev_sibling = other_last_head;
            other_last_head->next_sibling       = loc.node->first_child;
            loc.node->first_child               = other_first_head;
            other_first_head->prev_sibling      = 0;
        }
    }
    else
    {
        --n;
        graph_node* walk = loc.node->first_child;
        while(true)
        {
            if(walk == 0)
                throw std::range_error(
                    "graph: move_in_as_nth_child position out of range");
            if(n == 0)
                break;
            --n;
            walk = walk->next_sibling;
        }
        if(walk->next_sibling == 0)
            loc.node->last_child = other_last_head;
        else
            walk->next_sibling->prev_sibling = other_last_head;
        other_last_head->next_sibling  = walk->next_sibling;
        walk->next_sibling             = other_first_head;
        other_first_head->prev_sibling = walk;
    }

    // Adjust parent pointers.
    graph_node* walk = other_first_head;
    while(true)
    {
        walk->parent = loc.node;
        if(walk == other_last_head)
            break;
        walk = walk->next_sibling;
    }

    // Close other graph.
    other.head->next_sibling = other.feet;
    other.feet->prev_sibling = other.head;

    return other_first_head;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::merge(const sibling_iterator& to1, const sibling_iterator& to2,
                            sibling_iterator from1, const sibling_iterator& from2,
                            bool duplicate_leaves, bool first)
{
    while(from1 != from2)
    {
        sibling_iterator    fnd;
        auto                nsiblings = number_of_siblings(to1);
        decltype(nsiblings) count     = 0;
        for(sibling_iterator itr = to1; itr != to2; ++itr, ++count)
        {
            if(itr && from1 && *itr == *from1)
            {
                fnd = itr;
                break;
            }
            if(count > nsiblings)
            {
                fnd = to2;
                break;
            }
        }
        // auto fnd = std::find(to1, to2, *from1);
        if(fnd != to2)  // element found
        {
            if(from1.begin() == from1.end())  // full depth reached
            {
                if(duplicate_leaves)
                    append_child(parent(to1), (*from1));
            }
            else  // descend further
            {
                if(!first)
                    *fnd += *from1;
                if(from1 != from2)
                    merge(fnd.begin(), fnd.end(), from1.begin(), from1.end(),
                          duplicate_leaves);
            }
        }
        else
        {  // element missing
            insert_subgraph(to2, from1);
        }
        do
        {
            ++from1;
        } while(!from1 && from1 != from2);
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename Predicate>
void
graph<T, AllocatorT>::reduce(const sibling_iterator& beg1, const sibling_iterator& end1,
                             const sibling_iterator& beg2, sibling_iterator end2,
                             const Predicate& predicate)
{
    for(auto itr1 = beg1; itr1 != end1; ++itr1)
    {
        for(auto itr2 = beg2; itr2 != end2; ++itr2)
        {
            // skip if same iterator
            if(itr1 == itr2)
                continue;
            if(*itr1 == *itr2)
            {
                predicate(itr1, itr2);
                reduce(itr1.begin(), itr1.end(), itr2.begin(), itr2.end(), predicate);
                this->erase(itr2);
            }
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::sort(const sibling_iterator& from, const sibling_iterator& to,
                           bool deep)
{
    std::less<T> comp;
    sort(from, to, comp, deep);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <class StrictWeakOrdering>
void
graph<T, AllocatorT>::sort(sibling_iterator from, const sibling_iterator& to,
                           StrictWeakOrdering comp, bool deep)
{
    if(from == to)
        return;
    // make list of sorted nodes
    // CHECK: if multiset stores equivalent nodes in the order in which they
    // are inserted, then this routine should be called 'stable_sort'.
    std::multiset<graph_node*, compare_nodes<StrictWeakOrdering>> nodes(comp);
    sibling_iterator                                              it = from, it2 = to;
    while(it != to)
    {
        nodes.insert(it.node);
        ++it;
    }
    // reassemble
    --it2;

    // prev and next are the nodes before and after the sorted range
    graph_node* prev = from.node->prev_sibling;
    graph_node* next = it2.node->next_sibling;
    typename std::multiset<graph_node*, compare_nodes<StrictWeakOrdering>>::iterator
        nit = nodes.begin(),
        eit = nodes.end();
    if(prev == 0)
    {
        if((*nit)->parent != 0)  // to catch "sorting the head" situations, when
                                 // there is no parent
            (*nit)->parent->first_child = (*nit);
    }
    else
        prev->next_sibling = (*nit);

    --eit;
    while(nit != eit)
    {
        (*nit)->prev_sibling = prev;
        if(prev)
            prev->next_sibling = (*nit);
        prev = (*nit);
        ++nit;
    }
    // prev now points to the last-but-one node in the sorted range
    if(prev)
        prev->next_sibling = (*eit);

    // eit points to the last node in the sorted range.
    (*eit)->next_sibling = next;
    (*eit)->prev_sibling = prev;  // missed in the loop above
    if(next == 0)
    {
        if((*eit)->parent != 0)  // to catch "sorting the head" situations, when
                                 // there is no parent
            (*eit)->parent->last_child = (*eit);
    }
    else
        next->prev_sibling = (*eit);

    if(deep)
    {  // sort the children of each node too
        sibling_iterator bcs(*nodes.begin());
        sibling_iterator ecs(*eit);
        ++ecs;
        while(bcs != ecs)
        {
            sort(begin(bcs), end(bcs), comp, deep);
            ++bcs;
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
bool
graph<T, AllocatorT>::equal(const iter& one_, const iter& two, const iter& three_) const
{
    std::equal_to<T> comp;
    return equal(one_, two, three_, comp);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter>
bool
graph<T, AllocatorT>::equal_subgraph(const iter& one_, const iter& two_) const
{
    std::equal_to<T> comp;
    return equal_subgraph(one_, two_, comp);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter, class BinaryPredicate>
bool
graph<T, AllocatorT>::equal(const iter& one_, const iter& two, const iter& three_,
                            BinaryPredicate fun) const
{
    pre_order_iterator one(one_), three(three_);

    //	if(one==two && is_valid(three) && three.number_of_children()!=0)
    //		return false;
    while(one != two && is_valid(three))
    {
        if(!fun(*one, *three))
            return false;
        if(one.number_of_children() != three.number_of_children())
            return false;
        ++one;
        ++three;
    }
    return true;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename iter, class BinaryPredicate>
bool
graph<T, AllocatorT>::equal_subgraph(const iter& one_, const iter& two_,
                                     BinaryPredicate fun) const
{
    pre_order_iterator one(one_), two(two_);

    if(!fun(*one, *two))
        return false;
    if(number_of_children(one) != number_of_children(two))
        return false;
    return equal(begin(one), end(one), begin(two), fun);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::empty() const
{
    pre_order_iterator it = begin(), eit = end();
    return (it == eit);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
int
graph<T, AllocatorT>::depth(const iterator_base& it)
{
    graph_node* pos = it.node;
    assert(pos != 0);
    int ret = 0;
    while(pos->parent != 0)
    {
        pos = pos->parent;
        ++ret;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
int
graph<T, AllocatorT>::depth(const iterator_base& it, const iterator_base& root)
{
    graph_node* pos = it.node;
    assert(pos != 0);
    int ret = 0;
    while(pos->parent != 0 && pos != root.node)
    {
        pos = pos->parent;
        ++ret;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
int
graph<T, AllocatorT>::max_depth() const
{
    int maxd = -1;
    for(graph_node* it = head->next_sibling; it != feet; it = it->next_sibling)
        maxd = std::max(maxd, max_depth(it));

    return maxd;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
int
graph<T, AllocatorT>::max_depth(const iterator_base& pos) const
{
    graph_node* tmp = pos.node;

    if(tmp == 0 || tmp == head || tmp == feet)
        return -1;

    int curdepth = 0, maxdepth = 0;
    while(true)
    {  // try to walk the bottom of the graph
        while(tmp->first_child == 0)
        {
            if(tmp == pos.node)
                return maxdepth;
            if(tmp->next_sibling == 0)
            {
                // try to walk up and then right again
                do
                {
                    tmp = tmp->parent;
                    if(tmp == 0)
                        return maxdepth;
                    --curdepth;
                } while(tmp->next_sibling == 0);
            }
            if(tmp == pos.node)
                return maxdepth;
            tmp = tmp->next_sibling;
        }
        tmp = tmp->first_child;
        ++curdepth;
        maxdepth = std::max(curdepth, maxdepth);
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
unsigned int
graph<T, AllocatorT>::number_of_children(const iterator_base& it)
{
    graph_node* pos = it.node->first_child;
    if(pos == 0)
        return 0;

    unsigned int ret = 1;
    //	  while(pos!=it.node->last_child) {
    //		  ++ret;
    //		  pos=pos->next_sibling;
    //		  }
    while((pos = pos->next_sibling))
        ++ret;
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
unsigned int
graph<T, AllocatorT>::number_of_siblings(const iterator_base& it) const
{
    graph_node*  pos = it.node;
    unsigned int ret = 0;
    // count forward
    while(pos->next_sibling && pos->next_sibling != head && pos->next_sibling != feet)
    {
        ++ret;
        pos = pos->next_sibling;
    }
    // count backward
    pos = it.node;
    while(pos->prev_sibling && pos->prev_sibling != head && pos->prev_sibling != feet)
    {
        ++ret;
        pos = pos->prev_sibling;
    }

    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::swap(sibling_iterator it)
{
    graph_node* nxt = it.node->next_sibling;
    if(nxt)
    {
        if(it.node->prev_sibling)
            it.node->prev_sibling->next_sibling = nxt;
        else
            it.node->parent->first_child = nxt;
        nxt->prev_sibling  = it.node->prev_sibling;
        graph_node* nxtnxt = nxt->next_sibling;
        if(nxtnxt)
            nxtnxt->prev_sibling = it.node;
        else
            it.node->parent->last_child = it.node;
        nxt->next_sibling     = it.node;
        it.node->prev_sibling = nxt;
        it.node->next_sibling = nxtnxt;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::swap(iterator one, iterator two)
{
    // if one and two are adjacent siblings, use the sibling swap
    if(one.node->next_sibling == two.node)
        swap(one);
    else if(two.node->next_sibling == one.node)
        swap(two);
    else
    {
        graph_node* nxt1 = one.node->next_sibling;
        graph_node* nxt2 = two.node->next_sibling;
        graph_node* pre1 = one.node->prev_sibling;
        graph_node* pre2 = two.node->prev_sibling;
        graph_node* par1 = one.node->parent;
        graph_node* par2 = two.node->parent;

        // reconnect
        one.node->parent       = par2;
        one.node->next_sibling = nxt2;
        if(nxt2)
            nxt2->prev_sibling = one.node;
        else
            par2->last_child = one.node;
        one.node->prev_sibling = pre2;
        if(pre2)
            pre2->next_sibling = one.node;
        else
            par2->first_child = one.node;

        two.node->parent       = par1;
        two.node->next_sibling = nxt1;
        if(nxt1)
            nxt1->prev_sibling = two.node;
        else
            par1->last_child = two.node;
        two.node->prev_sibling = pre1;
        if(pre1)
            pre1->next_sibling = two.node;
        else
            par1->first_child = two.node;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::is_valid(const iterator_base& it) const
{
    if(it.node == 0 || it.node == feet || it.node == head)
        return false;
    else
        return true;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::is_head(const iterator_base& it)
{
    if(it.node->parent == 0)
        return true;
    return false;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::child(const iterator_base& it, unsigned int num)
{
    graph_node* tmp = it.node->first_child;
    while(num--)
    {
        assert(tmp != 0);
        tmp = tmp->next_sibling;
    }
    return tmp;
}

//--------------------------------------------------------------------------------------//
// Iterator base

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::iterator_base::iterator_base()
: node(nullptr)
, m_skip_current_children(false)
{
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::iterator_base::iterator_base(graph_node* tn)
: node(tn)
, m_skip_current_children(false)
{
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
T& graph<T, AllocatorT>::iterator_base::operator*() const
{
    return node->data;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
T* graph<T, AllocatorT>::iterator_base::operator->() const
{
    return &(node->data);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::pre_order_iterator::operator!=(
    const pre_order_iterator& other) const
{
    if(other.node != this->node)
        return true;
    else
        return false;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::pre_order_iterator::operator==(
    const pre_order_iterator& other) const
{
    if(other.node == this->node)
        return true;
    else
        return false;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::sibling_iterator::operator!=(const sibling_iterator& other) const
{
    if(other.node != this->node)
        return true;
    else
        return false;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::sibling_iterator::operator==(const sibling_iterator& other) const
{
    if(other.node == this->node)
        return true;
    else
        return false;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::iterator_base::begin() const
{
    if(node->first_child == 0)
        return end();

    sibling_iterator ret(node->first_child);
    ret.m_parent = this->node;
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::iterator_base::end() const
{
    sibling_iterator ret(nullptr);
    ret.m_parent = node;
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::iterator_base::skip_children()
{
    m_skip_current_children = true;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::iterator_base::skip_children(bool skip)
{
    m_skip_current_children = skip;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
unsigned int
graph<T, AllocatorT>::iterator_base::number_of_children() const
{
    graph_node* pos = node->first_child;
    if(pos == 0)
        return 0;

    unsigned int ret = 1;
    while(pos != node->last_child)
    {
        ++ret;
        pos = pos->next_sibling;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//
// Pre-order iterator

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::pre_order_iterator::pre_order_iterator()
: iterator_base(nullptr)
{
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::pre_order_iterator::pre_order_iterator(graph_node* tn)
: iterator_base(tn)
{
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::pre_order_iterator::pre_order_iterator(const iterator_base& other)
: iterator_base(other.node)
{
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::pre_order_iterator::pre_order_iterator(
    const sibling_iterator& other)
: iterator_base(other.node)
{
    if(this->node == 0)
    {
        if(other.range_last() != 0)
            this->node = other.range_last();
        else
            this->node = other.m_parent;
        this->skip_children();
        ++(*this);
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator&
graph<T, AllocatorT>::pre_order_iterator::operator++()
{
    assert(this->node != 0);
    if(!this->m_skip_current_children && this->node->first_child != 0)
    {
        this->node = this->node->first_child;
    }
    else
    {
        this->m_skip_current_children = false;
        while(this->node->next_sibling == 0)
        {
            this->node = this->node->parent;
            if(this->node == 0)
                return *this;
        }
        this->node = this->node->next_sibling;
    }
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator&
graph<T, AllocatorT>::pre_order_iterator::operator--()
{
    assert(this->node != 0);
    if(this->node->prev_sibling)
    {
        this->node = this->node->prev_sibling;
        while(this->node->last_child)
            this->node = this->node->last_child;
    }
    else
    {
        this->node = this->node->parent;
        if(this->node == 0)
            return *this;
    }
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::pre_order_iterator::operator++(int)
{
    pre_order_iterator copy = *this;
    ++(*this);
    return copy;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::pre_order_iterator::operator--(int)
{
    pre_order_iterator copy = *this;
    --(*this);
    return copy;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator&
graph<T, AllocatorT>::pre_order_iterator::operator+=(unsigned int num)
{
    while(num > 0)
    {
        ++(*this);
        --num;
    }
    return (*this);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator&
graph<T, AllocatorT>::pre_order_iterator::operator-=(unsigned int num)
{
    while(num > 0)
    {
        --(*this);
        --num;
    }
    return (*this);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::pre_order_iterator::operator+(unsigned int num)
{
    auto itr = *this;
    while(num > 0)
    {
        ++itr;
        --num;
    }
    return itr;
}

//--------------------------------------------------------------------------------------//
// Sibling iterator
template <typename T, typename AllocatorT>
graph<T, AllocatorT>::sibling_iterator::sibling_iterator()
: iterator_base()
{
    m_set_parent();
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::sibling_iterator::sibling_iterator(graph_node* tn)
: iterator_base(tn)
{
    m_set_parent();
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::sibling_iterator::sibling_iterator(const iterator_base& other)
: iterator_base(other.node)
{
    m_set_parent();
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::sibling_iterator::sibling_iterator(const sibling_iterator& other)
: iterator_base(other)
, m_parent(other.m_parent)
{
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::sibling_iterator::m_set_parent()
{
    m_parent = 0;
    if(this->node == 0)
        return;
    if(this->node->parent != 0)
        m_parent = this->node->parent;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator&
graph<T, AllocatorT>::sibling_iterator::operator++()
{
    if(this->node)
        this->node = this->node->next_sibling;
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator&
graph<T, AllocatorT>::sibling_iterator::operator--()
{
    if(this->node)
        this->node = this->node->prev_sibling;
    else
    {
        assert(m_parent);
        this->node = m_parent->last_child;
    }
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::sibling_iterator::operator++(int)
{
    sibling_iterator copy = *this;
    ++(*this);
    return copy;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::sibling_iterator::operator--(int)
{
    sibling_iterator copy = *this;
    --(*this);
    return copy;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator&
graph<T, AllocatorT>::sibling_iterator::operator+=(unsigned int num)
{
    while(num > 0)
    {
        ++(*this);
        --num;
    }
    return (*this);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator&
graph<T, AllocatorT>::sibling_iterator::operator-=(unsigned int num)
{
    while(num > 0)
    {
        --(*this);
        --num;
    }
    return (*this);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::sibling_iterator::operator+(unsigned int num)
{
    auto itr = *this;
    while(num > 0)
    {
        ++itr;
        --num;
    }
    return itr;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::graph_node*
graph<T, AllocatorT>::sibling_iterator::range_last() const
{
    return m_parent->last_child;
}

//--------------------------------------------------------------------------------------//

template <typename T>
void
print_graph_bracketed(const graph<T>& t, std::ostream& os = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T>
void
print_subgraph_bracketed(const graph<T>& t, typename graph<T>::iterator root,
                         std::ostream& os = std::cout);

//--------------------------------------------------------------------------------------//
// Iterate over all roots (the head) and print each one on a new line
// by calling printSingleRoot.
//--------------------------------------------------------------------------------------//

template <typename T>
void
print_graph_bracketed(const graph<T>& t, std::ostream& os)
{
    int head_count = t.number_of_siblings(t.begin());
    int nhead      = 0;
    for(typename graph<T>::sibling_iterator ritr = t.begin(); ritr != t.end();
        ++ritr, ++nhead)
    {
        print_subgraph_bracketed(t, ritr, os);
        if(nhead != head_count)
        {
            os << std::endl;
        }
    }
}

//--------------------------------------------------------------------------------------//
// Print everything under this root in a flat, bracketed structure.
//--------------------------------------------------------------------------------------//

template <typename T>
void
print_subgraph_bracketed(const graph<T>& t, typename graph<T>::iterator root,
                         std::ostream& os)
{
    if(t.empty())
        return;
    if(t.number_of_children(root) == 0)
    {
        os << *root;
    }
    else
    {
        // parent
        os << *root;
        os << "(";
        // child1, ..., childn
        int sibling_count = t.number_of_siblings(t.begin(root));
        int nsiblings;
        typename graph<T>::sibling_iterator children;
        for(children = t.begin(root), nsiblings = 0; children != t.end(root);
            ++children, ++nsiblings)
        {
            // recursively print child
            print_subgraph_bracketed(t, children, os);
            // comma after every child except the last one
            if(nsiblings != sibling_count)
            {
                os << ", ";
            }
        }
        os << ")";
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter = std::function<std::string(const T&)>>
void
print_graph(const tim::graph<T>& t, Formatter format, std::ostream& str = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter = std::function<std::string(const T&)>>
void
print_subgraph(const tim::graph<T>& t, Formatter format,
               typename tim::graph<T>::iterator root, std::ostream& str = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter>
void
print_graph(const tim::graph<T>& t, Formatter format, std::ostream& str)
{
    int head_count = t.number_of_siblings(t.begin());
    int nhead      = 0;
    for(typename tim::graph<T>::sibling_iterator ritr = t.begin(); ritr != t.end();
        ++ritr, ++nhead)
    {
        print_subgraph(t, format, ritr, str);
        if(nhead != head_count)
        {
            str << std::endl;
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter>
void
print_subgraph(const tim::graph<T>& t, Formatter format,
               typename tim::graph<T>::iterator root, std::ostream& os)
{
    if(t.empty())
        return;
    if(t.number_of_children(root) == 0)
    {
        os << format(*root);
    }
    else
    {
        // parent
        std::string str = format(*root);
        if(str.length() > 0)
            os << str << "\n";
        // child1, ..., childn
        int sibling_count = t.number_of_siblings(t.begin(root));
        int nsiblings;
        typename tim::graph<T>::sibling_iterator children;
        for(children = t.begin(root), nsiblings = 0; children != t.end(root);
            ++children, ++nsiblings)
        {
            // recursively print child
            print_subgraph(t, format, children, os);
            // comma after every child except the last one
            if(nsiblings != sibling_count)
            {
                os << "\n";
            }
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter = std::function<std::string(const T&)>>
void
print_graph_hierarchy(const tim::graph<T>& t, Formatter format,
                      std::ostream& str = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter = std::function<std::string(const T&)>>
void
print_subgraph_hierarchy(const tim::graph<T>& t, Formatter format,
                         typename tim::graph<T>::iterator root,
                         std::ostream&                    str = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter>
void
print_graph_hierarchy(const tim::graph<T>& t, Formatter format, std::ostream& str)
{
    int head_count = t.number_of_siblings(t.begin());
    int nhead      = 0;
    for(typename tim::graph<T>::sibling_iterator ritr = t.begin(); ritr != t.end();
        ++ritr, ++nhead)
    {
        print_subgraph_hierarchy(t, format, ritr, str);
        if(nhead != head_count)
        {
            str << std::endl;
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter>
void
print_subgraph_hierarchy(const tim::graph<T>& t, Formatter format,
                         typename tim::graph<T>::iterator root, std::ostream& os)
{
    if(t.empty())
        return;
    if(t.number_of_children(root) == 0)
    {
        os << format(*root);
    }
    else
    {
        // parent
        std::string str = format(*root);
        if(str.length() > 0)
            os << str << "\n" << std::setw(2 * (t.depth(root) + 1)) << "|_";
        // child1, ..., childn
        int sibling_count = t.number_of_siblings(t.begin(root));
        int nsiblings;
        typename tim::graph<T>::sibling_iterator children;
        for(children = t.begin(root), nsiblings = 0; children != t.end(root);
            ++children, ++nsiblings)
        {
            // recursively print child
            print_subgraph_hierarchy(t, format, children, os);
            // comma after every child except the last one
            if(nsiblings != sibling_count)
            {
                os << "\n" << std::setw(2 * (t.depth(root) + 1)) << "|_";
            }
        }
    }
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
