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

/** \file component_hybrid.hpp
 * \headerfile component_hybrid.hpp "timemory/variadic/component_hybrid.hpp"
 * This is the C++ class that bundles together components and enables
 * operation on the components as a single entity
 *
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include "timemory/variadic/component_list.hpp"
#include "timemory/variadic/component_tuple.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//
// forward declaration
//
template <typename _CompTuple, typename _CompList>
class auto_hybrid;

//======================================================================================//
// variadic list of components
//
template <typename _CompTuple, typename _CompList>
class component_hybrid
{
    static const std::size_t num_elements = _CompTuple::size() + _CompList::size();
    // empty init for friends
    explicit component_hybrid() {}
    // manager is friend so can use above
    friend class manager;

public:
    using size_type       = int64_t;
    using language_t      = tim::language;
    using string_hash     = std::hash<string_t>;
    using tuple_type      = _CompTuple;
    using list_type       = _CompList;
    using this_type       = component_hybrid<tuple_type, list_type>;
    using tuple_data_type = typename tuple_type::data_type;
    using list_data_type  = typename list_type::data_type;
    using data_type       = tim::impl::tuple_concat<tuple_data_type, list_data_type>;

public:
    using auto_type = auto_hybrid<tuple_type, list_type>;

public:
    explicit component_hybrid(const string_t& key, const bool& store,
                              const int64_t& ncount = 0, const int64_t& nhash = 0,
                              const language_t& lang = language_t::cxx())
    : m_tuple(key, store, ncount, nhash, lang)
    , m_list(key, store, ncount, nhash, lang)
    {
        m_tuple.m_print_laps  = false;
        m_list.m_print_laps   = false;
        m_list.m_print_prefix = false;
    }

    explicit component_hybrid(const string_t& key, const bool& store,
                              const language_t& lang, const int64_t& ncount = 0,
                              const int64_t& nhash = 0)
    : m_tuple(key, store, lang, ncount, nhash)
    , m_list(key, store, lang, ncount, nhash)
    {
        m_tuple.m_print_laps  = false;
        m_list.m_print_laps   = false;
        m_list.m_print_prefix = false;
    }

    explicit component_hybrid(const string_t&   key,
                              const language_t& lang = language_t::cxx(),
                              const int64_t& ncount = 0, const int64_t& nhash = 0,
                              bool store = true)
    : m_tuple(key, lang, ncount, nhash, store)
    , m_list(key, lang, ncount, nhash, store)
    {
        m_tuple.m_print_laps  = false;
        m_list.m_print_laps   = false;
        m_list.m_print_prefix = false;
    }

    ~component_hybrid() {}

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_hybrid(const tuple_type& _tuple, const list_type& _list)
    : m_tuple(_tuple)
    , m_list(_list)
    {
    }

    component_hybrid(const component_hybrid&) = default;
    component_hybrid(component_hybrid&&)      = default;

    component_hybrid& operator=(const component_hybrid& rhs) = default;
    component_hybrid& operator=(component_hybrid&&) = default;

    component_hybrid clone(const int64_t& nhash, bool store)
    {
        return component_hybrid(m_tuple.clone(nhash, store), m_list.clone(nhash, store));
    }

public:
    tuple_type&       get_tuple() { return m_tuple; }
    const tuple_type& get_tuple() const { return m_tuple; }
    list_type&        get_list() { return m_list; }
    const list_type&  get_list() const { return m_list; }

    tuple_type&       get_first() { return m_tuple; }
    const tuple_type& get_first() const { return m_tuple; }
    list_type&        get_second() { return m_list; }
    const list_type&  get_second() const { return m_list; }

    tuple_type&       get_lhs() { return m_tuple; }
    const tuple_type& get_lhs() const { return m_tuple; }
    list_type&        get_rhs() { return m_list; }
    const list_type&  get_rhs() const { return m_list; }

public:
    //----------------------------------------------------------------------------------//
    // get the size
    //
    static constexpr std::size_t size() { return num_elements; }

    //----------------------------------------------------------------------------------//
    // insert into graph
    inline void push()
    {
        m_tuple.push();
        m_list.push();
    }

    //----------------------------------------------------------------------------------//
    // pop out of graph
    inline void pop()
    {
        m_tuple.pop();
        m_list.pop();
    }

    //----------------------------------------------------------------------------------//
    // measure functions
    void measure()
    {
        m_tuple.measure();
        m_list.measure();
    }

    //----------------------------------------------------------------------------------//
    // start/stop functions
    void start()
    {
        m_tuple.start();
        m_list.start();
    }

    void stop()
    {
        m_tuple.stop();
        m_list.stop();
    }

    void conditional_start()
    {
        m_tuple.conditional_start();
        m_list.conditional_start();
    }

    void conditional_stop()
    {
        m_tuple.conditional_stop();
        m_list.conditional_stop();
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    void mark_begin()
    {
        m_tuple.mark_begin();
        m_list.mark_begin();
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    void mark_end()
    {
        m_tuple.mark_end();
        m_list.mark_end();
    }

    //----------------------------------------------------------------------------------//
    // recording
    //
    this_type& record()
    {
        m_tuple.record();
        m_list.record();
        return *this;
    }

    //----------------------------------------------------------------------------------//
    void reset()
    {
        m_tuple.reset();
        m_list.reset();
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        m_tuple -= rhs.m_tuple;
        m_list -= rhs.m_list;
        return *this;
    }

    this_type& operator-=(this_type& rhs)
    {
        m_tuple -= rhs.m_tuple;
        m_list -= rhs.m_list;
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        m_tuple += rhs.m_tuple;
        m_list += rhs.m_list;
        return *this;
    }

    this_type& operator+=(this_type& rhs)
    {
        m_tuple += rhs.m_tuple;
        m_list += rhs.m_list;
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // generic operators
    //
    template <typename _Op>
    this_type& operator-=(_Op&& rhs)
    {
        m_tuple -= std::forward<_Op>(rhs);
        m_list -= std::forward<_Op>(rhs);
        return *this;
    }

    template <typename _Op>
    this_type& operator+=(_Op&& rhs)
    {
        m_tuple += std::forward<_Op>(rhs);
        m_list += std::forward<_Op>(rhs);
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        m_tuple *= std::forward<_Op>(rhs);
        m_list *= std::forward<_Op>(rhs);
        return *this;
    }

    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        m_tuple /= std::forward<_Op>(rhs);
        m_list /= std::forward<_Op>(rhs);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // friend operators
    //
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        tmp.m_tuple += rhs.m_tuple;
        tmp.m_list += rhs.m_list;
        return tmp;
    }

    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        tmp.m_tuple -= rhs.m_tuple;
        tmp.m_list -= rhs.m_list;
        return tmp;
    }

    template <typename _Op>
    friend this_type operator*(const this_type& lhs, _Op&& rhs)
    {
        this_type tmp(lhs);
        tmp.m_tuple *= std::forward<_Op>(rhs);
        tmp.m_list *= std::forward<_Op>(rhs);
        return tmp;
    }

    template <typename _Op>
    friend this_type operator/(const this_type& lhs, _Op&& rhs)
    {
        this_type tmp(lhs);
        tmp.m_tuple /= std::forward<_Op>(rhs);
        tmp.m_list /= std::forward<_Op>(rhs);
        return tmp;
    }

    //----------------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        std::stringstream tss, lss;

        tss << obj.m_tuple;
        lss << obj.m_list;

        if(tss.str().length() > 0)
            os << tss.str();
        if(tss.str().length() > 0 && lss.str().length() > 0)
            os << ", ";
        if(lss.str().length() > 0)
            os << lss.str();

        if(obj.m_tuple.laps() > 0)
            os << " [laps: " << obj.m_tuple.laps() << "]";

        return os;
    }

    //----------------------------------------------------------------------------------//
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        m_tuple.serialize(ar, version);
        m_list.serialize(ar, version);
        /*
        using apply_types = std::tuple<operation::serialization<Types, Archive>...>;
        ar(serializer::make_nvp("identifier", m_identifier),
           serializer::make_nvp("laps", m_laps));
        ar.setNextName("data");
        ar.startNode();
        apply<void>::access<apply_types>(m_data, std::ref(ar), version);
        ar.finishNode();
        */
    }

    //----------------------------------------------------------------------------------//
    inline void report(std::ostream& os, bool endline, bool ign_cutoff) const
    {
        m_tuple.report(os, endline, ign_cutoff);
        m_list.report(os, endline, ign_cutoff);
    }

    //----------------------------------------------------------------------------------//
    static void print_storage()
    {
        tuple_type::print_storage();
        list_type::print_storage();
    }

public:
    //----------------------------------------------------------------------------------//
    template <typename _Tp, typename _Func, typename... _Args>
    void type_apply(_Func&& _func, _Args&&... _args)
    {
        m_tuple.template type_apply<_Tp>(_func, std::forward<_Args>(_args)...);
        m_list.template type_apply<_Tp>(_func, std::forward<_Args>(_args)...);
    }

protected:
    // protected member functions
    data_type get_data() const { return std::tuple_cat(m_tuple.m_data, m_list.m_data); }

protected:
    // objects
    tuple_type m_tuple;
    list_type  m_list;

protected:
    string_t get_prefix() { return m_tuple.get_prefix(); }

    void compute_identifier(const string_t& key, const language_t& lang)
    {
        m_tuple.compute_identifer(key, lang);
        m_list.compute_identifer(key, lang);
    }

    void update_identifier() const
    {
        m_tuple.update_identifier();
        m_list.update_identifier();
    }

    static int64_t output_width(int64_t width = 0)
    {
        return std::max<int64_t>(tuple_type::output_width(width),
                                 list_type::output_width(width));
    }

private:
    template <typename _Func, typename... _Args>
    void apply_to_members(_Func&& _func, _Args&&... _args)
    {
        ((m_tuple).*(_func))(std::forward<_Args>(_args)...);
        ((m_list).*(_func))(std::forward<_Args>(_args)...);
    }
};

}  // namespace tim

//--------------------------------------------------------------------------------------//
