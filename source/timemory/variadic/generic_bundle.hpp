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

/** \file timemory/variadic/generic_bundle.hpp
 * \headerfile variadic/generic_bundle.hpp "timemory/variadic/generic_bundle.hpp"
 * This holds the bundle of data for a component_tuple or component_list
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

#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

//======================================================================================//
//
namespace tim
{
//======================================================================================//
//
template <typename... Types>
class generic_bundle
{
public:
    template <typename... T>
    struct bundle;
    template <typename C, typename A, typename T>
    struct bundle_definitions;
    template <typename T>
    struct generic_counter;
    template <typename T>
    struct generic_deleter;
    template <template <typename> class Op, typename... T>
    struct generic_operation;

    using EmptyT = std::tuple<>;
    template <typename U>
    using sample_type_t =
        conditional_t<trait::sampler<U>::value, operation::sample<U>, EmptyT>;
    /*
    template <typename T>
    struct bundle<T>
    {
        using data_type      = T;
        using type_tuple     = remove_pointer_t<T>;
        using reference_type = remove_pointer_t<T>;
        using pointer_type   = add_pointer_t<T>;
        using sample_type    = sample_type_t<T>;

        using print_t = operation::print<T>;

        template <typename _Archive>
        using serialize_t =
            operation::generic_operator<T, operation::serialization<T, _Archive>>;
    };

    template <typename T>
    struct bundle<std::tuple<T>> : bundle<T>
    {
        using base_type      = bundle<T>;
        using data_type      = std::tuple<typename base_type::data_type>;
        using type_tuple     = std::tuple<typename base_type::type_tuple>;
        using reference_type = std::tuple<typename base_type::reference_type>;
        using pointer_type   = std::tuple<typename base_type::pointer_type>;
        using sample_type    = std::tuple<typename base_type::sample_type>;
        using print_t        = std::tuple<typename base_type::print_t>;
        template <typename _Archive>
        using serialize_t =
            std::tuple<typename base_type::template serialize_t<_Archive>>;
    };

    template <typename T>
    struct bundle<type_list<T>> : bundle<T>
    {
        using base_type      = bundle<T>;
        using data_type      = typename base_type::data_type;
        using type_tuple     = typename base_type::type_tuple;
        using reference_type = typename base_type::reference_type;
        using pointer_type   = typename base_type::pointer_type;
        using sample_type    = typename base_type::sample_type;
        using print_t        = typename base_type::print_t;
        template <typename _Archive>
        using serialize_t = typename base_type::template serialize_t<_Archive>;
    };

    template <template <typename...> class TypeL, typename... T>
    struct bundle<TypeL<T...>>
    {
        using data_type      = TypeL<typename bundle<T>::data_type...>;
        using type_tuple     = TypeL<typename bundle<T>::type_tuple...>;
        using reference_type = TypeL<typename bundle<T>::reference_type...>;
        using pointer_type   = TypeL<typename bundle<T>::pointer_type...>;
        using sample_type    = TypeL<typename bundle<T>::sample_type...>;
        using print_t        = TypeL<typename bundle<T>::print_t...>;
        template <typename _Archive>
        using serialize_t = TypeL<typename bundle<T>::template serialize_t<_Archive>...>;
    };
    */
    template <template <typename...> class TypeL, typename... T>
    struct bundle<TypeL<T...>>
    {
        using data_type      = TypeL<T...>;
        using type_tuple     = TypeL<remove_pointer_t<T>...>;
        using reference_type = TypeL<remove_pointer_t<T>...>;
        using pointer_type   = TypeL<add_pointer_t<T>...>;
        using sample_type    = TypeL<sample_type_t<T>...>;
        using print_t        = TypeL<operation::print<T>...>;
        template <typename _Archive>
        using serialize_t = TypeL<
                            operation::generic_operator<T,
                                                        operation::serialization<T,
                                                                 _Archive>>...>;
    };


    template <template <typename...> class CompL, template <typename...> class AutoL,
              template <typename...> class DataL, typename... L, typename... T>
    struct bundle_definitions<CompL<L...>, AutoL<L...>, DataL<T...>>
    {
        using component_type = CompL<T...>;
        using auto_type      = AutoL<T...>;
    };

    template <typename C, typename A, typename T>
    using component_type_definition_t =
        typename bundle_definitions<C, A, T>::component_type;

    template <typename C, typename A, typename T>
    using auto_type_definition_t = typename bundle_definitions<C, A, T>::auto_type;

    template <template <typename> class Op, template <typename...> class TypeL,
              typename... T>
    struct generic_operation<Op, TypeL<T...>>
    {
        using type = TypeL<operation::generic_operator<T, Op<T>>...>;
    };

    template <template <typename> class Op, typename... T>
    struct generic_operation
    {
        using type = std::tuple<operation::generic_operator<T, Op<T>>...>;
    };

    template <template <typename...> class TypeL, typename... T>
    struct generic_counter<TypeL<T...>>
    {
        using type = TypeL<operation::generic_counter<T>...>;
    };

    template <template <typename...> class TypeL, typename... T>
    struct generic_deleter<TypeL<T...>>
    {
        using type = TypeL<operation::generic_deleter<T>...>;
    };

public:
    using size_type   = int64_t;
    using string_t    = std::string;
    using string_hash = std::hash<string_t>;

    using impl_type       = available_tuple<concat<Types...>>;
    using type_bundler    = bundle<impl_type>;
    using data_type       = typename type_bundler::data_type;
    using sample_type     = typename type_bundler::sample_type;
    using type_tuple      = typename type_bundler::type_tuple;
    using pointer_type    = typename type_bundler::pointer_type;
    using reference_type  = typename type_bundler::reference_type;
    using data_value_type = get_data_value_t<impl_type>;
    using data_label_type = get_data_label_t<impl_type>;

    // used by gotcha component to prevent recursion
    using gotcha_components = typename get_true_types<trait::is_gotcha, impl_type>::type;
    static constexpr bool contains_gotcha =
        (mpl::get_tuple_size<gotcha_components>::value != 0);

public:
    template <template <typename> class Op, typename _Tuple = impl_type>
    using operation_t = typename generic_operation<Op, _Tuple>::type;

    template <typename _Tuple = impl_type>
    using deleter_t = typename generic_deleter<_Tuple>::type;

    template <typename _Tuple = impl_type>
    using counter_t = typename generic_counter<_Tuple>::type;

public:
    template <typename _Archive>
    using serialize_t  = typename type_bundler::template serialize_t<_Archive>;
    using push_node_t  = operation_t<operation::insert_node, data_type>;
    using pop_node_t   = operation_t<operation::pop_node, data_type>;
    using measure_t    = operation_t<operation::measure, data_type>;
    using record_t     = operation_t<operation::record, data_type>;
    using reset_t      = operation_t<operation::reset, data_type>;
    using plus_t       = operation_t<operation::plus, data_type>;
    using minus_t      = operation_t<operation::minus, data_type>;
    using multiply_t   = operation_t<operation::multiply, data_type>;
    using divide_t     = operation_t<operation::divide, data_type>;
    using print_t      = typename type_bundler::print_t;
    using mark_begin_t = operation_t<operation::mark_begin, data_type>;
    using mark_end_t   = operation_t<operation::mark_end, data_type>;
    using construct_t  = operation_t<operation::construct, data_type>;
    using audit_t      = operation_t<operation::audit, data_type>;
    using set_prefix_t = operation_t<operation::set_prefix, data_type>;
    using get_data_t   = operation_t<operation::get_data, data_type>;
    using copy_t       = operation_t<operation::copy, data_type>;

public:
    explicit generic_bundle(uint64_t _hash = 0, bool _store = settings::enabled(),
                            bool _flat = settings::flat_profile())
    : m_store(_store && settings::enabled())
    , m_flat(_flat)
    , m_is_pushed(false)
    , m_laps(0)
    , m_hash(_hash)
    {}

    ~generic_bundle()                     = default;
    generic_bundle(const generic_bundle&) = default;
    generic_bundle(generic_bundle&&)      = default;
    generic_bundle& operator=(const generic_bundle&) = default;
    generic_bundle& operator=(generic_bundle&&) = default;

    //----------------------------------------------------------------------------------//
    //
    inline int64_t laps() const { return m_laps; }

    //----------------------------------------------------------------------------------//
    //
    inline std::string key() const { return get_hash_ids()->find(m_hash)->second; }

    //----------------------------------------------------------------------------------//
    //
    inline uint64_t hash() const { return m_hash; }

    //----------------------------------------------------------------------------------//
    //
    inline void rekey(const string_t& _key)
    {
        m_hash = add_hash_id(_key);
        compute_width(_key);
    }

    //----------------------------------------------------------------------------------//
    //
    inline bool& store() { return m_store; }

    //----------------------------------------------------------------------------------//
    //
    inline const bool& store() const { return m_store; }

    //----------------------------------------------------------------------------------//
    //
    inline const string_t& prefix() const
    {
        auto _get_prefix = []() {
            if(!dmp::is_initialized())
                return string_t(">>> ");

            // prefix spacing
            static uint16_t width = 1;
            if(dmp::size() > 9)
                width = std::max(width, (uint16_t)(log10(dmp::size()) + 1));
            std::stringstream ss;
            ss.fill('0');
            ss << "|" << std::setw(width) << dmp::rank() << ">>> ";
            return ss.str();
        };
        static string_t _prefix = _get_prefix();
        return _prefix;
    }

    //----------------------------------------------------------------------------------//
    //
    inline const string_t& get_prefix() const { return prefix(); }

protected:
    //----------------------------------------------------------------------------------//
    //
    static int64_t output_width(int64_t width = 0)
    {
        static auto                 memorder_v = std::memory_order_relaxed;
        static std::atomic<int64_t> _instance(0);
        int64_t                     propose_width, current_width;
        auto compute = [&]() { return std::max(_instance.load(memorder_v), width); };
        while((propose_width = compute()) > (current_width = _instance.load(memorder_v)))
        {
            _instance.compare_exchange_strong(current_width, propose_width, memorder_v);
        }
        return _instance.load(memorder_v);
    }

    //----------------------------------------------------------------------------------//
    //
    inline void compute_width(const string_t& _key) const
    {
        static const string_t& _prefix = get_prefix();
        output_width(_key.length() + _prefix.length() + 1);
    }

    //----------------------------------------------------------------------------------//
    //
    inline void update_width() const { compute_width(key()); }

protected:
    // objects
    bool     m_store     = false;
    bool     m_flat      = false;
    bool     m_is_pushed = false;
    int64_t  m_laps      = 0;
    uint64_t m_hash      = 0;
};

//======================================================================================//

template <typename... Types>
class generic_bundle<std::tuple<Types...>> : public generic_bundle<Types...>
{
public:
    using bundle_type     = generic_bundle<Types...>;
    using data_type       = typename bundle_type::data_type;
    using impl_type       = typename bundle_type::impl_type;
    using type_tuple      = typename bundle_type::type_tuple;
    using sample_type     = typename bundle_type::sample_type;
    using pointer_type    = typename bundle_type::pointer_type;
    using reference_type  = typename bundle_type::reference_type;
    using data_value_type = typename bundle_type::data_value_type;
    using data_label_type = typename bundle_type::data_label_type;

    using apply_v     = apply<void>;
    using size_type   = typename bundle_type::size_type;
    using string_t    = typename bundle_type::string_t;
    using string_hash = typename bundle_type::string_hash;

    template <template <typename> class Op, typename _Tuple = impl_type>
    using operation_t =
        typename bundle_type::template generic_operation<Op, _Tuple>::type;

public:
    explicit generic_bundle(uint64_t _hash = 0, bool _store = settings::enabled(),
                            bool _flat = settings::flat_profile())
    : bundle_type(_hash, _store, _flat)
    {}

    ~generic_bundle()                     = default;
    generic_bundle(const generic_bundle&) = default;
    generic_bundle(generic_bundle&&)      = default;
    generic_bundle& operator=(const generic_bundle&) = default;
    generic_bundle& operator=(generic_bundle&&) = default;

public:
    int64_t         laps() const { return bundle_type::laps(); }
    std::string     key() const { return bundle_type::key(); }
    uint64_t        hash() const { return bundle_type::hash(); }
    void            rekey(const string_t& _key) { bundle_type::rekey(_key); }
    bool&           store() { return bundle_type::store(); }
    const bool&     store() const { return bundle_type::store(); }
    const string_t& prefix() const { return bundle_type::get_prefix(); }
    const string_t& get_prefix() const { return bundle_type::get_prefix(); }

protected:
    static int64_t output_width(int64_t w = 0) { return bundle_type::output_width(w); }
    void           update_width() const { bundle_type::update_width(); }
    void compute_width(const string_t& _key) const { bundle_type::compute_width(_key); }
};

//======================================================================================//

template <typename... Types>
class generic_bundle<type_list<Types...>> : public generic_bundle<Types...>
{
public:
    using bundle_type     = generic_bundle<Types...>;
    using data_type       = typename bundle_type::data_type;
    using impl_type       = typename bundle_type::impl_type;
    using type_tuple      = typename bundle_type::type_tuple;
    using sample_type     = typename bundle_type::sample_type;
    using pointer_type    = typename bundle_type::pointer_type;
    using reference_type  = typename bundle_type::reference_type;
    using data_value_type = typename bundle_type::data_value_type;
    using data_label_type = typename bundle_type::data_label_type;

    using apply_v     = apply<void>;
    using size_type   = typename bundle_type::size_type;
    using string_t    = typename bundle_type::string_t;
    using string_hash = typename bundle_type::string_hash;

    template <template <typename> class Op, typename _Tuple = impl_type>
    using operation_t =
        typename bundle_type::template generic_operation<Op, _Tuple>::type;

public:
    explicit generic_bundle(uint64_t _hash = 0, bool _store = settings::enabled(),
                            bool _flat = settings::flat_profile())
    : bundle_type(_hash, _store, _flat)
    {}

    ~generic_bundle()                     = default;
    generic_bundle(const generic_bundle&) = default;
    generic_bundle(generic_bundle&&)      = default;
    generic_bundle& operator=(const generic_bundle&) = default;
    generic_bundle& operator=(generic_bundle&&) = default;

public:
    int64_t         laps() const { return bundle_type::laps(); }
    std::string     key() const { return bundle_type::key(); }
    uint64_t        hash() const { return bundle_type::hash(); }
    void            rekey(const string_t& _key) { bundle_type::rekey(_key); }
    bool&           store() { return bundle_type::store(); }
    const bool&     store() const { return bundle_type::store(); }
    const string_t& prefix() const { return bundle_type::get_prefix(); }
    const string_t& get_prefix() const { return bundle_type::get_prefix(); }

protected:
    static int64_t output_width(int64_t w = 0) { return bundle_type::output_width(w); }
    void           update_width() const { bundle_type::update_width(); }
    void compute_width(const string_t& _key) const { bundle_type::compute_width(_key); }
};

}  // namespace tim
