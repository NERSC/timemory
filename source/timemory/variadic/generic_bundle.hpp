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

    template <template <typename...> class TypeL>
    struct bundle<TypeL<>>
    {
        using data_type      = TypeL<>;
        using type_tuple     = TypeL<>;
        using reference_type = TypeL<>;
        using sample_type    = TypeL<>;
        using print_t        = TypeL<>;
        template <typename _Archive>
        using serialize_t = TypeL<>;
    };

    template <template <typename...> class TypeL, typename T, typename... Tail>
    struct bundle<TypeL<T, Tail...>>
    {
        static_assert(!std::is_pointer<T>::value, "Error! pointer pointer!");
        using data_type      = std::tuple<T, Tail...>;
        using type_tuple     = std::tuple<T, Tail...>;
        using reference_type = TypeL<T, Tail...>;
        using sample_type    = TypeL<sample_type_t<T>, sample_type_t<Tail>...>;
        using print_t        = TypeL<operation::print<T>, operation::print<Tail>...>;
        template <typename _Archive>
        using serialize_t =
            TypeL<operation::generic_operator<T, operation::serialization<T, _Archive>>,
                  operation::generic_operator<
                      Tail, operation::serialization<Tail, _Archive>>...>;
    };

    template <template <typename...> class TypeL, typename T, typename... Tail>
    struct bundle<TypeL<T*, Tail*...>>
    {
        static_assert(!std::is_pointer<T>::value, "Error! pointer pointer!");
        using data_type      = std::tuple<T*, Tail*...>;
        using type_tuple     = std::tuple<T, Tail...>;
        using reference_type = TypeL<T, Tail...>;
        using sample_type    = TypeL<sample_type_t<T>, sample_type_t<Tail>...>;
        using print_t        = TypeL<operation::print<T>, operation::print<Tail>...>;
        template <typename _Archive>
        using serialize_t =
            TypeL<operation::generic_operator<T*, operation::serialization<T, _Archive>>,
                  operation::generic_operator<
                      Tail*, operation::serialization<Tail, _Archive>>...>;
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

    template <template <typename> class Op, template <typename...> class TypeL>
    struct generic_operation<Op, TypeL<>>
    {
        using type = TypeL<>;
    };

    template <template <typename> class Op, template <typename...> class TypeL,
              typename... T>
    struct generic_operation<Op, TypeL<T...>>
    {
        using type = TypeL<operation::generic_operator<T, Op<T>>...>;
    };

    template <template <typename> class Op, template <typename...> class TypeL,
              typename... T>
    struct generic_operation<Op, TypeL<T*...>>
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

    using impl_type      = std::tuple<Types...>;
    using type_bundler   = bundle<impl_type>;
    using data_type      = typename type_bundler::data_type;
    using sample_type    = typename type_bundler::sample_type;
    using type_tuple     = typename type_bundler::type_tuple;
    using reference_type = typename type_bundler::reference_type;
    using data_collect_type =
        typename get_true_types<trait::collects_data, impl_type>::type;
    using data_value_type = get_data_value_t<data_collect_type>;
    using data_label_type = get_data_label_t<data_collect_type>;

    // used by gotcha component to prevent recursion
    using gotcha_types = typename get_true_types<trait::is_gotcha, impl_type>::type;
    static constexpr bool has_gotcha_v = (mpl::get_tuple_size<gotcha_types>::value != 0);

    using user_bundle_types =
        typename get_true_types<trait::is_user_bundle, impl_type>::type;
    static constexpr bool has_user_bundle_v =
        (mpl::get_tuple_size<user_bundle_types>::value != 0);

public:
    template <template <typename> class Op, typename _Tuple = data_type>
    using operation_t = typename generic_operation<Op, _Tuple>::type;

    template <typename _Tuple = impl_type>
    using deleter_t = typename generic_deleter<_Tuple>::type;

    template <typename _Tuple = impl_type>
    using counter_t = typename generic_counter<_Tuple>::type;

public:
    template <typename _Archive>
    using serialize_t = typename type_bundler::template serialize_t<_Archive>;
    using print_t     = typename type_bundler::print_t;

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
    inline const string_t& prefix() const { return get_persistent_data().prefix; }

    //----------------------------------------------------------------------------------//
    //
    inline const string_t& get_prefix() const { return prefix(); }

protected:
    //----------------------------------------------------------------------------------//
    //
    static int64_t output_width(int64_t width = 0)
    {
        return get_persistent_data().get_width(width);
    }

    //----------------------------------------------------------------------------------//
    //
    inline void compute_width(const string_t& _key) const
    {
        output_width(_key.length() + get_prefix().length() + 1);
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

protected:
    struct persistent_data
    {
        int64_t get_width(int64_t _w)
        {
            auto&&  memorder_v = std::memory_order_relaxed;
            int64_t propose_width, current_width;
            auto    compute = [&]() { return std::max(width.load(memorder_v), _w); };
            while((propose_width = compute()) > (current_width = width.load(memorder_v)))
            {
                width.compare_exchange_strong(current_width, propose_width, memorder_v);
            }
            return width.load(memorder_v);
        }

        std::atomic<int64_t> width{ 0 };
        string_t             prefix = []() {
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
        }();
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }
};

//======================================================================================//

template <typename... Types>
class generic_bundle<std::tuple<Types...>> : public generic_bundle<Types...>
{
    template <typename... Args>
    generic_bundle(Args&&... args)
    : generic_bundle<Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
class generic_bundle<type_list<Types...>> : public generic_bundle<Types...>
{
    template <typename... Args>
    generic_bundle(Args&&... args)
    : generic_bundle<Types...>(std::forward<Args>(args)...)
    {}
};

//======================================================================================//
//
template <typename... Types>
struct stack_bundle : public generic_bundle<Types...>
{
    template <typename... Args>
    stack_bundle(Args&&... args)
    : generic_bundle<Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct stack_bundle<std::tuple<Types...>> : public generic_bundle<Types...>
{
    template <typename... Args>
    stack_bundle(Args&&... args)
    : generic_bundle<Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct stack_bundle<type_list<Types...>> : public generic_bundle<Types...>
{
    template <typename... Args>
    stack_bundle(Args&&... args)
    : generic_bundle<Types...>(std::forward<Args>(args)...)
    {}
};

//======================================================================================//
//
template <typename... Types>
struct heap_bundle : public generic_bundle<Types*...>
{
    template <typename... Args>
    heap_bundle(Args&&... args)
    : generic_bundle<Types*...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct heap_bundle<std::tuple<Types...>> : public generic_bundle<Types*...>
{
    template <typename... Args>
    heap_bundle(Args&&... args)
    : generic_bundle<Types*...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct heap_bundle<type_list<Types...>> : public generic_bundle<Types*...>
{
    template <typename... Args>
    heap_bundle(Args&&... args)
    : generic_bundle<Types*...>(std::forward<Args>(args)...)
    {}
};

//======================================================================================//

}  // namespace tim
