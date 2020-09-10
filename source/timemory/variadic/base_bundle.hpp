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

/** \headerfile "timemory/variadic/base_bundle.hpp"
 * This is the generic base class for a variadic bundle of components
 *
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include "timemory/components/base/types.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/runtime/types.hpp"
#include "timemory/variadic/types.hpp"

//======================================================================================//
//
namespace tim
{
template <typename T>
struct pointer
{
    using type = T*;
};

template <typename T>
struct pointer<T*> : pointer<T>
{};

//======================================================================================//
//
/// \class base_bundle
/// \brief This is the generic structure for a variadic bundle of components
///
template <typename Tag, typename... Types>
class base_bundle
{
public:
    using tag_type = Tag;

    template <typename... T>
    struct bundle;
    template <typename C, typename A, typename T>
    struct bundle_definitions;
    template <template <typename> class Op, typename... T>
    struct generic_operation;
    template <template <typename> class Op, typename... T>
    struct custom_operation;

    using EmptyT = std::tuple<>;
    template <typename U>
    using sample_type_t =
        conditional_t<trait::sampler<U>::value, operation::sample<U>, EmptyT>;

    template <typename... T>
    struct bundle
    {
        using tuple_type     = std::tuple<T...>;
        using reference_type = std::tuple<T...>;
        using sample_type    = std::tuple<sample_type_t<T>...>;
        using print_t        = std::tuple<operation::print<T>...>;
    };

    template <typename... T>
    struct bundle<std::tuple<T...>> : bundle<T...>
    {};

    template <typename... T>
    struct bundle<type_list<T...>> : bundle<T...>
    {};

    template <template <typename...> class CompL, template <typename...> class AutoL,
              template <typename...> class DataL, typename... L, typename... T>
    struct bundle_definitions<CompL<L...>, AutoL<L...>, DataL<T...>>
    {
        using component_type = CompL<T...>;
        using auto_type      = AutoL<T...>;
    };

    //----------------------------------------------------------------------------------//
    //
    template <template <typename> class Op, typename... T>
    struct generic_operation
    {
        using type =
            std::tuple<operation::generic_operator<remove_pointer_t<T>,
                                                   Op<remove_pointer_t<T>>, tag_type>...>;
    };

    template <template <typename> class Op, typename... T>
    struct generic_operation<Op, std::tuple<T...>> : generic_operation<Op, T...>
    {};

    //----------------------------------------------------------------------------------//
    //
    template <template <typename> class Op, typename... T>
    struct custom_operation
    {
        using type = std::tuple<Op<T>...>;
    };

    template <template <typename> class Op, typename... T>
    struct custom_operation<Op, std::tuple<T...>> : custom_operation<Op, T...>
    {};

public:
    using size_type   = int64_t;
    using string_t    = std::string;
    using string_hash = std::hash<string_t>;

    using impl_type      = std::tuple<Types...>;
    using type_bundler   = bundle<impl_type>;
    using sample_type    = typename type_bundler::sample_type;
    using tuple_type     = typename type_bundler::tuple_type;
    using reference_type = typename type_bundler::reference_type;

    // used by gotcha component to prevent recursion
    using gotcha_types = typename get_true_types<trait::is_gotcha, tuple_type>::type;
    static constexpr bool has_gotcha_v = (mpl::get_tuple_size<gotcha_types>::value != 0);

    using user_bundle_types =
        typename get_true_types<trait::is_user_bundle, tuple_type>::type;
    static constexpr bool has_user_bundle_v =
        (mpl::get_tuple_size<user_bundle_types>::value != 0);

public:
    template <template <typename> class Op, typename TupleT = tuple_type>
    using operation_t = typename generic_operation<Op, TupleT>::type;

    template <template <typename> class Op, typename TupleT = tuple_type>
    using custom_operation_t = typename custom_operation<Op, TupleT>::type;

public:
    using print_t = typename type_bundler::print_t;

private:
    template <typename T, typename... U>
    struct quirk_config
    {
        static constexpr bool value =
            (is_one_of<T, contains_one_of_t<quirk::is_config,
                                            type_concat_t<Types..., U...>>>::value ||
             is_one_of<T, type_concat_t<Types..., U...>>::value);
    };

public:
    explicit base_bundle(uint64_t _hash = 0, bool _store = settings::enabled(),
                         scope::config _scope = scope::get_default())
    : m_store(_store && settings::enabled() && get_store_config())
    , m_scope(_scope + get_scope_config())
    , m_hash(_hash)
    {}

    template <typename... T>
    explicit base_bundle(uint64_t hash, bool store, quirk::config<T...>)
    : m_store(store && settings::enabled() && get_store_config<T...>())
    , m_scope(get_scope_config<T...>())
    , m_hash(hash)
    {}

    template <typename... T>
    explicit base_bundle(uint64_t hash, quirk::config<T...>)
    : m_store(settings::enabled() && get_store_config<T...>())
    , m_scope(get_scope_config<T...>())
    , m_hash(hash)
    {}

    ~base_bundle()                      = default;
    base_bundle(const base_bundle&)     = default;
    base_bundle(base_bundle&&) noexcept = default;
    base_bundle& operator=(const base_bundle&) = default;
    base_bundle& operator=(base_bundle&&) noexcept = default;

    //----------------------------------------------------------------------------------//
    //
    int64_t laps() const { return m_laps; }

    //----------------------------------------------------------------------------------//
    //
    std::string key() const { return get_hash_ids()->find(m_hash)->second; }

    //----------------------------------------------------------------------------------//
    //
    uint64_t hash() const { return m_hash; }

    //----------------------------------------------------------------------------------//
    //
    void rekey(const string_t& _key)
    {
        m_hash = add_hash_id(_key);
        compute_width(_key);
    }

    //----------------------------------------------------------------------------------//
    //
    bool&           store() { return m_store; }
    const bool&     store() const { return m_store; }
    const string_t& prefix() const { return get_persistent_data().prefix; }

    //----------------------------------------------------------------------------------//
    //
    bool&           get_store() { return m_store; }
    auto&           get_scope() { return m_scope; }
    const bool&     get_store() const { return m_store; }
    const string_t& get_prefix() const { return prefix(); }
    const auto&     get_scope() const { return m_scope; }

    //----------------------------------------------------------------------------------//

    template <typename... T>
    static auto get_scope_config()
    {
        return scope::config(quirk_config<quirk::flat_scope, T...>::value,
                             quirk_config<quirk::timeline_scope, T...>::value,
                             quirk_config<quirk::tree_scope, T...>::value);
    }

    //----------------------------------------------------------------------------------//

    template <typename... T>
    static auto get_store_config()
    {
        return !quirk_config<quirk::no_store, T...>::value;
    }

protected:
    //----------------------------------------------------------------------------------//
    //
    static int64_t output_width(int64_t width = 0)
    {
        return get_persistent_data().get_width(width);
    }

    //----------------------------------------------------------------------------------//
    //
    void compute_width(const string_t& _key) const
    {
        output_width(_key.length() + get_prefix().length() + 1);
    }

    //----------------------------------------------------------------------------------//
    //
    void update_width() const { compute_width(key()); }

protected:
    // objects
    bool          m_store     = false;
    bool          m_is_pushed = false;
    scope::config m_scope     = scope::get_default();
    int64_t       m_laps      = 0;
    uint64_t      m_hash      = 0;

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
            static uint16_t _width = 1;
            if(dmp::size() > 9)
                _width = std::max(_width, (uint16_t)(log10(dmp::size()) + 1));
            std::stringstream ss;
            ss.fill('0');
            ss << "|" << std::setw(_width) << dmp::rank() << ">>> ";
            return ss.str();
        }();
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }
};
//
//======================================================================================//
//
template <typename Tag, typename... Types>
class base_bundle<Tag, std::tuple<Types...>> : public base_bundle<Tag, Types...>
{
public:
    using data_type = std::tuple<Types...>;

    template <typename... Args>
    base_bundle(Args&&... args)
    : base_bundle<Tag, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename Tag, typename... Types>
class base_bundle<Tag, type_list<Types...>> : public base_bundle<Tag, Types...>
{
public:
    using data_type = std::tuple<Types...>;

    template <typename... Args>
    base_bundle(Args&&... args)
    : base_bundle<Tag, Types...>(std::forward<Args>(args)...)
    {}
};
//
//======================================================================================//
//
template <typename... Types>
struct stack_bundle : public base_bundle<TIMEMORY_API, Types...>
{
    using data_type = std::tuple<Types...>;

    template <typename... Args>
    stack_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct stack_bundle<std::tuple<Types...>> : public base_bundle<TIMEMORY_API, Types...>
{
    using data_type = std::tuple<Types...>;

    template <typename... Args>
    stack_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct stack_bundle<type_list<Types...>> : public base_bundle<TIMEMORY_API, Types...>
{
    using data_type = std::tuple<Types...>;

    template <typename... Args>
    stack_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};
//
//======================================================================================//
//
template <typename... Types>
struct heap_bundle : public base_bundle<TIMEMORY_API, Types...>
{
    using data_type = std::tuple<Types*...>;

    template <typename... Args>
    heap_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct heap_bundle<std::tuple<Types...>> : public base_bundle<TIMEMORY_API, Types...>
{
    using data_type = std::tuple<Types*...>;

    template <typename... Args>
    heap_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct heap_bundle<type_list<Types...>> : public base_bundle<TIMEMORY_API, Types...>
{
    using data_type = std::tuple<Types*...>;

    template <typename... Args>
    heap_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};
//
//======================================================================================//
//
template <typename ApiT, typename... Types>
struct api_bundle
: public conditional_t<(trait::is_available<ApiT>::value),
                       base_bundle<ApiT, remove_pointer_t<Types>...>,
                       base_bundle<ApiT, std::tuple<>>>
{
    using base_bundle_type = conditional_t<(trait::is_available<ApiT>::value),
                                           base_bundle<ApiT, remove_pointer_t<Types>...>,
                                           base_bundle<ApiT, std::tuple<>>>;
    using data_type        = conditional_t<(trait::is_available<ApiT>::value),
                                    std::tuple<Types...>, std::tuple<>>;
    using tuple_type       = data_type;
    using impl_type        = data_type;

    template <typename... Args>
    api_bundle(Args&&... args)
    : base_bundle_type(std::forward<Args>(args)...)
    {}
};

template <typename ApiT, typename... Types>
struct api_bundle<ApiT, std::tuple<Types...>>
: public conditional_t<(trait::is_available<ApiT>::value),
                       base_bundle<ApiT, remove_pointer_t<Types>...>,
                       base_bundle<ApiT, std::tuple<>>>
{
    using base_bundle_type = conditional_t<(trait::is_available<ApiT>::value),
                                           base_bundle<ApiT, remove_pointer_t<Types>...>,
                                           base_bundle<ApiT, std::tuple<>>>;
    using data_type        = conditional_t<(trait::is_available<ApiT>::value),
                                    std::tuple<Types...>, std::tuple<>>;
    using tuple_type       = data_type;
    using impl_type        = data_type;

    template <typename... Args>
    api_bundle(Args&&... args)
    : base_bundle_type(std::forward<Args>(args)...)
    {}
};

template <typename ApiT, typename... Types>
struct api_bundle<ApiT, type_list<Types...>>
: public conditional_t<(trait::is_available<ApiT>::value),
                       base_bundle<ApiT, remove_pointer_t<Types>...>,
                       base_bundle<ApiT, std::tuple<>>>
{
    using base_bundle_type = conditional_t<(trait::is_available<ApiT>::value),
                                           base_bundle<ApiT, remove_pointer_t<Types>...>,
                                           base_bundle<ApiT, std::tuple<>>>;
    using data_type        = conditional_t<(trait::is_available<ApiT>::value),
                                    std::tuple<Types...>, std::tuple<>>;
    using tuple_type       = data_type;
    using impl_type        = data_type;

    template <typename... Args>
    api_bundle(Args&&... args)
    : base_bundle_type(std::forward<Args>(args)...)
    {}
};
//
//======================================================================================//
//
}  // namespace tim
