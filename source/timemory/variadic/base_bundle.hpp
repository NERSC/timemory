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

#include "timemory/components/base/types.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/variadic/functional.hpp"
#include "timemory/variadic/types.hpp"

#include <bitset>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

namespace tim
{
namespace mpl
{
namespace impl
{
template <typename... T>
struct bundle;

template <typename C, typename A, typename T>
struct bundle_definitions;

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
    using print_type     = std::tuple<operation::print<T>...>;
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

template <template <typename> class Op, typename TagT, typename... T>
struct generic_operation;

template <template <typename> class Op, typename... T>
struct custom_operation;

template <template <typename> class Op, typename TagT, typename... T>
struct generic_operation
{
    using type =
        std::tuple<operation::generic_operator<remove_pointer_t<T>,
                                               Op<remove_pointer_t<T>>, TagT>...>;
};

template <template <typename> class Op, typename TagT, typename... T>
struct generic_operation<Op, TagT, std::tuple<T...>> : generic_operation<Op, TagT, T...>
{};

template <template <typename> class Op, typename... T>
struct custom_operation
{
    using type = std::tuple<Op<T>...>;
};

template <template <typename> class Op, typename... T>
struct custom_operation<Op, std::tuple<T...>> : custom_operation<Op, T...>
{};

template <typename... U>
struct quirk_config;

template <typename T, typename... F, typename... U>
struct quirk_config<T, type_list<F...>, U...>
{
    static constexpr bool value =
        is_one_of<T, type_list<F..., U...>>::value ||
        is_one_of<T, contains_one_of_t<quirk::is_config, concat<F..., U...>>>::value;
};

TIMEMORY_HOT_INLINE
bool
global_enabled()
{
    static bool& _value = settings::enabled();
    return _value;
}
}  // namespace impl
}  // namespace mpl
//
namespace operation
{
template <typename Op, typename Tag, typename... Types>
struct generic_operator<quirk::config<Types...>, Op, Tag>
{
    TIMEMORY_DEFAULT_OBJECT(generic_operator)

    template <typename Arg, typename... Args>
    explicit generic_operator(Arg&&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(Args&&...) const
    {}
};
//
template <typename... Types>
struct generic_counter<quirk::config<Types...>>
{
    TIMEMORY_DEFAULT_OBJECT(generic_counter)

    template <typename Arg, typename... Args>
    explicit generic_counter(Arg&&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(Args&&...) const
    {}
};
//
template <typename... Types>
struct generic_deleter<quirk::config<Types...>>
{
    TIMEMORY_DEFAULT_OBJECT(generic_deleter)

    template <typename Arg, typename... Args>
    explicit generic_deleter(Arg&&, Args&&...)
    {}

    template <typename... Args>
    auto operator()(Args&&...) const
    {}
};
//
}  // namespace operation
//
/// \class tim::base_bundle<typename Tag, typename... Types>
/// \brief This is the generic structure for a variadic bundle of components
///
template <typename Tag, typename... Types>
class base_bundle
: public concepts::variadic
, public concepts::wrapper
{
public:
    // public alias section
    using size_type           = int64_t;
    using string_t            = string_view_t;
    using captured_location_t = source_location::captured;

    using tag_type       = Tag;
    using impl_type      = std::tuple<Types...>;
    using sample_type    = typename mpl::impl::bundle<impl_type>::sample_type;
    using tuple_type     = typename mpl::impl::bundle<impl_type>::tuple_type;
    using reference_type = typename mpl::impl::bundle<impl_type>::reference_type;
    using print_type     = typename mpl::impl::bundle<impl_type>::print_type;
    using gotcha_types   = typename get_true_types<trait::is_gotcha, tuple_type>::type;
    using user_bundle_types =
        typename get_true_types<trait::is_user_bundle, tuple_type>::type;

    template <template <typename> class Op, typename... T>
    using generic_operation = mpl::impl::generic_operation<Op, tag_type, T...>;

    template <template <typename> class Op, typename... T>
    using custom_operation = mpl::impl::custom_operation<Op, T...>;

    template <template <typename> class Op, typename TupleT = tuple_type>
    using operation_t = typename generic_operation<Op, TupleT>::type;

    template <template <typename> class Op, typename TupleT = tuple_type>
    using custom_operation_t = typename mpl::impl::custom_operation<Op, TupleT>::type;

    template <typename U>
    using sample_type_t = mpl::impl::sample_type_t<U>;

public:
    // public static function section
    static constexpr bool   empty() { return (size() == 0); }
    static constexpr size_t size() { return sizeof...(Types); }

    template <typename... T>
    static auto get_scope_config()
    {
        return scope::config{ quirk_config<quirk::flat_scope, T...>::value,
                              quirk_config<quirk::timeline_scope, T...>::value,
                              quirk_config<quirk::tree_scope, T...>::value };
    }

    template <typename... T>
    static auto get_store_config()
    {
        return !quirk_config<quirk::no_store, T...>::value;
    }

public:
    // public static data section
    static constexpr bool has_gotcha_v = (mpl::get_tuple_size<gotcha_types>::value != 0);
    static constexpr bool has_user_bundle_v =
        (mpl::get_tuple_size<user_bundle_types>::value != 0);

public:
    // public member function section
    auto hash() const { return m_hash; }
    auto get_hash() const { return m_hash; }

    int64_t     laps() const { return m_laps; }
    std::string key() const { return std::string{ get_hash_identifier_fast(m_hash) }; }

    void store(bool v) { m_store(v); }
    bool store() const { return m_store(); }
    bool get_store() const { return m_store(); }

    auto&       get_scope() { return m_scope; }
    const auto& get_scope() const { return m_scope; }

    void rekey(const string_t& _key)
    {
        m_hash = add_hash_id(_key);
        compute_width(_key);
    }

protected:
    using ctor_params_t = std::tuple<hash_value_type, bool, scope::config>;

    // protected construction / destruction section
    template <typename U = impl_type>
    base_bundle(ctor_params_t _params = ctor_params_t(0, settings::enabled(),
                                                      scope::get_default()),
                enable_if_t<std::tuple_size<U>::value != 0, int> = 0)
    : m_scope(std::get<2>(_params) + get_scope_config())
    , m_hash(std::get<0>(_params))
    {
        m_store(std::get<1>(_params) && get_store_config());
    }

    template <typename U = impl_type>
    base_bundle(ctor_params_t                                    = ctor_params_t(),
                enable_if_t<std::tuple_size<U>::value == 0, int> = 0)
    {}

    ~base_bundle()                      = default;
    base_bundle(const base_bundle&)     = default;
    base_bundle(base_bundle&&) noexcept = default;
    base_bundle& operator=(const base_bundle&) = default;
    base_bundle& operator=(base_bundle&&) noexcept = default;

protected:
    // protected member function section
    enum ConfigIdx
    {
        StoreIdx  = 0,
        PushedIdx = 1,
        ActiveIdx = 2
    };

    const auto& prefix() const { return get_persistent_data().prefix; }
    const auto& get_prefix() const { return prefix(); }

    TIMEMORY_ALWAYS_INLINE bool m_store() const { return m_config.test(StoreIdx); }
    TIMEMORY_ALWAYS_INLINE bool m_is_pushed() const { return m_config.test(PushedIdx); }
    TIMEMORY_ALWAYS_INLINE bool m_is_active() const { return m_config.test(ActiveIdx); }

    TIMEMORY_ALWAYS_INLINE void m_store(bool v) { m_config.set(StoreIdx, v); }
    TIMEMORY_ALWAYS_INLINE void m_is_pushed(bool v) { m_config.set(PushedIdx, v); }
    TIMEMORY_ALWAYS_INLINE void m_is_active(bool v) { m_config.set(ActiveIdx, v); }

    void compute_width(const string_t& _key) const
    {
        output_width(_key.length() + get_prefix().length() + 1);
    }

    void update_width() const { compute_width(key()); }

protected:
    // protected static function section
    static int64_t output_width(int64_t width = 0)
    {
        return get_persistent_data().get_width(width);
    }

protected:
    // protected static function section [HANDLERS]
    template <typename... T>
    static hash_value_type handle_key(type_list<T...>, hash_value_type _hash)
    {
        return _hash;
    }

    // HAS SIZE
    template <typename... T>
    static hash_value_type handle_key(type_list<T...>, const string_t& _key,
                                      enable_if_t<sizeof...(T) != 0, int> = {})
    {
        return (settings::enabled()) ? add_hash_id(_key) : 0;
    }

    template <typename... T>
    static hash_value_type handle_key(type_list<T...>, const captured_location_t& _loc,
                                      enable_if_t<sizeof...(T) != 0, int> = {})
    {
        return _loc.get_hash();
    }

    template <typename... T, typename... U>
    static scope::config handle_scope(type_list<T...>, scope::config _scope,
                                      quirk::config<U...>,
                                      enable_if_t<sizeof...(T) != 0, int> = {})
    {
        return _scope + get_scope_config<T..., U...>();
    }

    template <typename... T, typename... U>
    static scope::config handle_scope(type_list<T...>, quirk::config<U...>,
                                      enable_if_t<sizeof...(T) != 0, int> = {})
    {
        return scope::get_default() + get_scope_config<T..., U...>();
    }

    template <typename... T, typename... U>
    static bool handle_store(type_list<T...>, bool _enable, quirk::config<U...>,
                             enable_if_t<sizeof...(T) != 0, int> = {})
    {
        return get_store_config<T..., U...>() && _enable && mpl::impl::global_enabled();
    }

    template <typename... T, typename... U>
    static bool handle_store(type_list<T...>, std::true_type, quirk::config<U...>,
                             enable_if_t<sizeof...(T) != 0, int> = {})
    {
        return get_store_config<T..., U...>() && mpl::impl::global_enabled();
    }

    template <typename... T, typename... U>
    static bool handle_store(type_list<T...>, std::false_type, quirk::config<U...>,
                             enable_if_t<sizeof...(T) != 0, int> = {})
    {
        return get_store_config<T..., U...>();
    }

    // ZERO SIZE
    template <typename... T>
    static hash_value_type handle_key(type_list<T...>, const string_t&,
                                      enable_if_t<sizeof...(T) == 0, int> = {})
    {
        return 0;
    }

    template <typename... T>
    static hash_value_type handle_key(type_list<T...>, const captured_location_t&,
                                      enable_if_t<sizeof...(T) == 0, int> = {})
    {
        return 0;
    }

    template <typename... T, typename... U>
    static auto handle_scope(type_list<T...>, scope::config, quirk::config<U...>,
                             enable_if_t<sizeof...(T) == 0, int> = {})
    {
        return scope::config{};
    }

    template <typename... T, typename... U>
    static auto handle_scope(type_list<T...>, quirk::config<U...>,
                             enable_if_t<sizeof...(T) == 0, int> = {})
    {
        return scope::config{};
    }

    template <typename... T, typename... U>
    static bool handle_store(type_list<T...>, bool, quirk::config<U...>,
                             enable_if_t<sizeof...(T) == 0, int> = {})
    {
        return false;
    }

    template <typename... T, typename... U>
    static bool handle_store(type_list<T...>, std::true_type, quirk::config<U...>,
                             enable_if_t<sizeof...(T) == 0, int> = {})
    {
        return false;
    }

    template <typename... T, typename... U>
    static bool handle_store(type_list<T...>, std::false_type, quirk::config<U...>,
                             enable_if_t<sizeof...(T) == 0, int> = {})
    {
        return false;
    }

    // GENERIC HANDLERS
    template <typename... T, typename KeyT, typename StoreT, typename ScopeT,
              typename QuirkT>
    static auto handle(type_list<T...> _types, KeyT&& _key, StoreT&& _store,
                       ScopeT&& _scope, QuirkT&& _quirk)
    {
        return ctor_params_t{ handle_key(_types, std::forward<KeyT>(_key)),
                              handle_store(_types, std::forward<StoreT>(_store),
                                           std::forward<QuirkT>(_quirk)),
                              handle_scope(_types, std::forward<ScopeT>(_scope),
                                           std::forward<QuirkT>(_quirk)) };
    }

    template <typename... T, typename KeyT, typename StoreT>
    static auto handle(type_list<T...> _types, KeyT&& _key, StoreT&& _store,
                       scope::config _scope)
    {
        return handle(_types, std::forward<KeyT>(_key), std::forward<StoreT>(_store),
                      _scope, quirk::config<>{});
    }

    template <typename... T, typename KeyT, typename StoreT, typename... U>
    static auto handle(type_list<T...> _types, KeyT&& _key, StoreT&& _store,
                       quirk::config<U...> _quirk)
    {
        return ctor_params_t{ handle_key(_types, std::forward<KeyT>(_key)),
                              handle_store(_types, std::forward<StoreT>(_store), _quirk),
                              handle_scope(_types, _quirk) };
    }

    template <typename... T, typename Tp, typename TupleT, typename FuncT, typename... U>
    static void init(type_list<T...>, Tp& _this, TupleT&& _data, FuncT&& _init,
                     quirk::config<U...>                 = quirk::config<>{},
                     enable_if_t<sizeof...(T) != 0, int> = {})
    {
        if(mpl::impl::global_enabled())
        {
            IF_CONSTEXPR(!quirk_config<quirk::no_init, T..., U...>::value)
            {
                _init(_this);
            }
            _this.set_prefix(_this.get_hash());
            invoke::set_scope(_data, _this.get_scope());
            IF_CONSTEXPR(quirk_config<quirk::auto_start, T..., U...>::value)
            {
                _this.start();
            }
        }
    }

    template <typename... T, typename Tp, typename TupleT, typename FuncT, typename... U>
    static void init(type_list<T...>, Tp&, TupleT&&, FuncT&&,
                     quirk::config<U...>                 = quirk::config<>{},
                     enable_if_t<sizeof...(T) == 0, int> = {})
    {}

protected:
    // protected member data section
    std::bitset<3>  m_config = {};
    scope::config   m_scope  = scope::get_default();
    int64_t         m_laps   = 0;
    hash_value_type m_hash   = 0;

private:
    // private alias section
    template <typename T, typename... U>
    using quirk_config = mpl::impl::quirk_config<T, type_list<Types...>, U...>;

private:
    // private static data section
    struct persistent_data
    {
        int64_t get_width(int64_t _w)
        {
            IF_CONSTEXPR(!empty())
            {
                auto&&  memorder_v = std::memory_order_relaxed;
                int64_t propose_width, current_width;
                auto    compute = [&]() { return std::max(width.load(memorder_v), _w); };
                while((propose_width = compute()) >
                      (current_width = width.load(memorder_v)))
                {
                    width.compare_exchange_strong(current_width, propose_width,
                                                  memorder_v);
                }
                return width.load(memorder_v);
            }
            (void) _w;  // unused parameter warning when 'if constexpr' available
            return 0;
        }

        std::atomic<int64_t> width{ 0 };
        std::string          prefix = []() -> std::string {
            IF_CONSTEXPR(!empty())
            {
                if(!dmp::is_initialized())
                    return std::string(">>> ");

                // prefix spacing
                static uint16_t _width = 1;
                if(dmp::size() > 9)
                    _width = std::max(_width, (uint16_t)(log10(dmp::size()) + 1));
                std::stringstream ss;
                ss.fill('0');
                ss << "|" << std::setw(_width) << dmp::rank() << ">>> ";
                return ss.str();
            }
            else { return std::string{}; }
        }();
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance{};
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
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using data_type = data_tuple_t<Types...>;

    template <typename... Args>
    base_bundle(Args&&... args)
    : base_bundle<Tag, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename Tag, typename... Types>
class base_bundle<Tag, type_list<Types...>> : public base_bundle<Tag, Types...>
{
public:
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using data_type = data_tuple_t<Types...>;

    template <typename... Args>
    base_bundle(Args&&... args)
    : base_bundle<Tag, Types...>(std::forward<Args>(args)...)
    {}
};
//
//======================================================================================//
//
template <typename... Types>
struct stack_bundle
: public base_bundle<TIMEMORY_API, Types...>
, public concepts::stack_wrapper
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using data_type = data_tuple_t<Types...>;

    template <typename... Args>
    stack_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct stack_bundle<std::tuple<Types...>>
: public base_bundle<TIMEMORY_API, Types...>
, public concepts::stack_wrapper
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using data_type = data_tuple_t<Types...>;

    template <typename... Args>
    stack_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct stack_bundle<type_list<Types...>>
: public base_bundle<TIMEMORY_API, Types...>
, public concepts::stack_wrapper
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using data_type = data_tuple_t<Types...>;

    template <typename... Args>
    stack_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};
//
//======================================================================================//
//
template <typename... Types>
struct heap_bundle
: public base_bundle<TIMEMORY_API, Types...>
, public concepts::heap_wrapper
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using data_type = data_tuple_t<Types*...>;

    template <typename... Args>
    heap_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct heap_bundle<std::tuple<Types...>>
: public base_bundle<TIMEMORY_API, Types...>
, public concepts::heap_wrapper
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using data_type = data_tuple_t<Types*...>;

    template <typename... Args>
    heap_bundle(Args&&... args)
    : base_bundle<TIMEMORY_API, Types...>(std::forward<Args>(args)...)
    {}
};

template <typename... Types>
struct heap_bundle<type_list<Types...>>
: public base_bundle<TIMEMORY_API, Types...>
, public concepts::heap_wrapper
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using data_type = data_tuple_t<Types*...>;

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
: public conditional_t<trait::is_available<ApiT>::value,
                       base_bundle<ApiT, remove_pointer_t<Types>...>,
                       base_bundle<ApiT, std::tuple<>>>
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using base_bundle_type = conditional_t<trait::is_available<ApiT>::value,
                                           base_bundle<ApiT, remove_pointer_t<Types>...>,
                                           base_bundle<ApiT, std::tuple<>>>;
    using data_type        = conditional_t<trait::is_available<ApiT>::value,
                                    data_tuple_t<Types...>, std::tuple<>>;
    using tuple_type       = data_type;
    using impl_type        = data_type;

    template <typename... Args>
    api_bundle(Args&&... args)
    : base_bundle_type(std::forward<Args>(args)...)
    {}
};

template <typename ApiT, typename... Types>
struct api_bundle<ApiT, std::tuple<Types...>>
: public conditional_t<trait::is_available<ApiT>::value,
                       base_bundle<ApiT, remove_pointer_t<Types>...>,
                       base_bundle<ApiT, std::tuple<>>>
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using base_bundle_type = conditional_t<trait::is_available<ApiT>::value,
                                           base_bundle<ApiT, remove_pointer_t<Types>...>,
                                           base_bundle<ApiT, std::tuple<>>>;
    using data_type        = conditional_t<trait::is_available<ApiT>::value,
                                    data_tuple_t<Types...>, std::tuple<>>;
    using tuple_type       = data_type;
    using impl_type        = data_type;

    template <typename... Args>
    api_bundle(Args&&... args)
    : base_bundle_type(std::forward<Args>(args)...)
    {}
};

template <typename ApiT, typename... Types>
struct api_bundle<ApiT, type_list<Types...>>
: public conditional_t<trait::is_available<ApiT>::value,
                       base_bundle<ApiT, remove_pointer_t<Types>...>,
                       base_bundle<ApiT, std::tuple<>>>
{
    template <typename... T>
    using data_tuple_t = non_placeholder_t<non_quirk_t<std::tuple<T...>>>;

    using base_bundle_type = conditional_t<trait::is_available<ApiT>::value,
                                           base_bundle<ApiT, remove_pointer_t<Types>...>,
                                           base_bundle<ApiT, std::tuple<>>>;
    using data_type        = conditional_t<trait::is_available<ApiT>::value,
                                    data_tuple_t<Types...>, std::tuple<>>;
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
