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

/** \file timemory/utility/types.hpp
 * \headerfile timemory/utility/types.hpp "timemory/utility/types.hpp"
 * Declaration of types for utility directory
 *
 */

#pragma once

#include "timemory/macros/os.hpp"
#include "timemory/mpl/concepts.hpp"

#include <array>
#include <bitset>
#include <functional>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

//======================================================================================//

#if !defined(TIMEMORY_FOLD_EXPRESSION)
#    define TIMEMORY_FOLD_EXPRESSION(...)                                                \
        ::tim::consume_parameters(::std::initializer_list<int>{ (__VA_ARGS__, 0)... })
#endif

//======================================================================================//

#if !defined(TIMEMORY_RETURN_FOLD_EXPRESSION)
#    define TIMEMORY_RETURN_FOLD_EXPRESSION(...)                                         \
        ::std::make_tuple((::tim::consume_parameters(), __VA_ARGS__)...)
#endif

//======================================================================================//

#if !defined(TIMEMORY_DECLARE_EXTERN_TEMPLATE)
#    define TIMEMORY_DECLARE_EXTERN_TEMPLATE(...) extern template __VA_ARGS__;
#endif

//======================================================================================//

#if !defined(TIMEMORY_INSTANTIATE_EXTERN_TEMPLATE)
#    define TIMEMORY_INSTANTIATE_EXTERN_TEMPLATE(...) template __VA_ARGS__;
#endif

//======================================================================================//

#if !defined(TIMEMORY_ESC)
#    define TIMEMORY_ESC(...) __VA_ARGS__
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_ATTRIBUTE)
#    if defined(_WINDOWS)
#        define TIMEMORY_ATTRIBUTE(...) __declspec(__VA_ARGS__)
#    else
#        define TIMEMORY_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
#    endif
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_ALWAYS_INLINE) && !defined(_WINDOWS)
#    define TIMEMORY_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#    define TIMEMORY_ALWAYS_INLINE inline
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_FLATTEN) && !defined(_WINDOWS)
#    define TIMEMORY_FLATTEN [[gnu::flatten]]
#else
#    define TIMEMORY_FLATTEN
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_HOT) && !defined(_WINDOWS)
#    define TIMEMORY_HOT [[gnu::hot]]
#else
#    define TIMEMORY_HOT
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_COLD) && !defined(_WINDOWS)
#    define TIMEMORY_COLD [[gnu::cold]]
#else
#    define TIMEMORY_COLD
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_CONST)
#    define TIMEMORY_CONST [[gnu::const]]
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DEPRECATED)
#    define TIMEMORY_DEPRECATED(...) [[gnu::deprecated(__VA_ARGS__)]]
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_EXTERN_VISIBLE)
#    define TIMEMORY_EXTERN_VISIBLE [[gnu::externally_visible]]
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_ALIAS)
#    define TIMEMORY_ALIAS(...) [[gnu::alias(__VA_ARGS__)]]
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_NOINLINE)
#    define TIMEMORY_NOINLINE TIMEMORY_ATTRIBUTE(noinline)
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DELETED_OBJECT)
#    define TIMEMORY_DELETED_OBJECT(NAME)                                                \
        NAME()            = delete;                                                      \
        NAME(const NAME&) = delete;                                                      \
        NAME(NAME&&)      = delete;                                                      \
        NAME& operator=(const NAME&) = delete;                                           \
        NAME& operator=(NAME&&) = delete;
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DEFAULT_OBJECT)
#    define TIMEMORY_DEFAULT_OBJECT(NAME)                                                \
        NAME()                = default;                                                 \
        NAME(const NAME&)     = default;                                                 \
        NAME(NAME&&) noexcept = default;                                                 \
        NAME& operator=(const NAME&) = default;                                          \
        NAME& operator=(NAME&&) noexcept = default;
#endif

//======================================================================================//
//
namespace tim
{
//
/// Alias template make_integer_sequence
template <typename Tp, Tp Num>
using make_integer_sequence = std::make_integer_sequence<Tp, Num>;
//
/// Alias template index_sequence
template <size_t... Idx>
using index_sequence = std::integer_sequence<size_t, Idx...>;
//
/// Alias template make_index_sequence
template <size_t Num>
using make_index_sequence = std::make_integer_sequence<size_t, Num>;
//
/// Alias template index_sequence_for
template <typename... Types>
using index_sequence_for = std::make_index_sequence<sizeof...(Types)>;
//
/// Alias template for enable_if
template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;
//
/// Alias template for decay
template <typename T>
using decay_t = typename std::decay<T>::type;
//
template <bool Val, typename Lhs, typename Rhs>
using conditional_t = typename std::conditional<(Val), Lhs, Rhs>::type;
//
using true_type = std::true_type;
//
using false_type = std::false_type;
//
template <int N>
using priority_constant = std::integral_constant<int, N>;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct identity
{
    using type = T;
};
//
template <typename T>
using identity_t = typename identity<T>::type;
//
//--------------------------------------------------------------------------------------//
//
/// \struct null_type
/// \brief this is a placeholder type for optional type-traits. It is used as the default
/// type for the type-traits to signify there is no specialization.
struct null_type : concepts::null_type
{};
//
//--------------------------------------------------------------------------------------//
//
/// \struct type_list
/// \brief lightweight tuple-alternative for meta-programming logic
template <typename... Tp>
struct type_list
{};
//
//--------------------------------------------------------------------------------------//
//
/// \struct type_list
/// \brief lightweight tuple-alternative for meta-programming logic
template <size_t Idx, typename Tp>
struct type_list_element;

namespace internal
{
template <size_t Idx, size_t TargIdx, typename... Tail>
struct type_list_element;

template <size_t Idx, size_t TargIdx>
struct type_list_element<Idx, TargIdx>
{
    using type = tim::null_type;
    // TargIdx will be equal to Idx + 1 in second half of conditional statement
    // below. If the second half of that conditional statement is entered, the
    // following static_assert will be true
    static_assert(TargIdx < Idx + 2, "Error! Index exceeded size of of type_list");
};

template <size_t Idx, size_t TargIdx, typename Tp, typename... Tail>
struct type_list_element<Idx, TargIdx, Tp, Tail...>
{
    using type =
        conditional_t<(Idx == TargIdx), Tp,
                      typename type_list_element<Idx + 1, TargIdx, Tail...>::type>;
};
}  // namespace internal

template <size_t Idx, typename... Types>
struct type_list_element<Idx, type_list<Types...>>
{
    using type = typename internal::type_list_element<0, Idx, Types...>::type;
};

template <size_t Idx, typename Tp>
using type_list_element_t = typename type_list_element<Idx, Tp>::type;
//
//--------------------------------------------------------------------------------------//
//
/// \fn consume_parameters
/// \brief use this function to get rid of "unused parameter" warnings
template <typename... ArgsT>
void
consume_parameters(ArgsT&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename DeleterT>
class singleton;
//
//--------------------------------------------------------------------------------------//
//
namespace cupti
{
struct result;
}
//
//--------------------------------------------------------------------------------------//
//
namespace crtp
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::crtp::base
/// \brief a generic type for prioritizing a function call to the base class over
/// derived functions, e.g. void start(crtp::base, Args&&... args) { start(args...); }
struct base
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace crtp
//
//--------------------------------------------------------------------------------------//
//
namespace mpl
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::mpl::lightweight
/// \brief a generic type for indicating that function call or constructor should be
/// as lightweight as possible.
struct lightweight
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp>
struct piecewise_select
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace mpl
//
//--------------------------------------------------------------------------------------//
//
namespace scope
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct tree
/// \brief Dummy struct to designates tree (hierarchical) storage. This scope (default)
/// maintains nesting in the call-graph storage. In this scoping mode, the results
/// will be separated from each other based on the identifier AND the current number
/// of component instances in a "start" region. E.g. for two components with the
/// same identifiers where the first calls start, then the second calls start then
/// the second will be at a depth of +1 relative to the first (i.e. a child of the first).
struct tree : std::integral_constant<int, 2>
{};
//
//--------------------------------------------------------------------------------------//
//
/// \struct flat
/// \brief Dummy struct to designates flat (no hierarchy) storage.
/// When flat scoping is globally enabled, all entries to the call-graph storage at
/// entered at a depth of zero. Thus, if you want a report of all the function calls
/// and their total values for each identifier, flat scoping should be globally enabled.
/// This can be combined with timeline scoping to produce results where every measurement
/// is its own call-graph entry at a depth of zero (produces a large amount of data).
/// Flat-scoping can be enabled at the component bundler level also, there are two
/// ways to do this: (1) to enable flat-scoping for all instances of the bundle, add \ref
/// tim::quirk::flat_scope to the template parameters of the bundler; (2) to enable
/// flat-scoping for specific bundler instances, pass \code{.cpp}
/// tim::quirk::config<tim::quirk::flat_scope, ...>{} \endcode as the second argument to
/// the constructor of the bundle.
struct flat : std::integral_constant<int, 0>
{};
//
//--------------------------------------------------------------------------------------//
//
/// \struct timeline
/// \brief Dummy struct to designates timeline (hierarchical, non-duplicated) storage.
/// It is meaningless by itself and should be combined with \ref tim::scope::tree or
/// \ref tim::scope::flat. A tree timeline has all the hierarchy properties of the
/// tree scope but entries at the same depth with the same identifiers are separated
/// entries in the resuls.
/// Timeline-scoping can be enabled at the component bundler level also, there are two
/// ways to do this: (1) to enable timeline-scoping for all instances of the bundle, add
/// \ref tim::quirk::timeline_scope to the template parameters of the bundler; (2) to
/// enable timeline-scoping for specific bundler instances, pass \code{.cpp}
/// tim::quirk::config<tim::quirk::timeline_scope, ...>{} \endcode as the second argument
/// to the constructor of the bundle.
struct timeline : std::integral_constant<int, 1>
{};
//
//--------------------------------------------------------------------------------------//
//
static constexpr size_t scope_count = 3;
using data_type                     = std::bitset<scope_count>;
using input_type                    = std::array<bool, scope_count>;
//
//--------------------------------------------------------------------------------------//
//
inline input_type&
get_fields()
{
    static input_type _instance{ { false, false, false } };
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Arg, size_t... Idx>
static inline auto
generate(Arg&& arg, std::index_sequence<Idx...>)
{
    static_assert(sizeof...(Idx) <= scope_count, "Error! Bad index sequence size");
    data_type ret;
    TIMEMORY_FOLD_EXPRESSION(ret.set(Idx, arg[Idx]));
    return ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx>
static inline auto
either(data_type ret, data_type arg, std::index_sequence<Idx...>)
{
    static_assert(sizeof...(Idx) <= scope_count, "Error! Bad index sequence size");
    TIMEMORY_FOLD_EXPRESSION(ret.set(Idx, ret.test(Idx) || arg.test(Idx)));
    return ret;
}
//
//--------------------------------------------------------------------------------------//
//
static inline data_type
get_default()
{
    return generate(get_fields(), std::make_index_sequence<scope_count>{});
}
//
//--------------------------------------------------------------------------------------//
/// \struct tim::scope::config
/// \brief this data type encodes the options of storage scope. The default is
/// hierarchical (tree) scope. Specification of flat scope overrides the hierarchy
/// scope, e.g. you cannot have a hierarchical flat scope. The timeline scope
/// is meaningless should a specification of tree or flat, thus the valid combinations
/// are: tree, flat, tree + timeline, flat + timeline.
struct config : public data_type
{
    config()
    : data_type(get_default())
    {}

    config(const data_type& obj)
    : data_type(obj)
    {}

    config(data_type&& obj) noexcept
    : data_type(std::forward<data_type>(obj))
    {}

    explicit config(bool _flat)
    : data_type(generate(input_type{ { _flat, false, false } },
                         std::make_index_sequence<scope_count>{}))
    {}

    config(bool _flat, bool _timeline)
    : data_type(generate(input_type{ { _flat, _timeline, false } },
                         std::make_index_sequence<scope_count>{}))
    {}

    config(bool _flat, bool _timeline, bool _tree)
    : data_type(generate(input_type{ { _flat, _timeline, _tree } },
                         std::make_index_sequence<scope_count>{}))
    {}

    config(tree)
    : config(false, false, false)
    {}

    config(flat)
    : config(true, false, false)
    {}

    config(timeline)
    : config(false, true, false)
    {}

    template <typename Arg, typename... Args,
              std::enable_if_t<(std::is_same<Arg, tree>::value ||
                                std::is_same<Arg, flat>::value ||
                                std::is_same<Arg, timeline>::value),
                               int> = 0>
    explicit config(Arg&& arg, Args&&... args)
    {
        *this += std::forward<Arg>(arg);
        TIMEMORY_FOLD_EXPRESSION(*this += std::forward<Args>(args));
    }

    ~config()                 = default;
    config(const config&)     = default;
    config(config&&) noexcept = default;
    config& operator=(const config&) = default;
    config& operator=(config&&) noexcept = default;

    template <typename T, std::enable_if_t<(std::is_same<T, tree>::value ||
                                            std::is_same<T, flat>::value ||
                                            std::is_same<T, timeline>::value),
                                           int> = 0>
    config& operator+=(T)
    {
        this->data_type::set(T::value, true);
        return *this;
    }

    using data_type::set;

    template <typename T, std::enable_if_t<(std::is_same<T, tree>::value ||
                                            std::is_same<T, flat>::value ||
                                            std::is_same<T, timeline>::value),
                                           int> = 0>
    config& set(bool val = true)
    {
        this->data_type::set(T::value, val);
        return *this;
    }

    bool is_flat() const { return this->test(flat::value); }
    bool is_timeline() const { return this->test(timeline::value); }
    // "tree" is default behavior so it returns true if nothing is set but gives
    // priority to the flat setting
    bool is_tree() const
    {
        return this->none() || (this->test(tree::value) && !this->test(flat::value));
    }
    bool is_flat_timeline() const { return (is_flat() && is_timeline()); }
    bool is_tree_timeline() const { return (is_tree() && is_timeline()); }

    template <bool ForceFlatT>
    bool is_flat() const
    {
        return (ForceFlatT) ? true : this->test(flat::value);
    }

    template <bool ForceTreeT, bool ForceTimeT>
    bool is_tree() const
    {
        return (ForceTreeT)
                   ? true
                   : ((ForceTimeT) ? false
                                   : (this->none() || (this->test(tree::value) &&
                                                       !this->test(flat::value))));
    }

    friend std::ostream& operator<<(std::ostream& os, const config& obj)
    {
        std::stringstream ss;
        ss << std::boolalpha << "tree: " << obj.is_tree() << ", flat: " << obj.is_flat()
           << ", timeline: " << obj.is_timeline()
           << ". Values: flat::value = " << obj.test(flat::value)
           << ", timeline::value = " << obj.test(timeline::value)
           << ", tree::value = " << obj.test(tree::value);
        os << ss.str();
        return os;
    }

    template <typename ForceTreeT = false_type, typename ForceFlatT = false_type,
              typename ForceTimeT = false_type>
    uint64_t compute_depth(uint64_t _current)
    {
        static_assert(!(ForceTreeT::value && ForceFlatT::value),
                      "Error! Type cannot enforce tree-based call-stack depth storage "
                      "and flat call-stack depth storage simulatenously");
        // flat:     always at depth of 1
        // tree:     features nesting
        // timeline: features nesting if not flat and depth of 1 if flat
        if(ForceFlatT::value || is_flat())
        {
            // flat + timeline will be differentiated via compute_hash
            // flat + tree is invalid and flat takes precedence
            // printf("compute_depth is flat at %i\n", (int) _current);
            return 1;
        }
        // if not flat, return the nested depth and compute_hash will account
        // for whether the entry is a duplicate or not
        // printf("compute_depth is tree or timeline at %i\n", (int) _current);
        return _current + 1;
    }

    template <typename ForceTreeT = false_type, typename ForceFlatT = false_type,
              typename ForceTimeT = false_type>
    uint64_t compute_hash(uint64_t _id, uint64_t _depth, uint64_t& _counter)
    {
        // flat/tree:  always compute the same hash for a given depth and key
        // timeline:   uses a counter to differentiate idential hashes occurring at diff
        //             time points.
        // below is a fall-through (i.e. not an if-else) bc either tree or flat can be
        // combined with timeline but in the case of tree + flat, flat will take
        // precedence.
        if(is_tree<ForceTreeT::value, ForceTimeT::value>() ||
           is_flat<ForceFlatT::value>())
        {
            // printf("compute_hash is tree or flat at %i\n", (int) _depth);
            _id ^= _depth;
        }
        if(ForceTimeT::value || is_timeline())
        {
            // printf("compute_hash is timeline at %i\n", (int) _depth);
#if defined(TIMEMORY_USE_TIMELINE_RNG)
            _id ^= get_random_value<uint64_t>();
#else
            _id ^= _counter++;
#endif
        }
        // printf("compute_hash is %i at depth %i (counter = %i)\n", (int) _id, (int)
        // _depth,
        //       (int) _counter);
        return _id;
    }

private:
#if defined(TIMEMORY_USE_TIMELINE_RNG)
    // random number generator
    template <typename T = std::mt19937_64>
    static inline T& get_rng(size_t initial_seed = 0)
    {
        static T _instance = [=]() {
            T _rng;
            _rng.seed((initial_seed == 0) ? std::random_device()() : initial_seed);
            return _rng;
        }();
        return _instance;
    }

    // random integer
    template <typename T, typename R = std::mt19937_64,
              std::enable_if_t<std::is_integral<T>::value, int> = 0>
    static inline T get_random_value(T beg = 0, T end = std::numeric_limits<T>::max())
    {
        std::uniform_int_distribution<T> dist(beg, end);
        return dist(get_rng<R>());
    }
#endif
};
//
//--------------------------------------------------------------------------------------//
//
inline config
operator+(config _lhs, tree)
{
    _lhs.set(tree::value, true);
    return _lhs;
}
//
//--------------------------------------------------------------------------------------//
//
inline config
operator+(config _lhs, flat)
{
    _lhs.set(flat::value, true);
    return _lhs;
}
//
//--------------------------------------------------------------------------------------//
//
inline config
operator+(config _lhs, timeline)
{
    _lhs.set(timeline::value, true);
    return _lhs;
}
//
//--------------------------------------------------------------------------------------//
//
inline config
operator+(config _lhs, config _rhs)
{
    return either(_lhs, _rhs, std::make_index_sequence<scope_count>{});
}
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::scope::destructor
/// \brief provides an object which can be returned from functions that will execute
/// the lambda provided during construction when it is destroyed
///
struct destructor
{
    template <typename FuncT>
    destructor(FuncT&& _func)
    : m_functor(std::forward<FuncT>(_func))
    {}

    // delete copy operations
    destructor(const destructor&) = delete;
    destructor& operator=(const destructor&) = delete;

    // allow move operations
    destructor(destructor&& rhs) noexcept
    : m_functor(std::move(rhs.m_functor))
    {
        rhs.m_functor = []() {};
    }

    destructor& operator=(destructor&& rhs) noexcept
    {
        if(this != &rhs)
        {
            m_functor     = std::move(rhs.m_functor);
            rhs.m_functor = []() {};
        }
        return *this;
    }

    ~destructor() { m_functor(); }

private:
    std::function<void()> m_functor = []() {};
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace scope
//
//--------------------------------------------------------------------------------------//
//
namespace lifetime
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct lifetime::scoped
/// \brief Dummy struct for meta-programming to designate that a component activates
/// it's features at the first start() invocation and deactivates it's features when
/// all instances that called start() have called stop(). Thus, the component's
/// features are dependent on at least one component instance existing in memory
/// (excluding the instances in the call-graph, which never call start/stop)
///
struct scoped
{};
//
//--------------------------------------------------------------------------------------//
//
/// \struct lifetime::persistent
/// \brief Dummy struct for meta-programming to designate that a component activates its
/// features in {global,thread}_init and deactivates it's features in
/// {global,thead}_finalize
///
struct persistent
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace lifetime
//
//--------------------------------------------------------------------------------------//
//
namespace audit
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct incoming
/// \brief Used by component audit member function to designate the
/// parameters being passed are incoming (e.g. before a gotcha wrappee is invoked)
///
struct incoming
{};
//
//--------------------------------------------------------------------------------------//
//
/// \struct outgoing
/// \brief Used by component audit member function to designate the
/// parameters being passed are outgoing (e.g. the return value from a gotcha wrappee)
///
// audit the return type
struct outgoing
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace audit

//--------------------------------------------------------------------------------------//

}  // namespace tim

namespace std
{
template <size_t Idx, typename... Types>
struct tuple_element<Idx, tim::type_list<Types...>>
{
    using type = typename tim::type_list_element<Idx, tim::type_list<Types...>>::type;
};
}  // namespace std
