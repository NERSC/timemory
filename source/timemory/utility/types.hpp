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

#include <array>
#include <bitset>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

//======================================================================================//
//
#if !defined(TIMEMORY_FOLD_EXPRESSION)
#    define TIMEMORY_FOLD_EXPRESSION(...)                                                \
        ::tim::consume_parameters(::std::initializer_list<int>{ (__VA_ARGS__, 0)... })
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
        NAME()            = default;                                                     \
        NAME(const NAME&) = default;                                                     \
        NAME(NAME&&)      = default;                                                     \
        NAME& operator=(const NAME&) = default;                                          \
        NAME& operator=(NAME&&) = default;
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
template <bool B, typename T = void>
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
/// \class type_list
/// \brief lightweight tuple-alternative for meta-programming logic
template <typename... Tp>
struct type_list
{};
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
/// \class tim::crtp::base
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
/// \class tim::mpl::lightweight
/// \brief a generic type for indicating that function call or constructor should be
/// as lightweight as possible.
struct lightweight
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
/// \class flat
/// \brief Dummy struct to designates flat (no hierarchy) storage
struct flat : std::integral_constant<int, 0>
{};
//
//--------------------------------------------------------------------------------------//
//
/// \class timeline
/// \brief Dummy struct to designates timeline (hierarchical, non-duplicated) storage
struct timeline : std::integral_constant<int, 1>
{};
//
//--------------------------------------------------------------------------------------//
//
/// \class tree
/// \brief Dummy struct to designates tree (hierarchical) storage
struct tree : std::integral_constant<int, 2>
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
//
struct config : public data_type
{
    config()
    : data_type(get_default())
    {}

    config(const data_type& obj)
    : data_type(obj)
    {}

    config(data_type&& obj)
    : data_type(std::forward<data_type>(obj))
    {}

    config(bool _flat)
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

    template <typename Arg, typename... Args,
              std::enable_if_t<(std::is_same<Arg, tree>::value ||
                                std::is_same<Arg, flat>::value ||
                                std::is_same<Arg, timeline>::value),
                               int> = 0>
    config(Arg&& arg, Args&&... args)
    {
        *this += std::forward<Arg>(arg);
        TIMEMORY_FOLD_EXPRESSION(*this += std::forward<Args>(args));
    }

    ~config()             = default;
    config(const config&) = default;
    config(config&&)      = default;
    config& operator=(const config&) = default;
    config& operator=(config&&) = default;

    template <typename T, std::enable_if_t<(std::is_same<T, tree>::value ||
                                            std::is_same<T, flat>::value ||
                                            std::is_same<T, timeline>::value),
                                           int> = 0>
    config& operator+=(T)
    {
        this->data_type::set(T::value, true);
        return *this;
    }

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

    uint64_t compute_depth(uint64_t _current)
    {
        // flat:     always at depth of 1
        // tree:     features nesting
        // timeline: retains current depth
        if(is_flat())
        {
            // flat + timeline will be differentiated via compute_hash
            // flat + tree is invalid and flat takes precedence
            return 1;
        }
        else if(is_timeline())
        {
            // tree + timeline is essentially a flat profile at the given depth
            // bc returning a nested depth for tree + timeline would look like recursion
            return _current + 1;
        }
        // if neither flat nor timeline are enabled, return the nested depth
        return _current + 1;
    }

    uint64_t compute_hash(uint64_t _id, uint64_t _depth, uint64_t& _counter)
    {
        // flat/tree:  always compute the same hash for a given depth and key
        // timeline:   uses a counter to differentiate idential hashes occurring at diff
        //             time points.
        // below is a fall-through (i.e. not if-else). Thus, either tree or flat can be
        // combined with timeline but in the case of tree + flat, flat will take
        // precedence.
        if(is_tree() || is_flat())
            _id ^= _depth;
        if(is_timeline())
        {
#if defined(TIMEMORY_USE_TIMELINE_RNG)
            _id ^= get_random_value<uint64_t>();
#else
            _id ^= _counter++;
#endif
        }
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
              std::enable_if_t<(std::is_integral<T>::value), int> = 0>
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
inline const config
operator+(config _lhs, tree _rhs)
{
    return (_lhs += _rhs);
}
//
//--------------------------------------------------------------------------------------//
//
inline const config
operator+(config _lhs, flat _rhs)
{
    return (_lhs += _rhs);
}
//
//--------------------------------------------------------------------------------------//
//
inline const config
operator+(config _lhs, timeline _rhs)
{
    return (_lhs += _rhs);
}
//
//--------------------------------------------------------------------------------------//
//
inline const config
operator+(config _lhs, config _rhs)
{
    return either(_lhs, _rhs, std::make_index_sequence<scope_count>{});
}
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
/// \class lifetime::scoped
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
/// \class lifetime::persistent
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
/// \class incoming
/// \brief Used by component audit member function to designate the
/// parameters being passed are incoming (e.g. before a gotcha wrappee is invoked)
///
struct incoming
{};
//
//--------------------------------------------------------------------------------------//
//
/// \class outgoing
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
