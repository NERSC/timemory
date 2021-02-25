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

#include "timemory/macros/attributes.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"

#include <array>

namespace tim
{
namespace component
{
//
template <typename Tp, size_t Sz>
struct base_data;
//
template <typename Tp>
struct base_data<Tp, 0>
{
    using value_type = null_type;

    TIMEMORY_INLINE value_type get_value() const { return value_type{}; }
    TIMEMORY_INLINE value_type get_accum() const { return value_type{}; }
    TIMEMORY_INLINE value_type get_last() const { return value_type{}; }
    TIMEMORY_INLINE void       set_value(value_type) {}
    TIMEMORY_INLINE void       set_accum(value_type) {}
    TIMEMORY_INLINE void       set_last(value_type) {}

    base_data()  = default;
    ~base_data() = default;

    base_data& operator=(base_data&&) = default;  // NOLINT
    base_data(base_data&&)            = default;  // NOLINT

    base_data(const base_data&) = default;
    base_data& operator=(const base_data&) = default;

    TIMEMORY_INLINE value_type load(bool) { return value_type{}; }
    TIMEMORY_INLINE value_type load(bool) const { return value_type{}; }

    void plus(const value_type&) {}
    void minus(const value_type&) {}
    void multiply(const value_type&) {}
    void divide(const value_type&) {}

    template <typename Up>
    void plus(Up&&)
    {}

    template <typename Up>
    void minus(Up&&)
    {}

    template <typename Up>
    void multiply(Up&&)
    {}

    template <typename Up>
    void divide(Up&&)
    {}

protected:
    value_type value = {};  // NOLINT
    value_type accum = {};  // NOLINT
    value_type last  = {};  // NOLINT
};
//
template <typename Tp>
struct base_data<Tp, 1>
{
    using empty_type = std::tuple<>;
    using value_type = Tp;
    using accum_type = empty_type;
    using last_type  = empty_type;

    TIMEMORY_INLINE const value_type& get_value() const { return value; }
    TIMEMORY_INLINE const value_type& get_accum() const { return value; }
    TIMEMORY_INLINE const value_type& get_last() const { return value; }
    TIMEMORY_INLINE void              set_value(value_type v) { value = v; }
    TIMEMORY_INLINE void              set_accum(value_type) {}
    TIMEMORY_INLINE void              set_last(value_type) {}

    base_data()  = default;
    ~base_data() = default;

    base_data& operator=(base_data&&) = default;  // NOLINT
    base_data(base_data&&)            = default;  // NOLINT

    base_data(const base_data&) = default;
    base_data& operator=(const base_data&) = default;

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int = 0)
    {
        ar(cereal::make_nvp("value", value));
    }

    TIMEMORY_INLINE value_type& load(bool) { return value; }
    TIMEMORY_INLINE const value_type& load(bool) const { return value; }

    void plus(const value_type& lhs)
    {
        value = math::compute<value_type>::plus(value, lhs);
    }

    void minus(const value_type& lhs)
    {
        value = math::compute<value_type>::minus(value, lhs);
    }

    void multiply(const value_type& lhs)
    {
        value = math::compute<value_type>::multiply(value, lhs);
    }

    void divide(const value_type& lhs)
    {
        value = math::compute<value_type>::divide(value, lhs);
    }

    template <typename Up>
    auto plus(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(), void())
    {
        value = math::compute<value_type>::plus(value, std::forward<Up>(rhs).get_value());
    }

    template <typename Up>
    auto minus(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(), void())
    {
        value =
            math::compute<value_type>::minus(value, std::forward<Up>(rhs).get_value());
    }

    template <typename Up>
    auto multiply(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(), void())
    {
        value =
            math::compute<value_type>::multiply(value, std::forward<Up>(rhs).get_value());
    }

    template <typename Up>
    auto divide(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(), void())
    {
        value =
            math::compute<value_type>::divide(value, std::forward<Up>(rhs).get_value());
    }

    // we need 'using base_data<Tp>::accum' to be valid so make a function call
    value_type& accum() { return value; }

    // we need 'using base_data<Tp>::last' to be valid so make a function call
    value_type& last() { return value; }

protected:
    void reset() { value = Tp{}; }

protected:
    value_type value = Tp{};  // NOLINT
};
//
template <typename Tp>
struct base_data<Tp, 2>
{
    template <typename Up, typename ValueT>
    friend struct base;

    using empty_type = std::tuple<>;
    using value_type = Tp;
    using accum_type = Tp;
    using last_type  = empty_type;

    TIMEMORY_INLINE const value_type& get_value() const { return value; }
    TIMEMORY_INLINE const value_type& get_accum() const { return accum; }
    TIMEMORY_INLINE const value_type& get_last() const { return value; }
    TIMEMORY_INLINE void              set_value(value_type v) { value = v; }
    TIMEMORY_INLINE void              set_accum(value_type v) { accum = v; }
    TIMEMORY_INLINE void              set_last(value_type) {}

    base_data()  = default;
    ~base_data() = default;

    base_data& operator=(base_data&&) = default;  // NOLINT
    base_data(base_data&&)            = default;  // NOLINT

    base_data(const base_data& rhs) = default;
    base_data& operator=(const base_data& rhs) = default;

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int = 0)
    {
        ar(cereal::make_nvp("value", value), cereal::make_nvp("accum", accum));
    }

    TIMEMORY_INLINE value_type& load(bool is_transient)
    {
        return (is_transient) ? accum : value;
    }
    TIMEMORY_INLINE const value_type& load(bool is_transient) const
    {
        return (is_transient) ? accum : value;
    }

    void plus(const value_type& rhs)
    {
        value = math::compute<value_type>::plus(value, rhs);
        accum = math::compute<value_type>::plus(accum, rhs);
    }

    void minus(const value_type& rhs)
    {
        value = math::compute<value_type>::minus(value, rhs);
        accum = math::compute<value_type>::minus(accum, rhs);
    }

    void multiply(const value_type& rhs)
    {
        value = math::compute<value_type>::multiply(value, rhs);
        accum = math::compute<value_type>::multiply(accum, rhs);
    }

    void divide(const value_type& rhs)
    {
        value = math::compute<value_type>::divide(value, rhs);
        accum = math::compute<value_type>::divide(accum, rhs);
    }

    template <typename Up>
    auto plus(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(),
                                    (void) std::forward<Up>(rhs).get_accum(), void())
    {
        value = math::compute<value_type>::plus(value, std::forward<Up>(rhs).get_value());
        accum = math::compute<value_type>::plus(accum, std::forward<Up>(rhs).get_accum());
    }

    template <typename Up>
    auto minus(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(),
                                     (void) std::forward<Up>(rhs).get_accum(), void())
    {
        value =
            math::compute<value_type>::minus(value, std::forward<Up>(rhs).get_value());
        accum =
            math::compute<value_type>::minus(accum, std::forward<Up>(rhs).get_accum());
    }

    template <typename Up>
    auto multiply(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(),
                                        (void) std::forward<Up>(rhs).get_accum(), void())
    {
        value =
            math::compute<value_type>::multiply(value, std::forward<Up>(rhs).get_value());
        accum =
            math::compute<value_type>::multiply(accum, std::forward<Up>(rhs).get_accum());
    }

    template <typename Up>
    auto divide(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(),
                                      (void) std::forward<Up>(rhs).get_accum(), void())
    {
        value =
            math::compute<value_type>::divide(value, std::forward<Up>(rhs).get_value());
        accum =
            math::compute<value_type>::divide(accum, std::forward<Up>(rhs).get_accum());
    }

    // we need 'using base_data<Tp>::last' to be valid so make a function call
    value_type& last() { return value; }

protected:
    void reset()
    {
        value = Tp{};
        accum = Tp{};
    }

protected:
    value_type value = Tp{};  // NOLINT
    value_type accum = Tp{};  // NOLINT
};
//
template <typename Tp>
struct base_data<Tp, 3>
{
    template <typename Up, typename ValueT>
    friend struct base;

    using value_type = Tp;
    using accum_type = Tp;
    using last_type  = Tp;

    TIMEMORY_INLINE const value_type& get_value() const { return value; }
    TIMEMORY_INLINE const value_type& get_accum() const { return accum; }
    TIMEMORY_INLINE const value_type& get_last() const { return last; }
    TIMEMORY_INLINE void              set_value(value_type v) { value = v; }
    TIMEMORY_INLINE void              set_accum(value_type v) { accum = v; }
    TIMEMORY_INLINE void              set_last(value_type v) { last = v; }

    base_data()  = default;
    ~base_data() = default;

    base_data& operator=(base_data&&) = default;  // NOLINT
    base_data(base_data&&)            = default;  // NOLINT

    base_data(const base_data& rhs) = default;
    base_data& operator=(const base_data& rhs) = default;

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int = 0)
    {
        ar(cereal::make_nvp("value", value), cereal::make_nvp("accum", accum),
           cereal::make_nvp("last", last));
    }

    TIMEMORY_INLINE value_type& load(bool is_transient)
    {
        return (is_transient) ? accum : value;
    }
    TIMEMORY_INLINE const value_type& load(bool is_transient) const
    {
        return (is_transient) ? accum : value;
    }

    void plus(const value_type& rhs)
    {
        value = math::compute<value_type>::plus(value, rhs);
        accum = math::compute<value_type>::plus(accum, rhs);
    }

    void minus(const value_type& rhs)
    {
        value = math::compute<value_type>::minus(value, rhs);
        accum = math::compute<value_type>::minus(accum, rhs);
    }

    void multiply(const value_type& rhs)
    {
        value = math::compute<value_type>::multiply(value, rhs);
        accum = math::compute<value_type>::multiply(accum, rhs);
    }

    void divide(const value_type& rhs)
    {
        value = math::compute<value_type>::divide(value, rhs);
        accum = math::compute<value_type>::divide(accum, rhs);
    }

    template <typename Up>
    auto plus(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(),
                                    (void) std::forward<Up>(rhs).get_accum(), void())
    {
        value = math::compute<value_type>::plus(value, std::forward<Up>(rhs).get_value());
        accum = math::compute<value_type>::plus(accum, std::forward<Up>(rhs).get_accum());
    }

    template <typename Up>
    auto minus(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(),
                                     (void) std::forward<Up>(rhs).get_accum(), void())
    {
        value =
            math::compute<value_type>::minus(value, std::forward<Up>(rhs).get_value());
        accum =
            math::compute<value_type>::minus(accum, std::forward<Up>(rhs).get_accum());
    }

    template <typename Up>
    auto multiply(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(),
                                        (void) std::forward<Up>(rhs).get_accum(), void())
    {
        value =
            math::compute<value_type>::multiply(value, std::forward<Up>(rhs).get_value());
        accum =
            math::compute<value_type>::multiply(accum, std::forward<Up>(rhs).get_accum());
    }

    template <typename Up>
    auto divide(Up&& rhs) -> decltype((void) std::forward<Up>(rhs).get_value(),
                                      (void) std::forward<Up>(rhs).get_accum(), void())
    {
        value =
            math::compute<value_type>::divide(value, std::forward<Up>(rhs).get_value());
        accum =
            math::compute<value_type>::divide(accum, std::forward<Up>(rhs).get_accum());
    }

protected:
    void reset()
    {
        value = Tp{};
        accum = Tp{};
        last  = Tp{};
    }

protected:
    value_type value = Tp{};  // NOLINT
    value_type accum = Tp{};  // NOLINT
    value_type last  = Tp{};  // NOLINT
};
//
struct base_state
{
    TIMEMORY_INLINE bool get_is_running() const { return _compute_state(RunningIdx); }
    TIMEMORY_INLINE bool get_is_on_stack() const { return _compute_state(OnStackIdx); }
    TIMEMORY_INLINE bool get_is_transient() const { return _compute_state(TransientIdx); }
    TIMEMORY_INLINE bool get_is_flat() const { return _compute_state(FlatIdx); }
    TIMEMORY_INLINE bool get_depth_change() const { return _compute_state(DepthIdx); }

    TIMEMORY_INLINE void set_is_running(bool v) { _assign_state(RunningIdx, v); }
    TIMEMORY_INLINE void set_is_on_stack(bool v) { _assign_state(OnStackIdx, v); }
    TIMEMORY_INLINE void set_is_transient(bool v) { _assign_state(TransientIdx, v); }
    TIMEMORY_INLINE void set_is_flat(bool v) { _assign_state(FlatIdx, v); }
    TIMEMORY_INLINE void set_depth_change(bool v) { _assign_state(DepthIdx, v); }
    TIMEMORY_INLINE void reset() { m_state_value = 0; }

protected:
    TIMEMORY_INLINE uint8_t get_state_value() const { return m_state_value; }
    TIMEMORY_INLINE void    set_state_value(uint8_t v) { m_state_value = v; }

private:
    uint8_t m_state_value = 0;

    enum State
    {
        RunningIdx   = 1 << 0,  // 1
        OnStackIdx   = 1 << 1,  // 2
        TransientIdx = 1 << 2,  // 4
        FlatIdx      = 1 << 3,  // 8
        DepthIdx     = 1 << 4,  // 16
    };

    TIMEMORY_INLINE bool _compute_state(State idx) const { return (m_state_value & idx); }
    TIMEMORY_INLINE void _assign_state(State idx, bool v)
    {
        bool _curr = _compute_state(idx);
        if(_curr != v)
        {
            if(!_curr)
                m_state_value |= idx;
            else
                m_state_value &= (~idx);
        }
    }
};
//
namespace internal
{
template <typename Tp, typename ValueT>
struct base_data;
//
template <typename Tp>
struct base_data<Tp, void>
{
    using type = ::tim::component::base_data<void, 0>;
};
//
template <typename Tp>
struct base_data<Tp, type_list<>>
{
    using type = ::tim::component::base_data<type_list<>, 0>;
};
//
template <typename Tp>
struct base_data<Tp, null_type>
{
    using type = ::tim::component::base_data<null_type, 0>;
};
//
template <typename Tp, typename ValueT>
struct base_data
{
    static constexpr auto   has_accum  = trait::base_has_accum<Tp>::value;
    static constexpr auto   has_last   = trait::base_has_last<Tp>::value;
    static constexpr size_t array_size = 1 + ((has_accum) ? 1 : 0) + ((has_last) ? 1 : 0);

    using type = ::tim::component::base_data<ValueT, array_size>;

    static_assert(!(has_last && !has_accum), "Error! base cannot have last w/o accum");
};
//
}  // namespace internal
//
template <typename Tp, typename ValueT>
using base_data_t = typename internal::base_data<Tp, ValueT>::type;
//
}  // namespace component
}  // namespace tim
