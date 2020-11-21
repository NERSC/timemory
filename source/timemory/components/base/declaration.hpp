//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include "timemory/components/base/types.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"

//======================================================================================//

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace component
{
//
//======================================================================================//
//
//                      default base class for components
//
//======================================================================================//
//
/// \struct tim::component::empty_base
///
/// The default base class for timemory components.
///
struct empty_base
{};
//
//======================================================================================//
//
//                          static polymorphism base
//          base component class for all components with non-void types
//
//======================================================================================//
//
/// \struct tim::component::base<Tp, Value>
/// \tparam Tp the component type
/// \tparam Value the value type of the component (overrides tim::data<T>)
///
/// A helper static polymorphic base class for components. It is not required to be
/// used but generally recommended for ease of implementation.
///
template <typename Tp, typename Value>
struct base
: public trait::dynamic_base<Tp>::type
, public concepts::component
{
    using EmptyT = std::tuple<>;
    template <typename U>
    using vector_t = std::vector<U>;

public:
    static constexpr bool is_component = true;
    using Type                         = Tp;
    using value_type                   = Value;
    using accum_type =
        conditional_t<trait::base_has_accum<Tp>::value, value_type, EmptyT>;
    using last_type = conditional_t<trait::base_has_last<Tp>::value, value_type, EmptyT>;
    using dynamic_type = typename trait::dynamic_base<Tp>::type;
    using cache_type   = typename trait::cache<Tp>::type;

    using this_type          = Tp;
    using base_type          = base<Tp, Value>;
    using storage_type       = storage<Tp, Value>;
    using base_storage_type  = tim::base::storage;
    using graph_iterator     = typename storage_type::iterator;
    using state_t            = state<this_type>;
    using statistics_policy  = policy::record_statistics<Tp, Value>;
    using fmtflags           = std::ios_base::fmtflags;
    using value_compute_type = math::compute<value_type, value_type>;
    using accum_compute_type = math::compute<accum_type, accum_type>;

private:
    friend class impl::storage<Tp, trait::uses_value_storage<Tp, Value>::value>;
    friend class storage<Tp, Value>;
    friend struct node::graph<Tp>;

    friend struct operation::init_storage<Tp>;
    friend struct operation::cache<Tp>;
    friend struct operation::construct<Tp>;
    friend struct operation::set_prefix<Tp>;
    friend struct operation::push_node<Tp>;
    friend struct operation::pop_node<Tp>;
    friend struct operation::record<Tp>;
    friend struct operation::reset<Tp>;
    friend struct operation::measure<Tp>;
    friend struct operation::start<Tp>;
    friend struct operation::stop<Tp>;
    friend struct operation::minus<Tp>;
    friend struct operation::plus<Tp>;
    friend struct operation::multiply<Tp>;
    friend struct operation::divide<Tp>;
    friend struct operation::base_printer<Tp>;
    friend struct operation::print<Tp>;
    friend struct operation::print_storage<Tp>;
    friend struct operation::copy<Tp>;
    friend struct operation::sample<Tp>;
    friend struct operation::serialization<Tp>;
    friend struct operation::finalize::get<Tp, true>;
    friend struct operation::finalize::get<Tp, false>;
    friend struct operation::finalize::merge<Tp, true>;
    friend struct operation::finalize::merge<Tp, false>;
    friend struct operation::finalize::print<Tp, true>;
    friend struct operation::finalize::print<Tp, false>;

    template <typename Ret, typename Lhs, typename Rhs>
    friend struct operation::compose;

    static_assert(std::is_pointer<Tp>::value == false, "Error pointer base type");

public:
    TIMEMORY_DEFAULT_OBJECT(base)

public:
    template <typename... Args>
    static void configure(Args&&...)
    {}

public:
    /// store that start has been called
    TIMEMORY_INLINE void set_started();

    /// store that stop has been called
    TIMEMORY_INLINE void set_stopped();

    /// reset the values
    TIMEMORY_INLINE void reset();

    /// record a measurement
    TIMEMORY_INLINE void measure();

    /// phase
    TIMEMORY_INLINE bool get_is_transient() const { return (laps > 0 || is_transient); }

    /// currently collecting
    TIMEMORY_INLINE bool get_is_running() const { return is_running; }

    /// currently on call-stack
    TIMEMORY_INLINE bool get_is_on_stack() const { return is_on_stack; }

    /// flat call-stack entry
    TIMEMORY_INLINE bool get_is_flat() const { return is_flat; }

    /// last push/pop changed call-stack depth
    TIMEMORY_INLINE bool get_depth_change() const { return depth_change; }

    /// get number of measurement
    TIMEMORY_INLINE int64_t get_laps() const { return laps; }

    /// assign to pointer
    TIMEMORY_INLINE void get_opaque_data(void*& ptr, size_t _hash) const;

    /// assign type to a pointer
    TIMEMORY_INLINE void get(void*& ptr, size_t _typeid_hash) const;

    /// retrieve the current measurement value in the units for the type
    TIMEMORY_INLINE auto get() const { return this->load(); }

    /// retrieve the current measurement value in the units for the type in a format
    /// that can be piped to the output stream operator ('<<')
    TIMEMORY_INLINE auto get_display() const { return this->load(); }

    TIMEMORY_INLINE bool operator<(const base_type& rhs) const
    {
        return (load() < rhs.load());
    }
    TIMEMORY_INLINE bool operator>(const base_type& rhs) const
    {
        return (load() > rhs.load());
    }
    TIMEMORY_INLINE bool operator<=(const base_type& rhs) const { return !(*this > rhs); }
    TIMEMORY_INLINE bool operator>=(const base_type& rhs) const { return !(*this < rhs); }

    TIMEMORY_INLINE Type& operator+=(const base_type& rhs) { return plus_oper(rhs); }
    TIMEMORY_INLINE Type& operator-=(const base_type& rhs) { return minus_oper(rhs); }
    TIMEMORY_INLINE Type& operator*=(const base_type& rhs) { return multiply_oper(rhs); }
    TIMEMORY_INLINE Type& operator/=(const base_type& rhs) { return divide_oper(rhs); }

    TIMEMORY_INLINE Type& operator+=(const Type& rhs) { return plus_oper(rhs); }
    TIMEMORY_INLINE Type& operator-=(const Type& rhs) { return minus_oper(rhs); }
    TIMEMORY_INLINE Type& operator*=(const Type& rhs) { return multiply_oper(rhs); }
    TIMEMORY_INLINE Type& operator/=(const Type& rhs) { return divide_oper(rhs); }

    TIMEMORY_INLINE Type& operator+=(const Value& rhs) { return plus_oper(rhs); }
    TIMEMORY_INLINE Type& operator-=(const Value& rhs) { return minus_oper(rhs); }
    TIMEMORY_INLINE Type& operator*=(const Value& rhs) { return multiply_oper(rhs); }
    TIMEMORY_INLINE Type& operator/=(const Value& rhs) { return divide_oper(rhs); }

    template <typename Up = Tp, typename Vp = Value,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    void print(std::ostream&) const;

    template <typename Up = Tp, typename Vp = Value,
              enable_if_t<!trait::uses_value_storage<Up, Vp>::value, int> = 0>
    void print(std::ostream&) const;

    friend std::ostream& operator<<(std::ostream& os, const base_type& obj)
    {
        obj.print(os);
        return os;
    }

    /// serialization load (input)
    template <typename Archive, typename Up = Type,
              enable_if_t<!trait::custom_serialization<Up>::value, int> = 0>
    void load(Archive& ar, const unsigned int);

    /// serialization store (output)
    template <typename Archive, typename Up = Type,
              enable_if_t<!trait::custom_serialization<Up>::value, int> = 0>
    void save(Archive& ar, const unsigned int version) const;

    /// retrieve the raw value of the most recent measurement
    TIMEMORY_INLINE const value_type& get_value() const { return value; }

    /// retrieve the raw value of the all the measurements
    TIMEMORY_INLINE const accum_type& get_accum() const { return accum; }

    /// if \ref tim::trait::base_has_last is true, this will return the last valid
    /// measurement
    TIMEMORY_INLINE const last_type& get_last() const { return last; }

    TIMEMORY_INLINE void set_laps(int64_t v) { laps = v; }
    TIMEMORY_INLINE void set_value(value_type v) { value = v; }
    TIMEMORY_INLINE void set_accum(accum_type v) { accum = v; }
    TIMEMORY_INLINE void set_last(last_type v) { last = v; }
    TIMEMORY_INLINE void set_is_transient(bool v) { is_transient = v; }

    template <typename Vp, typename Up = Tp,
              enable_if_t<trait::sampler<Up>::value, int> = 0>
    static void add_sample(Vp&&);  /// add a sample

    TIMEMORY_INLINE void set_iterator(graph_iterator itr) { graph_itr = itr; }
    TIMEMORY_INLINE auto get_iterator() const { return graph_itr; }

protected:
    static base_storage_type* get_storage();

    template <typename Up = Tp, enable_if_t<trait::base_has_accum<Up>::value, int> = 0>
    TIMEMORY_INLINE value_type& load();
    template <typename Up = Tp, enable_if_t<trait::base_has_accum<Up>::value, int> = 0>
    TIMEMORY_INLINE const value_type& load() const;

    template <typename Up = Tp, enable_if_t<!trait::base_has_accum<Up>::value, int> = 0>
    TIMEMORY_INLINE value_type& load();
    template <typename Up = Tp, enable_if_t<!trait::base_has_accum<Up>::value, int> = 0>
    TIMEMORY_INLINE const value_type& load() const;

    TIMEMORY_INLINE Type& plus_oper(const base_type& rhs);
    TIMEMORY_INLINE Type& minus_oper(const base_type& rhs);
    TIMEMORY_INLINE Type& multiply_oper(const base_type& rhs);
    TIMEMORY_INLINE Type& divide_oper(const base_type& rhs);

    TIMEMORY_INLINE Type& plus_oper(const Type& rhs);
    TIMEMORY_INLINE Type& minus_oper(const Type& rhs);
    TIMEMORY_INLINE Type& multiply_oper(const Type& rhs);
    TIMEMORY_INLINE Type& divide_oper(const Type& rhs);

    TIMEMORY_INLINE Type& plus_oper(const Value& rhs);
    TIMEMORY_INLINE Type& minus_oper(const Value& rhs);
    TIMEMORY_INLINE Type& multiply_oper(const Value& rhs);
    TIMEMORY_INLINE Type& divide_oper(const Value& rhs);

    TIMEMORY_INLINE void plus(const base_type& rhs)
    {
        laps += rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

    TIMEMORY_INLINE void minus(const base_type& rhs)
    {
        laps -= rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

public:
    TIMEMORY_INLINE auto plus(crtp::base, const base_type& rhs) { this->plus(rhs); }
    TIMEMORY_INLINE auto minus(crtp::base, const base_type& rhs) { this->minus(rhs); }

protected:
    bool           is_running   = false;
    bool           is_on_stack  = false;
    bool           is_transient = false;
    bool           is_flat      = false;
    bool           depth_change = false;
    int64_t        laps         = 0;
    value_type     value        = value_type{};
    accum_type     accum        = accum_type{};
    last_type      last         = last_type{};
    graph_iterator graph_itr    = graph_iterator{ nullptr };

public:
    static constexpr bool timing_category_v = trait::is_timing_category<Type>::value;
    static constexpr bool memory_category_v = trait::is_memory_category<Type>::value;
    static constexpr bool timing_units_v    = trait::uses_timing_units<Type>::value;
    static constexpr bool memory_units_v    = trait::uses_memory_units<Type>::value;
    static constexpr bool percent_units_v   = trait::uses_percent_units<Type>::value;
    static constexpr auto ios_fixed         = std::ios_base::fixed;
    static constexpr auto ios_decimal       = std::ios_base::dec;
    static constexpr auto ios_showpoint     = std::ios_base::showpoint;
    static const short    precision         = percent_units_v ? 1 : 3;
    static const short    width             = percent_units_v ? 6 : 8;
    static const fmtflags format_flags      = ios_fixed | ios_decimal | ios_showpoint;

    template <typename Up = Type, typename UnitT = typename trait::units<Up>::type,
              enable_if_t<std::is_same<UnitT, int64_t>::value, int> = 0>
    static int64_t unit();

    template <typename Up    = Type,
              typename UnitT = typename trait::units<Up>::display_type,
              enable_if_t<std::is_same<UnitT, std::string>::value, int> = 0>
    static std::string display_unit();

    template <typename Up = Type, typename UnitT = typename trait::units<Up>::type,
              enable_if_t<std::is_same<UnitT, int64_t>::value, int> = 0>
    static int64_t get_unit();

    template <typename Up    = Type,
              typename UnitT = typename trait::units<Up>::display_type,
              enable_if_t<std::is_same<UnitT, std::string>::value, int> = 0>
    static std::string get_display_unit();

    static short       get_width();
    static short       get_precision();
    static fmtflags    get_format_flags();
    static std::string label();
    static std::string description();
    static std::string get_label();
    static std::string get_description();
};
//
//======================================================================================//
//
//                       static polymorphic base
//          base component class for all components with void types
//
//======================================================================================//
//
template <typename Tp>
struct base<Tp, void>
: public trait::dynamic_base<Tp>::type
, public concepts::component
{
    using EmptyT = std::tuple<>;

public:
    static constexpr bool is_component = true;
    using Type                         = Tp;
    using value_type                   = void;
    using accum_type                   = void;
    using last_type                    = void;
    using dynamic_type                 = typename trait::dynamic_base<Tp>::type;
    using cache_type                   = typename trait::cache<Tp>::type;

    using this_type    = Tp;
    using base_type    = base<Tp, value_type>;
    using storage_type = storage<Tp, void>;

private:
    friend class impl::storage<Tp, false>;
    friend class storage<Tp, void>;
    friend struct node::graph<Tp>;

    friend struct operation::init_storage<Tp>;
    friend struct operation::cache<Tp>;
    friend struct operation::construct<Tp>;
    friend struct operation::set_prefix<Tp>;
    friend struct operation::push_node<Tp>;
    friend struct operation::pop_node<Tp>;
    friend struct operation::record<Tp>;
    friend struct operation::reset<Tp>;
    friend struct operation::measure<Tp>;
    friend struct operation::start<Tp>;
    friend struct operation::stop<Tp>;
    friend struct operation::minus<Tp>;
    friend struct operation::plus<Tp>;
    friend struct operation::multiply<Tp>;
    friend struct operation::divide<Tp>;
    friend struct operation::print<Tp>;
    friend struct operation::print_storage<Tp>;
    friend struct operation::copy<Tp>;
    friend struct operation::serialization<Tp>;

    template <typename Ret, typename Lhs, typename Rhs>
    friend struct operation::compose;

public:
    TIMEMORY_DEFAULT_OBJECT(base)

public:
    template <typename... Args>
    static void configure(Args&&...)
    {}

public:
    TIMEMORY_INLINE void set_started();
    TIMEMORY_INLINE void set_stopped();
    TIMEMORY_INLINE void reset();
    TIMEMORY_INLINE void measure();
    TIMEMORY_INLINE bool get_is_running() const { return is_running; }
    TIMEMORY_INLINE bool get_is_on_stack() const { return is_on_stack; }
    TIMEMORY_INLINE bool get_is_transient() const { return is_transient; }
    TIMEMORY_INLINE int64_t get_laps() const { return 0; }
    TIMEMORY_INLINE void    get_opaque_data(void*& ptr, size_t _typeid_hash) const;

    friend std::ostream& operator<<(std::ostream& os, const base_type&) { return os; }

    TIMEMORY_INLINE void get() {}
    TIMEMORY_INLINE void get(void*& ptr, size_t _typeid_hash) const;

    // used by operation::finalize::print<Type>
    TIMEMORY_INLINE void operator-=(const base_type&) {}
    TIMEMORY_INLINE void operator-=(const Type&) {}

protected:
    TIMEMORY_INLINE void plus(const base_type& rhs)
    {
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

    TIMEMORY_INLINE void minus(const base_type& rhs)
    {
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

public:
    TIMEMORY_INLINE auto plus(crtp::base, const base_type& rhs) { this->plus(rhs); }
    TIMEMORY_INLINE auto minus(crtp::base, const base_type& rhs) { this->minus(rhs); }

protected:
    bool is_running   = false;
    bool is_on_stack  = false;
    bool is_transient = false;

public:
    //
    // components with void data types do not use label()/get_label()
    // to generate an output filename so provide a default one from
    // (potentially demangled) typeid(Type).name() and strip out
    // namespace and any template parameters + replace any spaces
    // with underscores
    //
    static std::string label();
    static std::string description();
    static std::string get_label();
    static std::string get_description();
};
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
struct base<Tp, std::tuple<>>
: public trait::dynamic_base<Tp>::type
, public concepts::component
{
public:
    static constexpr bool is_component = true;
    using Type                         = Tp;
    using value_type                   = std::tuple<>;
    using accum_type                   = value_type;
    using last_type                    = value_type;
    using dynamic_type                 = typename trait::dynamic_base<Tp>::type;
    using cache_type                   = typename trait::cache<Tp>::type;

    using this_type      = Tp;
    using base_type      = base<Tp, value_type>;
    using storage_type   = storage<Tp, void>;
    using graph_iterator = void*;

private:
    friend class impl::storage<Tp, false>;
    friend class storage<Tp, void>;
    friend struct node::graph<Tp>;

    friend struct operation::init_storage<Tp>;
    friend struct operation::cache<Tp>;
    friend struct operation::construct<Tp>;
    friend struct operation::set_prefix<Tp>;
    friend struct operation::push_node<Tp>;
    friend struct operation::pop_node<Tp>;
    friend struct operation::record<Tp>;
    friend struct operation::reset<Tp>;
    friend struct operation::measure<Tp>;
    friend struct operation::start<Tp>;
    friend struct operation::stop<Tp>;
    friend struct operation::minus<Tp>;
    friend struct operation::plus<Tp>;
    friend struct operation::multiply<Tp>;
    friend struct operation::divide<Tp>;
    friend struct operation::print<Tp>;
    friend struct operation::print_storage<Tp>;
    friend struct operation::copy<Tp>;
    friend struct operation::serialization<Tp>;

    template <typename Ret, typename Lhs, typename Rhs>
    friend struct operation::compose;

public:
    TIMEMORY_DEFAULT_OBJECT(base)

public:
    template <typename... Args>
    static void configure(Args&&...)
    {}

public:
    TIMEMORY_INLINE void reset();    // reset the values
    TIMEMORY_INLINE void measure();  // just record a measurment
    TIMEMORY_INLINE void start();
    TIMEMORY_INLINE void stop();

    TIMEMORY_INLINE auto start(crtp::base) {}
    TIMEMORY_INLINE auto stop(crtp::base) {}

    template <typename CacheT                                     = cache_type,
              enable_if_t<!concepts::is_null_type<CacheT>::value> = 0>
    TIMEMORY_INLINE void start(const CacheT&)
    {}
    template <typename CacheT                                     = cache_type,
              enable_if_t<!concepts::is_null_type<CacheT>::value> = 0>
    TIMEMORY_INLINE void stop(const CacheT&)
    {}

    TIMEMORY_INLINE void set_started() {}
    TIMEMORY_INLINE void set_stopped() {}

    friend std::ostream& operator<<(std::ostream& os, const base_type&) { return os; }

    TIMEMORY_INLINE void get() {}
    TIMEMORY_INLINE void get(void*&, size_t) const {}

    TIMEMORY_INLINE void get_opaque_data(void*&, size_t) const {}

    TIMEMORY_INLINE int64_t get_laps() const { return 0; }
    TIMEMORY_INLINE auto    get_is_running() const { return false; }
    TIMEMORY_INLINE auto    get_is_on_stack() const { return false; }
    TIMEMORY_INLINE auto    get_is_transient() const { return false; }

    // used by operation::finalize::print<Type>
    TIMEMORY_INLINE void operator-=(const base_type&) {}
    TIMEMORY_INLINE void operator-=(const Type&) {}

    TIMEMORY_INLINE auto load() { return Tp{}; }

protected:
    TIMEMORY_INLINE void plus(const base_type&) {}
    TIMEMORY_INLINE void minus(const base_type&) {}

public:
    TIMEMORY_INLINE auto plus(crtp::base, const base_type&) {}
    TIMEMORY_INLINE auto minus(crtp::base, const base_type&) {}

protected:
    bool           is_running   = false;
    bool           is_on_stack  = false;
    bool           is_transient = false;
    bool           is_flat      = false;
    bool           depth_change = false;
    int64_t        laps         = 0;
    value_type     value        = value_type{};
    accum_type     accum        = accum_type{};
    last_type      last         = last_type{};
    graph_iterator graph_itr    = nullptr;

public:
    static std::string label() { return ""; }
    static std::string description() { return ""; }
    static std::string get_label() { return ""; }
    static std::string get_description() { return ""; }
};
//
//----------------------------------------------------------------------------------//
//
}  // namespace component
//
//----------------------------------------------------------------------------------//
//
}  // namespace tim
