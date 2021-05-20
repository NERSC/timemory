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

#include "timemory/components/base/data.hpp"
#include "timemory/components/base/types.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/storage/graph.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"

namespace tim
{
namespace component
{
//
template <typename Tp>
using graph_iterator_t = typename graph<node::graph<Tp>>::iterator;
//
template <typename Tp>
using graph_const_iterator_t = typename graph<node::graph<Tp>>::const_iterator;
//
/// \struct tim::component::empty_base
/// \brief The default base class for timemory components.
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
/// \brief A helper static polymorphic base class for components. It is not required to be
/// used but generally recommended for ease of implementation.
///
template <typename Tp, typename Value>
struct base
: public trait::dynamic_base<Tp>::type
, private base_state
, private base_data_t<Tp, Value>
, public concepts::component
{
    using EmptyT = std::tuple<>;
    template <typename U>
    using vector_t = std::vector<U>;

public:
    static constexpr bool is_component = true;
    using Type                         = Tp;
    using value_type                   = Value;
    using data_type                    = base_data_t<Tp, Value>;
    using accum_type                   = typename data_type::accum_type;
    using last_type                    = typename data_type::last_type;
    using dynamic_type                 = typename trait::dynamic_base<Tp>::type;
    using cache_type                   = typename trait::cache<Tp>::type;

    using this_type         = Tp;
    using base_type         = base<Tp, Value>;
    using base_storage_type = tim::base::storage;
    using storage_type      = storage<Tp, Value>;
    using graph_iterator    = graph_iterator_t<Tp>;
    using state_t           = state<this_type>;
    using statistics_policy = policy::record_statistics<Tp, Value>;
    using fmtflags          = std::ios_base::fmtflags;

private:
    friend struct node::graph<Tp>;
    friend struct operation::init_storage<Tp>;
    friend struct operation::fini_storage<Tp>;
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
    friend struct operation::set_started<Tp>;
    friend struct operation::set_stopped<Tp>;
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
    /// get the opaque binding for user-bundle
    static opaque get_opaque(scope::config);

    /// store that start has been called
    void set_started();

    /// store that stop has been called
    void set_stopped();

    /// reset the values
    void reset();

    /// assign type to a pointer
    void get(void*& ptr, size_t _typeid_hash) const;

    /// retrieve the current measurement value in the units for the type
    TIMEMORY_NODISCARD auto get() const { return this->load(); }

    /// retrieve the current measurement value in the units for the type in a format
    /// that can be piped to the output stream operator ('<<')
    TIMEMORY_NODISCARD auto get_display() const { return this->load(); }

    Type& operator+=(const Type& rhs) { return plus_oper(rhs); }
    Type& operator-=(const Type& rhs) { return minus_oper(rhs); }
    Type& operator*=(const Type& rhs) { return multiply_oper(rhs); }
    Type& operator/=(const Type& rhs) { return divide_oper(rhs); }

    Type& operator+=(const Value& rhs) { return plus_oper(rhs); }
    Type& operator-=(const Value& rhs) { return minus_oper(rhs); }
    Type& operator*=(const Value& rhs) { return multiply_oper(rhs); }
    Type& operator/=(const Value& rhs) { return divide_oper(rhs); }

    template <typename Up = Tp>
    void print(std::ostream&,
               enable_if_t<trait::uses_value_storage<Up, Value>::value, int> = 0) const;

    template <typename Up = Tp>
    void print(std::ostream&,
               enable_if_t<!trait::uses_value_storage<Up, Value>::value, long> = 0) const;

    friend std::ostream& operator<<(std::ostream& os, const base_type& obj)
    {
        obj.print(os);
        return os;
    }

    /// serialization load (input)
    template <typename Archive, typename Up = Type,
              enable_if_t<!trait::custom_serialization<Up>::value, int> = 0>
    void load(Archive& ar, unsigned int);

    /// serialization store (output)
    template <typename Archive, typename Up = Type,
              enable_if_t<!trait::custom_serialization<Up>::value, int> = 0>
    void save(Archive& ar, unsigned int version) const;

    template <typename Vp, typename Up = Tp,
              enable_if_t<trait::sampler<Up>::value, int> = 0>
    static void add_sample(Vp&&);  /// add a sample

    /// get number of measurement
    TIMEMORY_NODISCARD TIMEMORY_INLINE int64_t get_laps() const { return laps; }
    TIMEMORY_NODISCARD TIMEMORY_INLINE auto get_iterator() const { return graph_itr; }
    TIMEMORY_INLINE void                    set_laps(int64_t v) { laps = v; }
    TIMEMORY_INLINE void set_iterator(graph_iterator itr) { graph_itr = itr; }

    using base_state::get_depth_change;
    using base_state::get_is_flat;
    using base_state::get_is_invalid;
    using base_state::get_is_on_stack;
    using base_state::get_is_running;
    using base_state::get_is_transient;
    using data_type::get_accum;
    using data_type::get_last;
    using data_type::get_value;

    using base_state::set_depth_change;
    using base_state::set_is_flat;
    using base_state::set_is_invalid;
    using base_state::set_is_on_stack;
    using base_state::set_is_running;
    using base_state::set_is_transient;
    using data_type::set_accum;
    using data_type::set_last;
    using data_type::set_value;

    decltype(auto) load() { return data_type::load(get_is_transient()); }
    TIMEMORY_NODISCARD decltype(auto) load() const
    {
        return data_type::load(get_is_transient());
    }

    static base_storage_type* get_storage();

protected:
    Type& plus_oper(const Type& rhs);
    Type& minus_oper(const Type& rhs);
    Type& multiply_oper(const Type& rhs);
    Type& divide_oper(const Type& rhs);

    Type& plus_oper(const Value& rhs);
    Type& minus_oper(const Value& rhs);
    Type& multiply_oper(const Value& rhs);
    Type& divide_oper(const Value& rhs);

    TIMEMORY_INLINE void plus(const base_type& rhs)
    {
        laps += rhs.laps;
        if(rhs.get_is_transient())
            set_is_transient(rhs.get_is_transient());
    }

    TIMEMORY_INLINE void minus(const base_type& rhs)
    {
        laps -= rhs.laps;
        if(rhs.get_is_transient())
            set_is_transient(rhs.get_is_transient());
    }

public:
    TIMEMORY_INLINE auto plus(crtp::base, const base_type& rhs) { this->plus(rhs); }
    TIMEMORY_INLINE auto minus(crtp::base, const base_type& rhs) { this->minus(rhs); }

protected:
    int64_t        laps      = 0;
    graph_iterator graph_itr = graph_iterator{ nullptr };

    using data_type::accum;
    using data_type::last;
    using data_type::value;

public:
    static constexpr bool timing_category_v = trait::is_timing_category<Type>::value;
    static constexpr bool memory_category_v = trait::is_memory_category<Type>::value;
    static constexpr bool timing_units_v    = trait::uses_timing_units<Type>::value;
    static constexpr bool memory_units_v    = trait::uses_memory_units<Type>::value;
    static constexpr bool percent_units_v   = trait::uses_percent_units<Type>::value;
    static constexpr auto ios_fixed         = std::ios_base::fixed;
    static constexpr auto ios_decimal       = std::ios_base::dec;
    static constexpr auto ios_showpoint     = std::ios_base::showpoint;
    static const fmtflags format_flags      = ios_fixed | ios_decimal | ios_showpoint;

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
    static const short precision = percent_units_v ? 1 : 3;
    static const short width     = percent_units_v ? 6 : 8;
#endif

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
, private base_state
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
    friend struct node::graph<Tp>;
    friend struct operation::init_storage<Tp>;
    friend struct operation::fini_storage<Tp>;
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
    friend struct operation::set_started<Tp>;
    friend struct operation::set_stopped<Tp>;
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
    /// get the opaque binding for user-bundle
    static opaque get_opaque(scope::config);

    void               set_started();
    void               set_stopped();
    void               reset();
    TIMEMORY_NODISCARD TIMEMORY_INLINE int64_t get_laps() const { return 0; }
    TIMEMORY_NODISCARD TIMEMORY_INLINE void* get_iterator() const { return nullptr; }

    TIMEMORY_INLINE void set_laps(int64_t) {}
    TIMEMORY_INLINE void set_iterator(void*) {}

    friend std::ostream& operator<<(std::ostream& os, const base_type&) { return os; }

    TIMEMORY_INLINE void get() {}
    TIMEMORY_INLINE void get(void*& ptr, size_t _typeid_hash) const;

    void operator+=(const base_type&) {}
    void operator-=(const base_type&) {}

    void operator+=(const Type&) {}
    void operator-=(const Type&) {}

    using base_state::get_depth_change;
    using base_state::get_is_flat;
    using base_state::get_is_invalid;
    using base_state::get_is_on_stack;
    using base_state::get_is_running;
    using base_state::get_is_transient;

    using base_state::set_depth_change;
    using base_state::set_is_flat;
    using base_state::set_is_invalid;
    using base_state::set_is_on_stack;
    using base_state::set_is_running;
    using base_state::set_is_transient;

protected:
    TIMEMORY_INLINE void plus(const base_type& rhs)
    {
        if(rhs.get_is_transient())
            set_is_transient(rhs.get_is_transient());
    }

    TIMEMORY_INLINE void minus(const base_type& rhs)
    {
        if(rhs.get_is_transient())
            set_is_transient(rhs.get_is_transient());
    }

public:
    TIMEMORY_INLINE auto plus(crtp::base, const base_type& rhs) { this->plus(rhs); }
    TIMEMORY_INLINE auto minus(crtp::base, const base_type& rhs) { this->minus(rhs); }

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
}  // namespace component
}  // namespace tim

#include "timemory/components/base/templates.hpp"
