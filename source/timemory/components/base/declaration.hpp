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

/**
 * \headerfile "timemory/components/base/declaration.hpp"
 * \brief Declares the static polymorphic base for the components
 *
 */

#pragma once

#include "timemory/components/base/types.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/serializer.hpp"

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
struct empty_base
{};
//
//======================================================================================//
//
//              dynamic polymorphism base class for all components
//
//======================================================================================//
//
struct dynamic_base
{
    TIMEMORY_DEFAULT_OBJECT(dynamic_base)

    virtual void          start()                                         = 0;
    virtual void          stop()                                          = 0;
    virtual void          get_opaque_data(void*& ptr, size_t _hash) const = 0;
    virtual dynamic_base* create() const                                  = 0;

    // virtual void init() {}
    // virtual void cleanup() {}
    // virtual void start(const std::string&, bool, bool) { start(); }
};
//
//======================================================================================//
//
//                          static polymorphism base
//          base component class for all components with non-void types
//
//======================================================================================//
//
template <typename Tp, typename Value>
struct base : public trait::dynamic_base<Tp>::type
{
    using EmptyT = std::tuple<>;
    template <typename U>
    using vector_t = std::vector<U>;

public:
    static constexpr bool has_accum_v        = trait::base_has_accum<Tp>::value;
    static constexpr bool has_last_v         = trait::base_has_last<Tp>::value;
    static constexpr bool has_secondary_data = trait::secondary_data<Tp>::value;
    static constexpr bool is_sampler_v       = trait::sampler<Tp>::value;
    static constexpr bool is_component_type  = false;
    static constexpr bool is_auto_type       = false;
    static constexpr bool is_component       = true;

    using Type         = Tp;
    using value_type   = Value;
    using accum_type   = conditional_t<has_accum_v, value_type, EmptyT>;
    using last_type    = conditional_t<has_last_v, value_type, EmptyT>;
    using dynamic_type = typename trait::dynamic_base<Tp>::type;
    using cache_type   = typename trait::cache<Tp>::type;

    using this_type         = Tp;
    using base_type         = base<Tp, Value>;
    using unit_type         = typename trait::units<Tp>::type;
    using display_unit_type = typename trait::units<Tp>::display_type;
    using storage_type      = storage<Tp, Value>;
    using base_storage_type = tim::base::storage;
    using graph_iterator    = typename storage_type::iterator;
    using state_t           = state<this_type>;
    using statistics_policy = policy::record_statistics<Tp, Value>;
    using fmtflags          = std::ios_base::fmtflags;

private:
    friend class impl::storage<Tp, trait::implements_storage<Tp, Value>::value>;
    friend class storage<Tp, Value>;
    friend struct node::graph<Tp>;

    friend struct operation::init_storage<Tp>;
    friend struct operation::cache<Tp>;
    friend struct operation::construct<Tp>;
    friend struct operation::set_prefix<Tp>;
    friend struct operation::insert_node<Tp>;
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
    base();
    virtual ~base() = default;

    explicit base(const base_type&)     = default;
    explicit base(base_type&&) noexcept = default;

    base& operator=(const base_type&) = default;
    base& operator=(base_type&&) noexcept = default;

public:
    template <typename Archive>
    static void extra_serialization(Archive&, const unsigned int)
    {}

public:
    template <typename... Args>
    static void configure(Args&&...)
    {}

public:
    void reset();    /// reset the values
    void measure();  /// just record a measurment
    void start();    /// start measurement
    void stop();     /// stop measurement

    auto start(crtp::base) { this->start(); }
    auto stop(crtp::base) { this->stop(); }

    void set_started();  // store that start has been called
    void set_stopped();  // store that stop has been called

    void get(void*& ptr, size_t typeid_hash) const;    /// assign type to a pointer
    auto get() const { return this->load(); }          /// default get routine
    auto get_display() const { return this->load(); }  /// default display routine

    void          get_opaque_data(void*& ptr,
                                  size_t typeid_hash) const;  /// assign get() to a pointer
    dynamic_type* create() const;

    template <typename Up = Tp, typename Vp = Value,
              enable_if_t<(trait::implements_storage<Up, Vp>::value), int> = 0>
    void print(std::ostream&) const;

    template <typename Up = Tp, typename Vp = Value,
              enable_if_t<!(trait::implements_storage<Up, Vp>::value), int> = 0>
    void print(std::ostream&) const;

    bool operator<(const base_type& rhs) const { return (load() < rhs.load()); }
    bool operator>(const base_type& rhs) const { return (load() > rhs.load()); }
    bool operator<=(const base_type& rhs) const { return !(*this > rhs); }
    bool operator>=(const base_type& rhs) const { return !(*this < rhs); }

    Type& operator+=(const base_type& rhs) { return plus_oper(rhs); }
    Type& operator-=(const base_type& rhs) { return minus_oper(rhs); }
    Type& operator*=(const base_type& rhs) { return multiply_oper(rhs); }
    Type& operator/=(const base_type& rhs) { return divide_oper(rhs); }

    Type& operator+=(const Type& rhs) { return plus_oper(rhs); }
    Type& operator-=(const Type& rhs) { return minus_oper(rhs); }
    Type& operator*=(const Type& rhs) { return multiply_oper(rhs); }
    Type& operator/=(const Type& rhs) { return divide_oper(rhs); }

    Type& operator+=(const Value& rhs) { return plus_oper(rhs); }
    Type& operator-=(const Value& rhs) { return minus_oper(rhs); }
    Type& operator*=(const Value& rhs) { return multiply_oper(rhs); }
    Type& operator/=(const Value& rhs) { return divide_oper(rhs); }

    friend std::ostream& operator<<(std::ostream& os, const base_type& obj)
    {
        obj.print(os);
        return os;
    }

    // serialization load (input)
    template <typename Archive, typename Up = Type,
              enable_if_t<!(trait::custom_serialization<Up>::value), int> = 0>
    void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, const unsigned int);

    // serialization store (output)
    template <typename Archive, typename Up = Type,
              enable_if_t<!(trait::custom_serialization<Up>::value), int> = 0>
    void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const unsigned int version) const;

    int64_t           get_laps() const { return laps; }
    const value_type& get_value() const { return value; }
    const accum_type& get_accum() const { return accum; }
    const last_type&  get_last() const { return last; }
    bool              get_is_transient() const { return (laps > 0 || is_transient); }
    auto              get_is_running() const { return is_running; }
    auto              get_is_on_stack() const { return is_on_stack; }
    auto              get_is_flat() const { return is_flat; }
    auto              get_depth_change() const { return depth_change; }

    void set_laps(int64_t v) { laps = v; }
    void set_value(value_type v) { value = v; }
    void set_accum(accum_type v) { accum = v; }
    void set_last(last_type v) { last = v; }
    void set_is_transient(bool v) { is_transient = v; }

    template <typename Vp, typename Up = Tp,
              enable_if_t<(trait::sampler<Up>::value), int> = 0>
    static void add_sample(Vp&&);  /// add a sample

    void set_iterator(graph_iterator itr) { graph_itr = itr; }
    auto get_iterator() const { return graph_itr; }

protected:
    static base_storage_type* get_storage();
    static void               cleanup() {}

    template <typename Up = Tp, enable_if_t<(trait::base_has_accum<Up>::value), int> = 0>
    value_type& load();
    template <typename Up = Tp, enable_if_t<(trait::base_has_accum<Up>::value), int> = 0>
    const value_type& load() const;

    template <typename Up = Tp, enable_if_t<!(trait::base_has_accum<Up>::value), int> = 0>
    value_type& load();
    template <typename Up = Tp, enable_if_t<!(trait::base_has_accum<Up>::value), int> = 0>
    const value_type& load() const;

    Type& plus_oper(const base_type& rhs);
    Type& minus_oper(const base_type& rhs);
    Type& multiply_oper(const base_type& rhs);
    Type& divide_oper(const base_type& rhs);

    Type& plus_oper(const Type& rhs);
    Type& minus_oper(const Type& rhs);
    Type& multiply_oper(const Type& rhs);
    Type& divide_oper(const Type& rhs);

    Type& plus_oper(const Value& rhs);
    Type& minus_oper(const Value& rhs);
    Type& multiply_oper(const Value& rhs);
    Type& divide_oper(const Value& rhs);

    void plus(const base_type& rhs)
    {
        laps += rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

    void minus(const base_type& rhs)
    {
        laps -= rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

public:
    auto plus(crtp::base, const base_type& rhs) { this->plus(rhs); }
    auto minus(crtp::base, const base_type& rhs) { this->minus(rhs); }

    static Type dummy();  // create an instance

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
    static const short    precision         = (percent_units_v) ? 1 : 3;
    static const short    width             = (percent_units_v) ? 6 : 8;
    static const fmtflags format_flags      = ios_fixed | ios_decimal | ios_showpoint;

    template <typename Up = Type, typename UnitT = typename trait::units<Up>::type,
              enable_if_t<(std::is_same<UnitT, int64_t>::value), int> = 0>
    static int64_t unit();

    template <typename Up = Type, typename UnitT = typename Up::display_unit_type,
              enable_if_t<(std::is_same<UnitT, std::string>::value), int> = 0>
    static std::string display_unit();

    template <typename Up = Type, typename UnitT = typename trait::units<Up>::type,
              enable_if_t<(std::is_same<UnitT, int64_t>::value), int> = 0>
    static int64_t get_unit();

    template <typename Up = Type, typename UnitT = typename Up::display_unit_type,
              enable_if_t<(std::is_same<UnitT, std::string>::value), int> = 0>
    static std::string get_display_unit();

    static short                   get_width();
    static short                   get_precision();
    static std::ios_base::fmtflags get_format_flags();
    static std::string             label();
    static std::string             description();
    static std::string             get_label();
    static std::string             get_description();
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
struct base<Tp, void> : public trait::dynamic_base<Tp>::type
{
    using EmptyT = std::tuple<>;

public:
    static constexpr bool has_secondary_data = false;
    static constexpr bool is_sampler_v       = trait::sampler<Tp>::value;
    static constexpr bool is_component_type  = false;
    static constexpr bool is_auto_type       = false;
    static constexpr bool is_component       = true;

    using Type             = Tp;
    using value_type       = void;
    using sample_type      = EmptyT;
    using sample_list_type = EmptyT;
    using dynamic_type     = typename trait::dynamic_base<Tp>::type;
    using cache_type       = typename trait::cache<Tp>::type;

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
    friend struct operation::insert_node<Tp>;
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
    base();
    virtual ~base()                     = default;
    explicit base(const base_type&)     = default;
    explicit base(base_type&&) noexcept = default;
    base& operator=(const base_type&) = default;
    base& operator=(base_type&&) noexcept = default;

public:
    template <typename Archive>
    static void extra_serialization(Archive&, const unsigned int)
    {}

public:
    template <typename... Args>
    static void configure(Args&&...)
    {}

public:
    void reset();    // reset the values
    void measure();  // just record a measurment
    void start();
    void stop();

    auto start(crtp::base) { this->start(); }
    auto stop(crtp::base) { this->stop(); }

    template <typename CacheT                                     = cache_type,
              enable_if_t<!concepts::is_null_type<CacheT>::value> = 0>
    void start(const CacheT&);
    template <typename CacheT                                     = cache_type,
              enable_if_t<!concepts::is_null_type<CacheT>::value> = 0>
    void stop(const CacheT&);

    void set_started();
    void set_stopped();

    friend std::ostream& operator<<(std::ostream& os, const base_type&) { return os; }

    void get() {}
    void get(void*& ptr, size_t typeid_hash) const;

    void          get_opaque_data(void*& ptr, size_t typeid_hash) const;
    dynamic_type* create() const;

    int64_t get_laps() const { return 0; }
    auto    get_is_running() const { return is_running; }
    auto    get_is_on_stack() const { return is_on_stack; }
    auto    get_is_transient() const { return is_transient; }

    // used by operation::finalize::print<Type>
    void operator-=(const base_type&) {}
    void operator-=(const Type&) {}

protected:
    static void cleanup() {}

    static Type dummy() { return Tp{}; }

    void plus(const base_type& rhs)
    {
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

    void minus(const base_type& rhs)
    {
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

public:
    auto plus(crtp::base, const base_type& rhs) { this->plus(rhs); }
    auto minus(crtp::base, const base_type& rhs) { this->minus(rhs); }

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
}  // namespace component
//
//----------------------------------------------------------------------------------//
//
}  // namespace tim
