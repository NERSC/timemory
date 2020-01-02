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

/** \file components/base.hpp
 * \headerfile components/base.hpp "timemory/components/base.hpp"
 * Defines the static polymorphic base for the components
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/data/statistics.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/storage.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//
//
//      base component class
//
//======================================================================================//

namespace component
{
template <typename _Tp, typename _Value>
struct base
{
public:
    static constexpr bool implements_storage_v = implements_storage<_Tp, _Value>::value;
    static constexpr bool has_secondary_data   = trait::secondary_data<_Tp>::value;
    static constexpr bool record_statistics_v  = trait::record_statistics<_Tp>::value;
    static constexpr bool is_component_type    = false;
    static constexpr bool is_auto_type         = false;
    static constexpr bool is_component         = true;

    using Type       = _Tp;
    using value_type = _Value;
    using accum_type =
        typename std::conditional<record_statistics_v, statistics<_Value>, _Value>::type;
    using this_type      = base<_Tp, _Value>;
    using base_type      = this_type;
    using storage_type   = impl::storage<_Tp, implements_storage_v>;
    using graph_iterator = typename storage_type::iterator;
    using state_t        = state<this_type>;

private:
    friend class impl::storage<_Tp, implements_storage_v>;
    friend class storage<_Tp>;

    friend struct operation::init_storage<_Tp>;
    friend struct operation::construct<_Tp>;
    friend struct operation::set_prefix<_Tp>;
    friend struct operation::pop_node<_Tp>;
    friend struct operation::record<_Tp>;
    friend struct operation::reset<_Tp>;
    friend struct operation::measure<_Tp>;
    friend struct operation::start<_Tp>;
    friend struct operation::stop<_Tp>;
    friend struct operation::minus<_Tp>;
    friend struct operation::plus<_Tp>;
    friend struct operation::multiply<_Tp>;
    friend struct operation::divide<_Tp>;
    friend struct operation::base_printer<_Tp>;
    friend struct operation::print<_Tp>;
    friend struct operation::print_storage<_Tp>;
    friend struct operation::copy<_Tp>;

    template <typename _Up, typename _Scope>
    friend struct operation::insert_node;

    template <typename _Up, typename Archive>
    friend struct operation::serialization;

    template <typename _Ret, typename _Lhs, typename _Rhs>
    friend struct operation::compose;

    static_assert(std::is_pointer<_Tp>::value == false, "Error pointer base type");

public:
    base()
    : is_running(false)
    , is_on_stack(false)
    , is_transient(false)
    , is_flat(false)
    , depth_change(false)
    , value(value_type())
    , accum(value_type())
    , laps(0)
    , graph_itr(graph_iterator{ nullptr })
    {
        if(!storage_type::is_finalizing())
        {
            static thread_local auto _storage = get_storage();
            consume_parameters(_storage);
        }
    }

    ~base() = default;

    explicit base(const this_type&) = default;
    explicit base(this_type&&)      = default;

    base& operator=(const this_type&) = default;
    base& operator=(this_type&&) = default;

public:
    static void global_init(storage_type*) {}
    static void thread_init(storage_type*) {}
    static void global_finalize(storage_type*) {}
    static void thread_finalize(storage_type*) {}
    template <typename _Archive>
    static void extra_serialization(_Archive&, const unsigned int)
    {}

public:
    static void initialize_storage()
    {
        static thread_local auto _instance = storage_type::instance();
        consume_parameters(_instance);
    }

    template <typename... _Args>
    static void configure(_Args&&...)
    {
        // this is generically allowable
        static_assert(sizeof...(_Args) == 0,
                      "Error! component::<Type>::configure not handled!");
    }

    //----------------------------------------------------------------------------------//
    /// type contains secondary data resembling the original data
    /// but should be another node entry in the graph. These types
    /// must provide a get_secondary() member function and that member function
    /// must return a pair-wise iterable container, e.g. std::map, of types:
    ///     - std::string
    ///     - value_type
    ///
    static void append(graph_iterator itr, const Type& rhs)
    {
        using has_secondary_type = typename trait::secondary_data<_Tp>::type;
        this_type::append_impl<value_type>(has_secondary_type{}, itr, rhs);
    }

public:
    //----------------------------------------------------------------------------------//
    // reset the values
    //
    void reset()
    {
        is_running   = false;
        is_on_stack  = false;
        is_transient = false;
        is_flat      = false;
        laps         = 0;
        value        = value_type();
        accum        = value_type();
    }

    //----------------------------------------------------------------------------------//
    // just record a measurment
    //
    void measure()
    {
        is_running   = false;
        is_transient = false;
        value        = Type::record();
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    bool start()
    {
        if(!is_running)
        {
            ++laps;
            static_cast<Type&>(*this).start();
            set_started();
            return true;
        }
        return false;
    }

    //----------------------------------------------------------------------------------//
    // stop
    //
    bool stop()
    {
        if(is_running)
        {
            static_cast<Type&>(*this).stop();
            set_stopped();
            return true;
        }
        return false;
    }

    //----------------------------------------------------------------------------------//
    // mark a point in the execution, by default, this does nothing
    //
    void mark_begin() {}

    //----------------------------------------------------------------------------------//
    // mark a point in the execution, by default, this does nothing
    //
    void mark_end() {}

    //----------------------------------------------------------------------------------//
    // set the firsts notify that start has been called
    //
    void set_started()
    {
        is_running   = true;
        is_transient = true;
    }

    //----------------------------------------------------------------------------------//
    // set the firsts notify that stop has been called
    //
    void set_stopped()
    {
        is_running   = false;
        is_transient = true;
    }

    //----------------------------------------------------------------------------------//
    // default get
    //
    value_type get() const { return (is_transient) ? value : accum; }

    //----------------------------------------------------------------------------------//
    // comparison operators
    //
    bool operator==(const this_type& rhs) const { return (load() == rhs.load()); }
    bool operator<(const this_type& rhs) const { return (load() < rhs.load()); }
    bool operator>(const this_type& rhs) const { return (load() > rhs.load()); }
    bool operator!=(const this_type& rhs) const { return !(*this == rhs); }
    bool operator<=(const this_type& rhs) const { return !(*this > rhs); }
    bool operator>=(const this_type& rhs) const { return !(*this < rhs); }

    // this_type operators
    //
    Type& operator+=(const this_type& rhs)
    {
        return operator+=(static_cast<const Type&>(rhs));
    }

    Type& operator-=(const this_type& rhs)
    {
        return operator-=(static_cast<const Type&>(rhs));
    }

    Type& operator*=(const this_type& rhs)
    {
        return operator*=(static_cast<const Type&>(rhs));
    }

    Type& operator/=(const this_type& rhs)
    {
        return operator/=(static_cast<const Type&>(rhs));
    }

    //----------------------------------------------------------------------------------//
    // Type operators
    //
    Type& operator+=(const Type& rhs)
    {
        value += rhs.value;
        accum += rhs.accum;
        return static_cast<Type&>(*this);
    }

    Type& operator-=(const Type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        return static_cast<Type&>(*this);
    }

    Type& operator*=(const Type& rhs)
    {
        value *= rhs.value;
        accum *= rhs.accum;
        return static_cast<Type&>(*this);
    }

    Type& operator/=(const Type& rhs)
    {
        value /= rhs.value;
        accum /= rhs.accum;
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // value type operators
    //
    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value), int> = 0>
    Type& operator+=(const value_type& rhs)
    {
        value += rhs;
        accum += rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value), int> = 0>
    Type& operator-=(const value_type& rhs)
    {
        value -= rhs;
        accum -= rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value), int> = 0>
    Type& operator*=(const value_type& rhs)
    {
        value *= rhs;
        accum *= rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value), int> = 0>
    Type& operator/=(const value_type& rhs)
    {
        value /= rhs;
        accum /= rhs;
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // value type operators
    //
    template <typename U = value_type, enable_if_t<!(std::is_pod<U>::value), int> = 0>
    Type& operator+=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator+=(rhs);
    }

    template <typename U = value_type, enable_if_t<!(std::is_pod<U>::value), int> = 0>
    Type& operator-=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator-=(rhs);
    }

    template <typename U = value_type, enable_if_t<!(std::is_pod<U>::value), int> = 0>
    Type& operator*=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator*=(rhs);
    }

    template <typename U = value_type, enable_if_t<!(std::is_pod<U>::value), int> = 0>
    Type& operator/=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator/=(rhs);
    }

    //----------------------------------------------------------------------------------//
    // friend operators
    //
    friend Type operator+(const this_type& lhs, const this_type& rhs)
    {
        return this_type(lhs) += rhs;
    }

    friend Type operator-(const this_type& lhs, const this_type& rhs)
    {
        return this_type(lhs) -= rhs;
    }

    friend Type operator*(const this_type& lhs, const this_type& rhs)
    {
        return this_type(lhs) *= rhs;
    }

    friend Type operator/(const this_type& lhs, const this_type& rhs)
    {
        return this_type(lhs) /= rhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        operation::base_printer<Type>(os, obj);
        return os;
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive, typename _Up = Type,
              enable_if_t<!(trait::split_serialization<_Up>::value), int> = 0>
    void serialize(Archive& ar, const unsigned int)
    {
        // operation::serialization<Type, Archive>(*this, ar, version);
        auto _data = static_cast<const Type&>(*this).get();
        ar(cereal::make_nvp("is_transient", is_transient), cereal::make_nvp("laps", laps),
           cereal::make_nvp("repr_data", _data), cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum), cereal::make_nvp("units", get_unit()),
           cereal::make_nvp("display_units", get_display_unit()));
    }

    const int64_t&    nlaps() const { return laps; }
    const value_type& get_value() const { return value; }
    const value_type& get_accum() const { return accum; }
    const bool&       get_is_transient() const { return is_transient; }

protected:
    const value_type& load() const { return (is_transient) ? accum : value; }

protected:
    //----------------------------------------------------------------------------------//
    // insert the node into the graph
    //
    template <typename _Scope, typename _Up = this_type,
              enable_if_t<(_Up::implements_storage_v), int>                 = 0,
              enable_if_t<!(std::is_same<_Scope, scope::flat>::value), int> = 0>
    void insert_node(const _Scope&, const int64_t& _hash)
    {
        if(!is_on_stack)
        {
            is_flat          = false;
            auto  _storage   = get_storage();
            auto  _beg_depth = _storage->depth();
            Type& obj        = static_cast<Type&>(*this);
            graph_itr        = _storage->template insert<_Scope>(obj, _hash);
            is_on_stack      = true;
            auto _end_depth  = _storage->depth();
            depth_change     = (_beg_depth < _end_depth);
            _storage->stack_push(&obj);
        }
    }

    template <typename _Scope, typename _Up = this_type,
              enable_if_t<(_Up::implements_storage_v), int>                = 0,
              enable_if_t<(std::is_same<_Scope, scope::flat>::value), int> = 0>
    void insert_node(const _Scope&, const int64_t& _hash)
    {
        if(!is_on_stack)
        {
            is_flat        = true;
            auto  _storage = get_storage();
            Type& obj      = static_cast<Type&>(*this);
            graph_itr      = _storage->template insert<_Scope>(obj, _hash);
            is_on_stack    = true;
            depth_change   = false;
            _storage->stack_push(&obj);
        }
    }

    template <typename _Scope, typename _Up = this_type,
              enable_if_t<!(_Up::implements_storage_v), int> = 0>
    void insert_node(const _Scope&, const int64_t&)
    {
        if(!is_on_stack)
        {
            is_flat        = true;
            auto  _storage = get_storage();
            Type& obj      = static_cast<Type&>(*this);
            is_on_stack    = true;
            _storage->stack_push(&obj);
        }
    }

    //----------------------------------------------------------------------------------//
    // pop the node off the graph
    //
    template <typename _Up = this_type, enable_if_t<(_Up::implements_storage_v), int> = 0>
    void pop_node()
    {
        if(is_on_stack)
        {
            Type& obj    = graph_itr->obj();
            Type& rhs    = static_cast<Type&>(*this);
            depth_change = false;

            if(storage_type::is_finalizing())
            {
                obj += rhs;
                obj.plus(rhs);
                Type::append(graph_itr, rhs);
            }
            else if(is_flat)
            {
                auto _storage = get_storage();

                obj += rhs;
                obj.plus(rhs);
                Type::append(graph_itr, rhs);
                _storage->stack_pop(&rhs);
            }
            else
            {
                auto _storage   = get_storage();
                auto _beg_depth = _storage->depth();

                obj += rhs;
                obj.plus(rhs);
                Type::append(graph_itr, rhs);
                if(_storage)
                {
                    _storage->pop();
                    _storage->stack_pop(&rhs);

                    auto _end_depth = _storage->depth();
                    depth_change    = (_beg_depth > _end_depth);
                }
            }
            obj.is_running = false;
            is_on_stack    = false;
        }
    }

    template <typename _Up                                   = this_type,
              enable_if_t<!(_Up::implements_storage_v), int> = 0>
    void pop_node()
    {
        if(is_on_stack)
        {
            auto  _storage = get_storage();
            Type& rhs      = static_cast<Type&>(*this);
            if(_storage)
                _storage->stack_pop(&rhs);
            is_on_stack = false;
        }
    }

    //----------------------------------------------------------------------------------//
    // initialize the storage
    //
    template <typename _Up = _Tp, typename _Vp = _Value,
              enable_if_t<(implements_storage<_Up, _Vp>::value), int> = 0>
    static bool init_storage(storage_type*& _instance)
    {
        if(!_instance)
        {
            static thread_local int32_t _count = 0;
            if(_count++ == 0)
                _instance = storage_type::instance();
        }

        if(!state_t::has_storage() && _instance)
            _instance->initialize();
        return state_t::has_storage();
    }

    template <typename _Up = _Tp, typename _Vp = _Value,
              enable_if_t<!(implements_storage<_Up, _Vp>::value), int> = 0>
    static bool init_storage(storage_type*&)
    {
        return true;
    }

    //----------------------------------------------------------------------------------//
    // create an instance without calling constructor
    //
    static Type dummy()
    {
        state_t::has_storage() = true;
        Type _fake{};
        return _fake;
    }

protected:
    void plus(const this_type& rhs)
    {
        laps += rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

    void minus(const this_type& rhs)
    {
        laps -= rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

    static void cleanup() {}

protected:
    bool           is_running   = false;
    bool           is_on_stack  = false;
    bool           is_transient = false;
    bool           is_flat      = false;
    bool           depth_change = false;
    value_type     value        = value_type();
    accum_type     accum        = accum_type();
    int64_t        laps         = 0;
    graph_iterator graph_itr    = graph_iterator{ nullptr };

    static storage_type*& get_storage()
    {
        static thread_local storage_type* _instance = nullptr;
        static thread_local bool          _inited   = init_storage(_instance);
        consume_parameters(_inited);
        return _instance;
    }

private:
    template <typename _Vp>
    static void append_impl(std::true_type, graph_iterator, const Type&);
    template <typename _Vp>
    static void append_impl(std::false_type, graph_iterator, const Type&);

public:
    static constexpr bool timing_category_v = trait::is_timing_category<Type>::value;
    static constexpr bool memory_category_v = trait::is_memory_category<Type>::value;
    static constexpr bool timing_units_v    = trait::uses_timing_units<Type>::value;
    static constexpr bool memory_units_v    = trait::uses_memory_units<Type>::value;
    static constexpr bool percent_units_v   = trait::uses_percent_units<Type>::value;

    static const short precision = (memory_units_v || percent_units_v) ? 1 : 3;
    static const short width     = (memory_units_v || percent_units_v) ? 6 : 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    static int64_t unit()
    {
        if(timing_units_v)
            return units::sec;
        else if(memory_units_v)
            return units::megabyte;
        else if(percent_units_v)
            return 1;

        return 1;
    }

    static std::string display_unit()
    {
        if(timing_units_v)
            return units::time_repr(unit());
        else if(memory_units_v)
            return units::mem_repr(unit());
        else if(percent_units_v)
            return "%";

        return "";
    }

    static short get_width()
    {
        static short _instance = Type::width;
        if(settings::width() >= 0)
            _instance = settings::width();

        if(timing_category_v && settings::timing_width() >= 0)
            _instance = settings::timing_width();
        else if(memory_category_v && settings::memory_width() >= 0)
            _instance = settings::memory_width();

        return _instance;
    }

    static short get_precision()
    {
        static short _instance = Type::precision;
        if(settings::precision() >= 0)
            _instance = settings::precision();

        if(timing_category_v && settings::timing_precision() >= 0)
            _instance = settings::timing_precision();
        else if(memory_category_v && settings::memory_precision() >= 0)
            _instance = settings::memory_precision();

        return _instance;
    }

    static std::ios_base::fmtflags get_format_flags()
    {
        static std::ios_base::fmtflags _instance = Type::format_flags;

        auto _set_scientific = [&]() {
            _instance &= (std::ios_base::fixed & std::ios_base::scientific);
            _instance |= (std::ios_base::scientific);
        };

        if(settings::scientific() ||
           (timing_category_v && settings::timing_scientific()) ||
           (memory_category_v && settings::memory_scientific()))
            _set_scientific();

        return _instance;
    }

    static int64_t get_unit()
    {
        static int64_t _instance = Type::unit();

        if(timing_units_v && settings::timing_units().length() > 0)
            _instance = std::get<1>(units::get_timing_unit(settings::timing_units()));
        else if(memory_units_v && settings::memory_units().length() > 0)
            _instance = std::get<1>(units::get_memory_unit(settings::memory_units()));

        return _instance;
    }

    static std::string get_display_unit()
    {
        static std::string _instance = Type::display_unit();

        if(timing_units_v && settings::timing_units().length() > 0)
            _instance = std::get<0>(units::get_timing_unit(settings::timing_units()));
        else if(memory_units_v && settings::memory_units().length() > 0)
            _instance = std::get<0>(units::get_memory_unit(settings::memory_units()));

        return _instance;
    }

    static std::string get_label()
    {
        static std::string _instance = Type::label();
        return _instance;
    }

    static std::string get_description()
    {
        static std::string _instance = Type::description();
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct base<_Tp, void>
{
public:
    static constexpr bool implements_storage_v = false;
    static constexpr bool has_secondary_data   = false;
    static constexpr bool record_statistics_v  = false;
    static constexpr bool is_component_type    = false;
    static constexpr bool is_auto_type         = false;
    static constexpr bool is_component         = true;

    using Type         = _Tp;
    using value_type   = void;
    using this_type    = base<_Tp, value_type>;
    using base_type    = this_type;
    using storage_type = impl::storage<_Tp, implements_storage_v>;

private:
    friend class impl::storage<_Tp, implements_storage_v>;

    friend struct operation::init_storage<_Tp>;
    friend struct operation::construct<_Tp>;
    friend struct operation::set_prefix<_Tp>;
    friend struct operation::pop_node<_Tp>;
    friend struct operation::record<_Tp>;
    friend struct operation::reset<_Tp>;
    friend struct operation::measure<_Tp>;
    friend struct operation::start<_Tp>;
    friend struct operation::stop<_Tp>;
    friend struct operation::minus<_Tp>;
    friend struct operation::plus<_Tp>;
    friend struct operation::multiply<_Tp>;
    friend struct operation::divide<_Tp>;
    friend struct operation::print<_Tp>;
    friend struct operation::print_storage<_Tp>;
    friend struct operation::copy<_Tp>;

    template <typename _Up, typename _Scope>
    friend struct operation::insert_node;

    template <typename _Up, typename Archive>
    friend struct operation::serialization;

    template <typename _Ret, typename _Lhs, typename _Rhs>
    friend struct operation::compose;

public:
    base()
    : is_running(false)
    , is_on_stack(false)
    , is_transient(false)
    {}

    ~base()                         = default;
    explicit base(const this_type&) = default;
    explicit base(this_type&&)      = default;
    base& operator=(const this_type&) = default;
    base& operator=(this_type&&) = default;

public:
    static void global_init(storage_type*) {}
    static void thread_init(storage_type*) {}
    static void global_finalize(storage_type*) {}
    static void thread_finalize(storage_type*) {}
    template <typename _Archive>
    static void extra_serialization(_Archive&, const unsigned int)
    {}

public:
    static void initialize_storage()
    {
        static thread_local auto _instance = storage_type::instance();
        consume_parameters(_instance);
    }

    template <typename... _Args>
    static void configure(_Args&&...)
    {
        // this is generically allowable
        static_assert(sizeof...(_Args) == 0,
                      "Error! component::<Type>::configure not handled!");
    }

    template <typename _GraphItr>
    static void append(_GraphItr, const Type&)
    {}

public:
    //----------------------------------------------------------------------------------//
    // reset the values
    //
    void reset()
    {
        is_running   = false;
        is_on_stack  = false;
        is_transient = false;
    }

    //----------------------------------------------------------------------------------//
    // just record a measurment
    //
    void measure()
    {
        is_running   = false;
        is_transient = false;
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    bool start()
    {
        if(!is_running)
        {
            set_started();
            static_cast<Type&>(*this).start();
            return true;
        }
        return false;
    }

    //----------------------------------------------------------------------------------//
    // stop
    //
    bool stop()
    {
        if(is_running)
        {
            static_cast<Type&>(*this).stop();
            set_stopped();
            return true;
        }
        return false;
    }

    //----------------------------------------------------------------------------------//
    // mark a point in the execution, by default, this does nothing
    //
    void mark_begin() {}

    //----------------------------------------------------------------------------------//
    // mark a point in the execution, by default, this does nothing
    //
    void mark_end() {}

    //----------------------------------------------------------------------------------//
    // set the firsts notify that start has been called
    //
    void set_started()
    {
        is_running   = true;
        is_transient = true;
    }

    //----------------------------------------------------------------------------------//
    // set the firsts notify that stop has been called
    //
    void set_stopped()
    {
        is_running   = false;
        is_transient = true;
    }

    friend std::ostream& operator<<(std::ostream& os, const this_type&) { return os; }

    int64_t nlaps() const { return 0; }

    void* get() { return nullptr; }

private:
    //----------------------------------------------------------------------------------//
    // insert the node into the graph
    //
    template <typename _Scope = scope::process, typename... _Args>
    void insert_node(const _Scope&, _Args&&...)
    {
        if(!is_on_stack)
        {
            // auto  _storage = get_storage();
            // Type& obj      = static_cast<Type&>(*this);
            is_on_stack = true;
            // _storage->stack_push(&obj);
        }
    }

    //----------------------------------------------------------------------------------//
    // pop the node off the graph
    //
    void pop_node()
    {
        if(is_on_stack)
        {
            // auto  _storage = get_storage();
            // Type& rhs      = static_cast<Type&>(*this);
            is_on_stack = false;
            // _storage->stack_pop(&rhs);
        }
    }

protected:
    void plus(const this_type& rhs)
    {
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

    void minus(const this_type& rhs)
    {
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
    }

    static void cleanup() {}

protected:
    bool is_running   = false;
    bool is_on_stack  = false;
    bool is_transient = false;

public:
    static std::string get_label()
    {
        static std::string _instance = Type::label();
        return _instance;
    }

    static std::string get_description()
    {
        static std::string _instance = Type::description();
        return _instance;
    }
};

//----------------------------------------------------------------------------------//
/// type contains secondary data resembling the original data
/// but should be another node entry in the graph. These types
/// must provide a get_secondary() member function and that member function
/// must return a pair-wise iterable container, e.g. std::map, of types:
///     - std::string
///     - value_type
///
template <typename _Tp, typename _Value>
template <typename _Vp>
void
base<_Tp, _Value>::append_impl(std::true_type, graph_iterator itr, const Type& rhs)
{
    static_assert(trait::secondary_data<_Tp>::value,
                  "append_impl should not be compiled");
    static_assert(std::is_same<_Vp, _Value>::value, "Type mismatch");

    auto _storage          = storage_type::instance();
    using string_t         = std::string;
    using secondary_data_t = std::tuple<graph_iterator, const string_t&, _Vp>;
    for(const auto& dat : rhs.get_secondary())
        _storage->append(secondary_data_t{ itr, dat.first, dat.second });
}

//----------------------------------------------------------------------------------//
//  type does not contain secondary data
//
template <typename _Tp, typename _Value>
template <typename _Vp>
void
base<_Tp, _Value>::append_impl(std::false_type, graph_iterator, const Type&)
{}

}  // namespace component
}  // namespace tim

//======================================================================================//
