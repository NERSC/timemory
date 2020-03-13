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
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"

#include <array>

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
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
struct base
{
    using EmptyT = std::tuple<>;
    template <typename U>
    using vector_t = std::vector<U>;

public:
    static constexpr bool has_accum_v          = trait::base_has_accum<Tp>::value;
    static constexpr bool has_last_v           = trait::base_has_last<Tp>::value;
    static constexpr bool implements_storage_v = implements_storage<Tp, Value>::value;
    static constexpr bool has_secondary_data   = trait::secondary_data<Tp>::value;
    static constexpr bool is_sampler_v         = trait::sampler<Tp>::value;
    static constexpr bool is_component_type    = false;
    static constexpr bool is_auto_type         = false;
    static constexpr bool is_component         = true;

    using Type             = Tp;
    using value_type       = Value;
    using accum_type       = conditional_t<has_accum_v, value_type, EmptyT>;
    using last_type        = conditional_t<has_last_v, value_type, EmptyT>;
    using sample_type      = conditional_t<is_sampler_v, operation::sample<Tp>, EmptyT>;
    using sample_list_type = conditional_t<is_sampler_v, vector_t<sample_type>, EmptyT>;

    using this_type         = Tp;
    using base_type         = base<Tp, Value>;
    using unit_type         = typename trait::units<Tp>::type;
    using display_unit_type = typename trait::units<Tp>::display_type;
    using storage_type      = impl::storage<Tp, implements_storage_v>;
    using graph_iterator    = typename storage_type::iterator;
    using state_t           = state<this_type>;
    using statistics_policy = policy::record_statistics<Tp, Value>;

private:
    friend class impl::storage<Tp, implements_storage_v>;
    friend class storage<Tp>;

    friend struct operation::init_storage<Tp>;
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

    template <typename _Ret, typename _Lhs, typename _Rhs>
    friend struct operation::compose;

    static_assert(std::is_pointer<Tp>::value == false, "Error pointer base type");

public:
    base()
    {
        if(!storage_type::is_finalizing())
        {
            static thread_local auto _storage = get_storage();
            consume_parameters(_storage);
        }
    }

    ~base() = default;

    explicit base(const base_type&) = default;
    explicit base(base_type&&)      = default;

    base& operator=(const base_type&) = default;
    base& operator=(base_type&&) = default;

public:
    static void global_init(storage_type*) {}
    static void thread_init(storage_type*) {}
    static void global_finalize(storage_type*) {}
    static void thread_finalize(storage_type*) {}
    template <typename Archive>
    static void extra_serialization(Archive&, const unsigned int)
    {}

public:
    template <typename... _Args>
    static void configure(_Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    /// type contains secondary data resembling the original data
    /// but should be another node entry in the graph. These types
    /// must provide a get_secondary() member function and that member function
    /// must return a pair-wise iterable container, e.g. std::map, of types:
    ///     - std::string
    ///     - value_type
    ///
    template <typename T = Type>
    static void append(graph_iterator itr, const T& rhs)
    {
        auto _storage = storage_type::instance();
        if(_storage)
            operation::add_secondary<T>(_storage, itr, rhs);
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
        depth_change = false;
        laps         = 0;
        value        = value_type{};
        accum        = accum_type{};
        last         = last_type{};
        samples      = sample_type{};
    }

    //----------------------------------------------------------------------------------//
    // just record a measurment
    //
    void measure()
    {
        is_transient                = false;
        Type&                   obj = static_cast<Type&>(*this);
        operation::record<Type> m(obj);
    }

    //----------------------------------------------------------------------------------//
    // sample statistics
    //
    void sample() {}

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        if(!is_running)
        {
            set_started();
            static_cast<Type&>(*this).start();
        }
    }

    //----------------------------------------------------------------------------------//
    // stop
    //
    void stop()
    {
        if(is_running)
        {
            set_stopped();
            ++laps;
            static_cast<Type&>(*this).stop();
        }
    }

    /*
    //----------------------------------------------------------------------------------//
    // mark a point in the execution, by default, this does nothing
    //
    void mark_begin() {}

    //----------------------------------------------------------------------------------//
    // mark a point in the execution, by default, this does nothing
    //
    void mark_end() {}

    //----------------------------------------------------------------------------------//
    // store a value, by default, this does nothing
    //
    void store() {}
    */

    //----------------------------------------------------------------------------------//
    // set the firsts notify that start has been called
    //
    void set_started() { is_running = true; }

    //----------------------------------------------------------------------------------//
    // set the firsts notify that stop has been called
    //
    void set_stopped()
    {
        is_running   = false;
        is_transient = true;
    }

    //----------------------------------------------------------------------------------//
    // assign the type to a pointer
    //
    void get(void*& ptr, size_t typeid_hash) const
    {
        static size_t this_typeid_hash = std::hash<std::string>()(demangle<Type>());
        if(!ptr && typeid_hash == this_typeid_hash)
            ptr = reinterpret_cast<void*>(const_cast<base_type*>(this));
    }

    //----------------------------------------------------------------------------------//
    // default get
    //
    auto get() const { return this->load(); }

    //----------------------------------------------------------------------------------//
    // default get_display if not defined by type
    //
    auto get_display() const { return this->load(); }

    //----------------------------------------------------------------------------------//
    // add a sample
    //
    template <typename Up = Tp, enable_if_t<(Up::is_sampler_v), int> = 0>
    void add_sample(sample_type&& _sample)
    {
        samples.emplace_back(std::forward<sample_type>(_sample));
    }

    //----------------------------------------------------------------------------------//
    // comparison operators
    //
    bool operator<(const base_type& rhs) const { return (load() < rhs.load()); }
    bool operator>(const base_type& rhs) const { return (load() > rhs.load()); }
    bool operator<=(const base_type& rhs) const { return !(*this > rhs); }
    bool operator>=(const base_type& rhs) const { return !(*this < rhs); }

    //----------------------------------------------------------------------------------//
    // base_type operators
    //
    Type& operator+=(const base_type& rhs)
    {
        return operator+=(static_cast<const Type&>(rhs));
    }

    Type& operator-=(const base_type& rhs)
    {
        return operator-=(static_cast<const Type&>(rhs));
    }

    Type& operator*=(const base_type& rhs)
    {
        return operator*=(static_cast<const Type&>(rhs));
    }

    Type& operator/=(const base_type& rhs)
    {
        return operator/=(static_cast<const Type&>(rhs));
    }

    //----------------------------------------------------------------------------------//
    // Type operators
    //
    Type& operator+=(const Type& rhs)
    {
        math::plus(value, rhs.value);
        math::plus(accum, rhs.accum);
        return static_cast<Type&>(*this);
    }

    Type& operator-=(const Type& rhs)
    {
        math::minus(value, rhs.value);
        math::minus(accum, rhs.accum);
        return static_cast<Type&>(*this);
    }

    Type& operator*=(const Type& rhs)
    {
        math::multiply(value, rhs.value);
        math::multiply(accum, rhs.accum);
        return static_cast<Type&>(*this);
    }

    Type& operator/=(const Type& rhs)
    {
        math::divide(value, rhs.value);
        math::divide(accum, rhs.accum);
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // value type operators
    //
    Type& operator+=(const value_type& rhs)
    {
        math::plus(value, rhs);
        math::plus(accum, rhs);
        return static_cast<Type&>(*this);
    }

    Type& operator-=(const value_type& rhs)
    {
        math::minus(value, rhs);
        math::minus(accum, rhs);
        return static_cast<Type&>(*this);
    }

    Type& operator*=(const value_type& rhs)
    {
        math::multiply(value, rhs);
        math::multiply(accum, rhs);
        return static_cast<Type&>(*this);
    }

    Type& operator/=(const value_type& rhs)
    {
        math::divide(value, rhs);
        math::divide(accum, rhs);
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // friend operators
    //
    friend Type operator+(const base_type& lhs, const base_type& rhs)
    {
        return this_type(static_cast<const Type&>(lhs)) += static_cast<const Type&>(rhs);
    }

    friend Type operator-(const base_type& lhs, const base_type& rhs)
    {
        return this_type(static_cast<const Type&>(lhs)) -= static_cast<const Type&>(rhs);
    }

    friend Type operator*(const base_type& lhs, const base_type& rhs)
    {
        return this_type(static_cast<const Type&>(lhs)) *= static_cast<const Type&>(rhs);
    }

    friend Type operator/(const base_type& lhs, const base_type& rhs)
    {
        return this_type(static_cast<const Type&>(lhs)) /= static_cast<const Type&>(rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const base_type& obj)
    {
        operation::base_printer<Type>(os, static_cast<const Type&>(obj));
        return os;
    }

    //----------------------------------------------------------------------------------//
    // serialization load (input)
    //
    template <typename Archive, typename Up = Type,
              enable_if_t<!(trait::custom_serialization<Up>::value), int> = 0>
    void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar(cereal::make_nvp("is_transient", is_transient),
           cereal::make_nvp("laps", laps),
           cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum),
           cereal::make_nvp("last", last),
           cereal::make_nvp("samples", samples));
        // clang-format on
    }

    //----------------------------------------------------------------------------------//
    // serialization store (output)
    //
    template <typename Archive, typename Up = Type,
              enable_if_t<!(trait::custom_serialization<Up>::value), int> = 0>
    void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const unsigned int version) const
    {
        operation::serialization<Type>(static_cast<const Type&>(*this), ar, version);
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    /*
    template <typename Archive, typename Up = Type,
              enable_if_t<!(trait::custom_serialization<Up>::value), int> = 0>
    void serialize(Archive& ar, const unsigned int version)
    {
        // operation::serialization<Type>(static_cast<Type&>(*this), ar, version);
        // clang-format off
        ar(cereal::make_nvp("is_transient", is_transient),
           cereal::make_nvp("laps", laps),
           cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum),
           cereal::make_nvp("last", last),
           cereal::make_nvp("samples", samples),
           cereal::make_nvp("repr_data", static_cast<Type&>(*this).get()),
           cereal::make_nvp("repr_display", static_cast<Type&>(*this).get_display()),
           cereal::make_nvp("units", Type::get_unit()),
           cereal::make_nvp("display_units", Type::get_display_unit()));
        // clang-format on
    }
    */

    int64_t           nlaps() const { return laps; }
    int64_t           get_laps() const { return laps; }
    const value_type& get_value() const { return value; }
    const accum_type& get_accum() const { return accum; }
    const last_type&  get_last() const { return last; }
    const bool&       get_is_transient() const { return is_transient; }
    sample_list_type  get_samples() const { return samples; }

protected:
    template <typename Up = Tp, enable_if_t<(Up::has_accum_v), int> = 0>
    const value_type& load() const
    {
        return (is_transient) ? accum : value;
    }

    template <typename Up = Tp, enable_if_t<!(Up::has_accum_v), int> = 0>
    const value_type& load() const
    {
        return value;
    }

protected:
    //----------------------------------------------------------------------------------//
    // insert the node into the graph
    //
    template <typename Scope, typename Up = base_type,
              enable_if_t<(Up::implements_storage_v), int>                = 0,
              enable_if_t<(std::is_same<Scope, scope::tree>::value), int> = 0>
    void insert_node(Scope&&, const int64_t& _hash)
    {
        if(!is_on_stack)
        {
            is_flat          = false;
            auto  _storage   = get_storage();
            auto  _beg_depth = _storage->depth();
            Type& obj        = static_cast<Type&>(*this);
            graph_itr        = _storage->template insert<Scope>(obj, _hash);
            is_on_stack      = true;
            auto _end_depth  = _storage->depth();
            depth_change     = (_beg_depth < _end_depth);
            _storage->stack_push(&obj);
        }
    }

    template <typename Scope, typename Up = base_type,
              enable_if_t<(Up::implements_storage_v), int>                 = 0,
              enable_if_t<!(std::is_same<Scope, scope::tree>::value), int> = 0>
    void insert_node(Scope&&, const int64_t& _hash)
    {
        if(!is_on_stack)
        {
            is_flat        = std::is_same<Scope, scope::flat>::value;
            auto  _storage = get_storage();
            Type& obj      = static_cast<Type&>(*this);
            graph_itr      = _storage->template insert<Scope>(obj, _hash);
            is_on_stack    = true;
            depth_change   = std::is_same<Scope, scope::timeline>::value;
            _storage->stack_push(&obj);
        }
    }

    template <typename Scope, typename Up = base_type,
              enable_if_t<!(Up::implements_storage_v), int> = 0>
    void insert_node(Scope&&, const int64_t&)
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
    template <typename Up = base_type, enable_if_t<(Up::implements_storage_v), int> = 0>
    void pop_node()
    {
        if(is_on_stack)
        {
            Type& obj     = graph_itr->obj();
            auto& stats   = graph_itr->stats();
            Type& rhs     = static_cast<Type&>(*this);
            depth_change  = false;
            auto _storage = get_storage();

            if(storage_type::is_finalizing())
            {
                obj += rhs;
                obj.plus(rhs);
                operation::add_secondary<Type>(_storage, graph_itr, rhs);
                operation::add_statistics<Type>(rhs, stats);
            }
            else if(is_flat)
            {
                obj += rhs;
                obj.plus(rhs);
                operation::add_secondary<Type>(_storage, graph_itr, rhs);
                operation::add_statistics<Type>(rhs, stats);
                _storage->stack_pop(&rhs);
            }
            else
            {
                auto _beg_depth = _storage->depth();
                obj += rhs;
                obj.plus(rhs);
                operation::add_secondary<Type>(_storage, graph_itr, rhs);
                operation::add_statistics<Type>(rhs, stats);
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

    template <typename Up = base_type, enable_if_t<!(Up::implements_storage_v), int> = 0>
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
    template <typename Up = Tp, typename Vp = Value,
              enable_if_t<(implements_storage<Up, Vp>::value), int> = 0>
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

    template <typename Up = Tp, typename Vp = Value,
              enable_if_t<!(implements_storage<Up, Vp>::value), int> = 0>
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

    static void cleanup() {}

protected:
    bool             is_running   = false;
    bool             is_on_stack  = false;
    bool             is_transient = false;
    bool             is_flat      = false;
    bool             depth_change = false;
    int64_t          laps         = 0;
    value_type       value        = value_type{};
    accum_type       accum        = accum_type{};
    last_type        last         = last_type{};
    sample_list_type samples      = sample_type{};
    graph_iterator   graph_itr    = graph_iterator{ nullptr };

    static storage_type*& get_storage()
    {
        static thread_local storage_type* _instance = nullptr;
        static thread_local bool          _inited   = init_storage(_instance);
        consume_parameters(_inited);
        return _instance;
    }

public:
    using fmtflags = std::ios_base::fmtflags;

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

    template <typename Up = Type, typename _Unit = typename trait::units<Up>::type,
              enable_if_t<(std::is_same<_Unit, int64_t>::value), int> = 0>
    static int64_t unit()
    {
        IF_CONSTEXPR(timing_units_v)
        return units::sec;
        else IF_CONSTEXPR(memory_units_v) return units::megabyte;
        else IF_CONSTEXPR(percent_units_v) return 1;

        return 1;
    }

    template <typename Up = Type, typename _Unit = typename Up::display_unit_type,
              enable_if_t<(std::is_same<_Unit, std::string>::value), int> = 0>
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

    template <typename Up = Type, typename _Unit = typename trait::units<Up>::type,
              enable_if_t<(std::is_same<_Unit, int64_t>::value), int> = 0>
    static int64_t get_unit()
    {
        static int64_t _instance = []() {
            auto _value = Type::unit();
            if(timing_units_v && settings::timing_units().length() > 0)
                _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
            else if(memory_units_v && settings::memory_units().length() > 0)
                _value = std::get<1>(units::get_memory_unit(settings::memory_units()));
            return _value;
        }();
        return _instance;
    }

    template <typename Up = Type, typename _Unit = typename Up::display_unit_type,
              enable_if_t<(std::is_same<_Unit, std::string>::value), int> = 0>
    static std::string get_display_unit()
    {
        static std::string _instance = Type::display_unit();

        if(timing_units_v && settings::timing_units().length() > 0)
            _instance = std::get<0>(units::get_timing_unit(settings::timing_units()));
        else if(memory_units_v && settings::memory_units().length() > 0)
            _instance = std::get<0>(units::get_memory_unit(settings::memory_units()));

        return _instance;
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

        if(!percent_units_v && (settings::scientific() ||
                                (timing_category_v && settings::timing_scientific()) ||
                                (memory_category_v && settings::memory_scientific())))
            _set_scientific();

        return _instance;
    }

    //
    // generate a default output filename from
    // (potentially demangled) typeid(Type).name() and strip out
    // namespace and any template parameters + replace any spaces
    // with underscores
    //
    static std::string label()
    {
        std::string       _label = demangle<Type>();
        std::stringstream msg;
        msg << "Warning! " << _label << " does not provide a custom label!";
#if defined(DEBUG)
        // throw error when debugging
        throw std::runtime_error(msg.str().c_str());
#else
        // warn when not debugging
        if(settings::debug())
            std::cerr << msg.str() << std::endl;
#endif
        if(_label.find(':') != std::string::npos)
            _label = _label.substr(_label.find_last_of(':'));
        if(_label.find('<') != std::string::npos)
            _label = _label.substr(0, _label.find_first_of('<'));
        while(_label.find(' ') != std::string::npos)
            _label = _label.replace(_label.find(' '), 1, "_");
        while(_label.find("__") != std::string::npos)
            _label = _label.replace(_label.find("__"), 2, "_");
        return _label;
    }

    static std::string description()
    {
        std::string       _label = demangle<Type>();
        std::stringstream msg;
        msg << "Warning! " << _label << " does not provide a custom description!";
#if defined(DEBUG)
        // throw error when debugging
        throw std::runtime_error(msg.str().c_str());
#else
        // warn when not debugging
        if(settings::debug())
            std::cerr << msg.str() << std::endl;
#endif
        return _label;
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
//
//              void overload of base
//
//--------------------------------------------------------------------------------------//

template <typename Tp>
struct base<Tp, void>
{
    using EmptyT = std::tuple<>;

public:
    static constexpr bool implements_storage_v = false;
    static constexpr bool has_secondary_data   = false;
    static constexpr bool is_sampler_v         = trait::sampler<Tp>::value;
    static constexpr bool is_component_type    = false;
    static constexpr bool is_auto_type         = false;
    static constexpr bool is_component         = true;

    using Type             = Tp;
    using value_type       = void;
    using sample_type      = EmptyT;
    using sample_list_type = EmptyT;

    using this_type    = Tp;
    using base_type    = base<Tp, value_type>;
    using storage_type = impl::storage<Tp, implements_storage_v>;

private:
    friend class impl::storage<Tp, implements_storage_v>;

    friend struct operation::init_storage<Tp>;
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

    template <typename _Ret, typename _Lhs, typename _Rhs>
    friend struct operation::compose;

public:
    base()
    : is_running(false)
    , is_on_stack(false)
    , is_transient(false)
    {}

    ~base()                         = default;
    explicit base(const base_type&) = default;
    explicit base(base_type&&)      = default;
    base& operator=(const base_type&) = default;
    base& operator=(base_type&&) = default;

public:
    static void global_init(storage_type*) {}
    static void thread_init(storage_type*) {}
    static void global_finalize(storage_type*) {}
    static void thread_finalize(storage_type*) {}
    template <typename Archive>
    static void extra_serialization(Archive&, const unsigned int)
    {}

public:
    template <typename... _Args>
    static void configure(_Args&&...)
    {}

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
        // is_running   = false;
        is_transient = false;
    }

    //----------------------------------------------------------------------------------//
    // perform a sample
    //
    void sample() {}

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        if(!is_running)
        {
            set_started();
            static_cast<Type&>(*this).start();
        }
    }

    //----------------------------------------------------------------------------------//
    // stop
    //
    void stop()
    {
        if(is_running)
        {
            set_stopped();
            static_cast<Type&>(*this).stop();
        }
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

    friend std::ostream& operator<<(std::ostream& os, const base_type&) { return os; }

    int64_t nlaps() const { return 0; }
    int64_t get_laps() const { return 0; }

    void get() {}

    //----------------------------------------------------------------------------------//
    // assign the type to a pointer
    //
    void get(void*& ptr, size_t typeid_hash) const
    {
        static size_t this_typeid_hash = std::hash<std::string>()(demangle<Type>());
        if(!ptr && typeid_hash == this_typeid_hash)
            ptr = reinterpret_cast<void*>(const_cast<base_type*>(this));
    }

private:
    //----------------------------------------------------------------------------------//
    // insert the node into the graph
    //
    template <typename Scope = scope::tree, typename... _Args>
    void insert_node(Scope&&, _Args&&...)
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

    static void cleanup() {}

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
    static std::string label()
    {
        std::string _label = demangle<Type>();
        if(settings::debug())
            fprintf(stderr, "Warning! '%s' does not provide a custom label!\n",
                    _label.c_str());
        if(_label.find(':') != std::string::npos)
            _label = _label.substr(_label.find_last_of(':'));
        if(_label.find('<') != std::string::npos)
            _label = _label.substr(0, _label.find_first_of('<'));
        while(_label.find(' ') != std::string::npos)
            _label = _label.replace(_label.find(' '), 1, "_");
        while(_label.find("__") != std::string::npos)
            _label = _label.replace(_label.find("__"), 2, "_");
        return _label;
    }

    static std::string description()
    {
        std::string _label = demangle<Type>();
        if(settings::debug())
            fprintf(stderr, "Warning! '%s' does not provide a custom description!\n",
                    _label.c_str());
        return _label;
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

//----------------------------------------------------------------------------------//

}  // namespace component

//----------------------------------------------------------------------------------//
//
namespace variadic
{
//
template <typename... Types>
struct config
: component::base<config<Types...>, void>
, type_list<Types...>
{
    using type = type_list<Types...>;
    void start() {}
    void stop() {}
};
//
//----------------------------------------------------------------------------------//
//
template <typename T>
struct is_config : false_type
{};
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
struct is_config<config<Types...>> : true_type
{};
//
//----------------------------------------------------------------------------------//
//
}  // namespace variadic
//
//----------------------------------------------------------------------------------//
//
}  // namespace tim

//======================================================================================//
