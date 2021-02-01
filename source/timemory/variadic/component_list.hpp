
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/general/source_location.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/base_bundle.hpp"
#include "timemory/variadic/functional.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

//======================================================================================//

namespace tim
{
//======================================================================================//
// variadic list of components
//
/// \class tim::component_list
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper where all components are optional
/// at runtime. Accept unlimited number of parameters. The default behavior is
/// to query the TIMEMORY_COMPONENT_LIST_INIT environment variable once (the first
/// time the bundle is used) and use that list of components (if any) to
/// initialize the components which are part of it's template parameters.
/// This behavior can be modified by assigning a new lambda/functor to the
/// reference which is returned from \ref
/// tim::component_list<Types...>::get_initializer(). Assignment is not thread-safe since
/// this is relatively unnecessary... if a different set of components are required on a
/// particular thread, just create a different type with those particular components or
/// pass the initialization functor to the constructor.
///
/// \code{.cpp}
/// using bundle_t = tim::component_list<wall_clock, cpu_clock, peak_rss>;
///
/// void foo()
/// {
///     setenv("TIMEMORY_COMPONENT_LIST_INIT", "wall_clock", 0);
///
///     auto bar = bundle_t("bar");
///
///     bundle_t::get_initializer() = [](bundle_t& b)
///     {
///         b.initialize<cpu_clock, peak_rss>();
///     };
///
///     auto qix = bundle_t("qix");
///
///     auto local_init = [](bundle_t& b)
///     {
///         b.initialize<thread_cpu_clock, peak_rss>();
///     };
///
///     auto spam = bundle_t("spam", ..., local_init);
///
/// }
/// \endcode
///
/// The above code will record wall-clock timer on first use of "bar", and
/// will record cpu-clock, peak-rss at "qix", and peak-rss at "spam". If foo()
/// is called a second time, "bar" will record cpu-clock and peak-rss. "spam" will
/// always use the local initialized. If none of these initializers are set, wall-clock
/// will be recorded for all of them. The intermediate storage will happen on the heap and
/// when the destructor is called, it will add itself to the call-graph
template <typename... Types>
class component_list
: public heap_bundle<available_t<concat<Types...>>>
, public concepts::comp_wrapper
{
    using apply_v     = apply<void>;
    using bundle_type = heap_bundle<available_t<concat<Types...>>>;
    using impl_type   = typename bundle_type::impl_type;

    template <typename T, typename... U>
    friend class base_bundle;

    template <typename... Tp>
    friend class auto_list;

#if defined(TIMEMORY_USE_DEPRECATED)
    template <typename TupleC, typename ListC>
    friend class component_hybrid;
#endif

public:
    using captured_location_t = source_location::captured;

    using this_type      = component_list<Types...>;
    using type_list_type = type_list<Types...>;

    using data_type         = typename bundle_type::data_type;
    using tuple_type        = typename bundle_type::tuple_type;
    using sample_type       = typename bundle_type::sample_type;
    using reference_type    = typename bundle_type::reference_type;
    using user_bundle_types = typename bundle_type::user_bundle_types;

    using size_type = typename bundle_type::size_type;
    using string_t  = typename bundle_type::string_t;

    template <template <typename> class Op, typename Tuple = impl_type>
    using operation_t = typename bundle_type::template generic_operation<Op, Tuple>::type;

    template <template <typename> class Op, typename Tuple = impl_type>
    using custom_operation_t =
        typename bundle_type::template custom_operation<Op, Tuple>::type;

    // used by gotcha
    using component_type   = component_list<Types...>;
    using auto_type        = auto_list<Types...>;
    using type             = convert_t<tuple_type, component_list<>>;
    using initializer_type = std::function<void(this_type&)>;

    static constexpr bool has_gotcha_v      = bundle_type::has_gotcha_v;
    static constexpr bool has_user_bundle_v = bundle_type::has_user_bundle_v;

public:
    static initializer_type& get_initializer();

    template <typename T, typename... U>
    using quirk_config = mpl::impl::quirk_config<T, type_list<Types...>, U...>;

public:
    component_list();

    template <typename Func = initializer_type>
    explicit component_list(const string_t& _key, const bool& _store = true,
                            scope::config _scope = scope::get_default(),
                            const Func&          = get_initializer());

    template <typename Func = initializer_type>
    explicit component_list(const captured_location_t& _loc, const bool& _store = true,
                            scope::config _scope = scope::get_default(),
                            const Func&          = get_initializer());

    template <typename Func = initializer_type>
    explicit component_list(size_t _hash, const bool& _store = true,
                            scope::config _scope = scope::get_default(),
                            const Func&          = get_initializer());

    ~component_list();

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_list(component_list&&) noexcept = default;
    component_list& operator=(component_list&&) noexcept = default;

    component_list(const component_list& rhs);
    component_list& operator=(const component_list& rhs);

    component_list clone(bool store, scope::config _scope = scope::get_default());

public:
    //----------------------------------------------------------------------------------//
    // public static functions
    //
    /// requests the components to output their storage
    static void print_storage();
    /// requests the component initialize their storage
    static void init_storage();

    //----------------------------------------------------------------------------------//
    // public member functions
    //
    /// tells each component to push itself into the call-stack hierarchy
    void push();
    /// tells each component to pop itself off of the call-stack hierarchy
    void pop();
    /// requests each component record a measurment
    template <typename... Args>
    void measure(Args&&...);
    /// requests each component take a sample (if supported)
    template <typename... Args>
    void sample(Args&&...);
    /// invokes start on all the components
    template <typename... Args>
    void start(Args&&...);
    /// invokes stop on all the components
    template <typename... Args>
    void stop(Args&&...);
    /// requests each component performs a measurement
    template <typename... Args>
    this_type& record(Args&&...);
    /// invokes reset member function on all the components
    template <typename... Args>
    void reset(Args&&...);
    /// returns a tuple of invoking get() on all the components
    template <typename... Args>
    auto get(Args&&...) const;
    /// returns a tuple of the component label + invoking get() on all the components
    template <typename... Args>
    auto get_labeled(Args&&...) const;
    /// returns a reference to the underlying tuple of components
    data_type& data();
    /// returns a const reference to the underlying tuple of components
    const data_type& data() const;

    using bundle_type::get_prefix;
    using bundle_type::get_scope;
    using bundle_type::get_store;
    using bundle_type::hash;
    using bundle_type::key;
    using bundle_type::laps;
    using bundle_type::prefix;
    using bundle_type::rekey;
    using bundle_type::size;
    using bundle_type::store;

    //----------------------------------------------------------------------------------//
    /// construct the objects that have constructors with matching arguments
    //
    template <typename... Args>
    void construct(Args&&... _args)
    {
        using construct_t = operation_t<operation::construct>;
        apply_v::access<construct_t>(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// provide preliminary info to the objects with matching arguments
    //
    template <typename... Args>
    void assemble(Args&&... _args)
    {
        invoke::assemble(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// provide conclusive info to the objects with matching arguments
    //
    template <typename... Args>
    void derive(Args&&... _args)
    {
        invoke::derive(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// mark an atomic event
    //
    template <typename... Args>
    void mark(Args&&... _args)
    {
        invoke::mark(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// mark a beginning position in the execution (typically used by asynchronous
    /// structures)
    //
    template <typename... Args>
    void mark_begin(Args&&... _args)
    {
        invoke::mark_begin(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// mark a beginning position in the execution (typically used by asynchronous
    /// structures)
    //
    template <typename... Args>
    void mark_end(Args&&... _args)
    {
        invoke::mark_end(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// store a value
    //
    template <typename... Args>
    void store(Args&&... _args)
    {
        invoke::store(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// perform a audit operation (typically for GOTCHA)
    //
    template <typename... Args>
    void audit(Args&&... _args)
    {
        invoke::audit(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// perform an add_secondary operation
    //
    template <typename... Args>
    void add_secondary(Args&&... _args)
    {
        invoke::add_secondary(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// generic member function for invoking user-provided operations
    /// \tparam OpT Operation struct
    //
    template <template <typename> class OpT, typename... Args>
    void invoke(Args&&... _args)
    {
        invoke::invoke<OpT>(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// generic member function for invoking user-provided operations on a specific
    /// set of component types
    /// \tparam OpT Operation struct
    //
    template <template <typename> class OpT, typename... Tp, typename... Args>
    void invoke(mpl::piecewise_select<Tp...>, Args&&... _args)
    {
        TIMEMORY_FOLD_EXPRESSION(operation::generic_operator<Tp, OpT<Tp>, TIMEMORY_API>(
            this->get<Tp>(), std::forward<Args>(_args)...));
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs);
    this_type& operator-=(this_type& rhs);
    this_type& operator+=(const this_type& rhs);
    this_type& operator+=(this_type& rhs);

    //----------------------------------------------------------------------------------//
    // generic operators
    //
    template <typename Op>
    this_type& operator-=(Op&& rhs)
    {
        using minus_t = operation_t<operation::minus>;
        apply_v::access<minus_t>(m_data, std::forward<Op>(rhs));
        return *this;
    }

    template <typename Op>
    this_type& operator+=(Op&& rhs)
    {
        using plus_t = operation_t<operation::plus>;
        apply_v::access<plus_t>(m_data, std::forward<Op>(rhs));
        return *this;
    }

    template <typename Op>
    this_type& operator*=(Op&& rhs)
    {
        using multiply_t = operation_t<operation::multiply>;
        apply_v::access<multiply_t>(m_data, std::forward<Op>(rhs));
        return *this;
    }

    template <typename Op>
    this_type& operator/=(Op&& rhs)
    {
        using divide_t = operation_t<operation::divide>;
        apply_v::access<divide_t>(m_data, std::forward<Op>(rhs));
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // friend operators
    //
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        return tmp += rhs;
    }

    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        return tmp -= rhs;
    }

    template <typename Op>
    friend this_type operator*(const this_type& lhs, Op&& rhs)
    {
        this_type tmp(lhs);
        return tmp *= std::forward<Op>(rhs);
    }

    template <typename Op>
    friend this_type operator/(const this_type& lhs, Op&& rhs)
    {
        this_type tmp(lhs);
        return tmp /= std::forward<Op>(rhs);
    }

    //----------------------------------------------------------------------------------//
    //
    template <bool PrintPrefix = true, bool PrintLaps = true>
    void print(std::ostream& os) const
    {
        using printer_t       = typename bundle_type::print_type;
        using pointer_count_t = operation_t<operation::generic_counter>;

        uint64_t count = 0;
        apply_v::access<pointer_count_t>(m_data, std::ref(count));
        if(count < 1 || m_hash == 0)
            return;
        std::stringstream ss_data;
        apply_v::access_with_indices<printer_t>(m_data, std::ref(ss_data), false);
        if(ss_data.str().length() > 0)
        {
            if(PrintPrefix)
            {
                update_width();
                std::stringstream ss_prefix;
                std::stringstream ss_id;
                ss_id << get_prefix() << " " << std::left << key();
                ss_prefix << std::setw(output_width()) << std::left << ss_id.str()
                          << " : ";
                os << ss_prefix.str();
            }
            os << ss_data.str();
            if(laps() > 0 && PrintLaps)
                os << " [laps: " << m_laps << "]";
        }
    }

    //----------------------------------------------------------------------------------//
    //
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        obj.print<true, true>(os);
        return os;
    }

    //----------------------------------------------------------------------------------//
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        std::string _key;
        auto        keyitr = get_hash_ids()->find(m_hash);
        if(keyitr != get_hash_ids()->end())
            _key = keyitr->second;

        ar(cereal::make_nvp("hash", m_hash), cereal::make_nvp("key", _key),
           cereal::make_nvp("laps", m_laps));

        if(keyitr == get_hash_ids()->end())
        {
            auto _hash = add_hash_id(_key);
            if(_hash != m_hash)
            {
                PRINT_HERE("Warning! Hash for '%s' (%llu) != %llu", _key.c_str(),
                           (unsigned long long) _hash, (unsigned long long) m_hash);
            }
        }

        ar(cereal::make_nvp("data", m_data));
    }

public:
    // get member functions taking a type
    /// return pointer to component instance
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              enable_if_t<is_one_of<T*, data_type>::value, int> = 0>
    T* get()
    {
        return std::get<index_of<T*, data_type>::value>(m_data);
    }

    /// return pointer to component instance
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              enable_if_t<is_one_of<T*, data_type>::value, int> = 0>
    const T* get() const
    {
        return std::get<index_of<T*, data_type>::value>(m_data);
    }

    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              enable_if_t<!is_one_of<T*, data_type>::value, int> = 0>
    T* get()
    {
        return nullptr;
    }

    /// return pointer to component instance
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              enable_if_t<!is_one_of<T*, data_type>::value, int> = 0>
    const T* get() const
    {
        return nullptr;
    }

    /// generic get routine for opaque types
    void get(void*& ptr, size_t _hash)
    {
        using get_t = operation_t<operation::get>;
        apply_v::access<get_t>(m_data, ptr, _hash);
    }

    //----------------------------------------------------------------------------------//
    /// this is a simple alternative to get<T>() when used from SFINAE in operation
    /// namespace which has a struct get also templated. Usage there can cause error
    /// with older compilers
    template <
        typename U, typename T = std::remove_pointer_t<decay_t<U>>,
        enable_if_t<trait::is_available<T>::value && is_one_of<T*, data_type>::value,
                    int> = 0>
    auto get_component()
    {
        return get<T>();
    }

    /// get a reference to the underlying pointer to component instance
    template <
        typename U, typename T = std::remove_pointer_t<decay_t<U>>,
        enable_if_t<trait::is_available<T>::value && is_one_of<T*, data_type>::value,
                    int> = 0>
    auto& get_reference()
    {
        return std::get<index_of<T*, data_type>::value>(m_data);
    }

    //----------------------------------------------------------------------------------//
    ///  initialize a type that is in variadic list AND is available
    ///
    template <
        typename U, typename T = std::remove_pointer_t<decay_t<U>>, typename... Args,
        enable_if_t<is_one_of<T*, data_type>::value && trait::is_available<T>::value,
                    char> = 0>
    void init(Args&&... _args)
    {
        T*& _obj = std::get<index_of<T*, data_type>::value>(m_data);
        if(!_obj)
        {
            if(settings::debug())
            {
                printf("[component_list::init]> initializing type '%s'...\n",
                       demangle(typeid(T).name()).c_str());
            }
            _obj = new T(std::forward<Args>(_args)...);
            set_prefix(_obj);
        }
        else
        {
            static std::atomic<int> _count(0);
            if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
            {
                std::string _id = demangle(typeid(T).name());
                printf("[component_list::init]> skipping re-initialization of type"
                       " \"%s\"...\n",
                       _id.c_str());
            }
        }
    }

    //----------------------------------------------------------------------------------//
    ///  "initialize" a type that is NOT in variadic list but IS available
    ///
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args,
              enable_if_t<!is_one_of<T*, data_type>::value &&
                              trait::is_available<T>::value && has_user_bundle_v,
                          int> = 0>
    void init(Args&&... args)
    {
        using bundle_t = decltype(std::get<0>(std::declval<user_bundle_types>()));
        this->init<bundle_t>();
        this->get<bundle_t>()->insert(
            component::factory::get_opaque<T>(m_scope, std::forward<Args>(args)...),
            component::factory::get_typeids<T>());
    }

    //----------------------------------------------------------------------------------//
    ///  "initialize" a type that is in variadic list but is NOT available
    ///
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args,
              enable_if_t<!trait::is_available<T>::value ||
                              (!is_one_of<T*, data_type>::value && !has_user_bundle_v),
                          long> = 0>
    void init(Args&&...)
    {
        static std::atomic<int> _count(0);
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            PRINT_HERE("%s %s", "skipping init because type is unavailable or because",
                       "wrapper does not contain a user_bundle");
        }
    }

    /// activate component types
    template <typename T, typename... Tail, typename... Args>
    void initialize(Args&&... args)
    {
        this->init<T>(std::forward<Args>(args)...);
        TIMEMORY_FOLD_EXPRESSION(this->init<Tail>(std::forward<Args>(args)...));
    }

    /// disable a component that was previously initialized
    template <typename... Tail>
    void disable()
    {
        TIMEMORY_FOLD_EXPRESSION(operation::generic_deleter<remove_pointer_t<Tail>>{
            this->get_reference<Tail>() });
    }

    //----------------------------------------------------------------------------------//
    /// apply a member function to a type that is in variadic list AND is available
    ///
    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T, reference_type>::value == true, int> = 0,
              enable_if_t<trait::is_available<T>::value == true, int>       = 0>
    void type_apply(Func&& _func, Args&&... _args)
    {
        auto* _obj = get<T>();
        if(_obj != nullptr)
            ((*_obj).*(_func))(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// "apply" a member function to a type that is in variadic list BUT is NOT available
    ///
    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T, reference_type>::value == true, int> = 0,
              enable_if_t<trait::is_available<T>::value == false, int>      = 0>
    void type_apply(Func&&, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    /// invoked when a request to apply a member function to a type not in variadic list
    ///
    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T, reference_type>::value == false, int> = 0>
    void type_apply(Func&&, Args&&...)
    {}

protected:
    static int64_t output_width(int64_t w = 0) { return bundle_type::output_width(w); }
    void           update_width() const { bundle_type::update_width(); }
    void compute_width(const string_t& _key) const { bundle_type::compute_width(_key); }

protected:
    // protected member functions
    data_type&       get_data();
    const data_type& get_data() const;
    void             set_scope(scope::config);

    template <typename T>
    void set_prefix(T* obj) const;
    void set_prefix(const string_t&) const;
    void set_prefix(size_t) const;

protected:
    // objects
    using bundle_type::m_config;
    using bundle_type::m_hash;
    using bundle_type::m_is_active;
    using bundle_type::m_is_pushed;
    using bundle_type::m_laps;
    using bundle_type::m_scope;
    using bundle_type::m_store;
    mutable data_type m_data = data_type();
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get(const component_list<Types...>& _obj)
    -> decltype(std::declval<component_list<Types...>>().get())
{
    return _obj.get();
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get_labeled(const component_list<Types...>& _obj)
    -> decltype(std::declval<component_list<Types...>>().get_labeled())
{
    return _obj.get_labeled();
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//
//
//      std::get operator
//
namespace std
{
//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(tim::component_list<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const tim::component_list<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(tim::component_list<Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::component_list<Types...>>(obj).data()))
{
    using obj_type = tim::component_list<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std
