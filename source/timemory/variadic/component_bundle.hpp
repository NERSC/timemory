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

/** \file timemory/variadic/component_bundle.hpp
 * \headerfile variadic/component_bundle.hpp "timemory/variadic/component_bundle.hpp"
 *
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/general/source_location.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/base_bundle.hpp"
#include "timemory/variadic/functional.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

//======================================================================================//
/// \class tim::component_bundle
/// \tparam Tag unique identifying type for the bundle which when \ref
/// tim::trait::is_available<Tag> is false at compile-time or \ref
/// tim::trait::runtime_enabled<Tag>() is false at runtime, then none of the components
/// will be collected
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper which combines the features of \ref
/// tim::component_tuple<T...> and \ref tim::component_list<U..>. The "T" types
/// (compile-time fixed, allocated on stack) should be specified as usual, the "U" types
/// (runtime-time optional, allocated on the heap) should be specified as a pointer.
/// Initialization of the optional types is similar to \ref tim::auto_list<U...> but no
/// environment variable is built-in since, ideally, this environment variable should be
/// customized based on the Tag template parameter.
///
/// See also: \ref tim::auto_bundle.
/// The primary difference b/t the "component_*" and "auto_*" is that the latter
/// used the constructor/destructor to call start and stop and is thus easier to
/// just copy-and-paste into different places. However, the former is better suited for
/// special configuration, data-access, etc.
///
namespace tim
{
//
template <typename Tag, typename... Types>
class component_bundle<Tag, Types...>
: public api_bundle<Tag, implemented_t<Types...>>
, public concepts::tagged
, public concepts::comp_wrapper
, public concepts::mixed_wrapper
{
    static_assert(concepts::is_api<Tag>::value,
                  "Error! The first template parameter of a 'component_bundle' must "
                  "statisfy the 'is_api' concept");

protected:
    using apply_v     = apply<void>;
    using bundle_type = api_bundle<Tag, implemented_t<Types...>>;
    using impl_type   = typename bundle_type::impl_type;

    template <typename T, typename... U>
    friend class base_bundle;

    template <typename... Tp>
    friend class auto_bundle;

public:
    using captured_location_t = source_location::captured;

    using this_type      = component_bundle<Tag, Types...>;
    using type_list_type = type_list<Types...>;

    using data_type         = typename bundle_type::data_type;
    using tuple_type        = typename bundle_type::tuple_type;
    using sample_type       = typename bundle_type::sample_type;
    using reference_type    = typename bundle_type::reference_type;
    using user_bundle_types = typename bundle_type::user_bundle_types;
    using value_type        = data_type;

    using size_type = typename bundle_type::size_type;
    using string_t  = typename bundle_type::string_t;

    template <template <typename> class Op, typename Tuple = data_type>
    using operation_t = typename bundle_type::template generic_operation<Op, Tuple>::type;

    template <template <typename> class Op, typename Tuple = data_type>
    using custom_operation_t =
        typename bundle_type::template custom_operation<Op, Tuple>::type;

    // used by gotcha
    using component_type   = component_bundle<Tag, Types...>;
    using auto_type        = auto_bundle<Tag, Types...>;
    using type             = convert_t<tuple_type, component_bundle<Tag>>;
    using initializer_type = std::function<void(this_type&)>;

    static constexpr bool has_gotcha_v      = bundle_type::has_gotcha_v;
    static constexpr bool has_user_bundle_v = bundle_type::has_user_bundle_v;

public:
    static initializer_type& get_initializer();

    template <typename T, typename... U>
    using quirk_config = mpl::impl::quirk_config<T, type_list<Types...>, U...>;

public:
    component_bundle();

    template <typename... T, typename Func = initializer_type>
    explicit component_bundle(const string_t& _key, quirk::config<T...>,
                              const Func& = get_initializer());

    template <typename... T, typename Func = initializer_type>
    explicit component_bundle(const captured_location_t& _loc, quirk::config<T...>,
                              const Func& = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(size_t _hash, bool _store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&          = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(const string_t& _key, bool _store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&          = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(const captured_location_t& _loc, bool _store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&          = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(size_t _hash, scope::config _scope,
                              const Func& = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(const string_t& _key, scope::config _scope,
                              const Func& = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(const captured_location_t& _loc, scope::config _scope,
                              const Func& = get_initializer());

    ~component_bundle();

    //------------------------------------------------------------------------//
    //      Copy/move construct and assignment
    //------------------------------------------------------------------------//
    component_bundle(const component_bundle& rhs);
    component_bundle(component_bundle&&) noexcept = default;

    component_bundle& operator=(const component_bundle& rhs);
    component_bundle& operator=(component_bundle&&) noexcept = default;

    component_bundle clone(bool store, scope::config _scope = scope::get_default());

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

    /// variant of start() which excludes push()
    template <typename... Args>
    void start(mpl::lightweight, Args&&...);
    /// variant of stop() which excludes pop()
    template <typename... Args>
    void stop(mpl::lightweight, Args&&...);

    /// variant of start() which only gets applied to Tp types
    template <typename... Tp, typename... Args>
    void start(mpl::piecewise_select<Tp...>, Args&&...);
    /// variant of stop() which only gets applied to Tp types
    template <typename... Tp, typename... Args>
    void stop(mpl::piecewise_select<Tp...>, Args&&...);

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
    /// query the number of (compile-time) fixed components
    //
    static constexpr uint64_t fixed_count()
    {
        return (size() -
                mpl::get_tuple_size<
                    typename get_true_types<std::is_pointer, data_type>::type>::value);
    }

    //----------------------------------------------------------------------------------//
    /// query the number of (run-time) optional components
    //
    static constexpr uint64_t optional_count()
    {
        return mpl::get_tuple_size<
            typename get_true_types<std::is_pointer, data_type>::type>::value;
    }

    //----------------------------------------------------------------------------------//
    /// number of objects that will be performing measurements
    //
    uint64_t count()
    {
        uint64_t _count = 0;
        invoke::invoke<operation::generic_counter>(m_data, std::ref(_count));
        return _count;
    }

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
    /// provide preliminary info to the objects with matching arguments. This is typically
    /// used to notify a component that it has been bundled alongside another component
    /// that it can extract data from.
    //
    template <typename... Args>
    void assemble(Args&&... _args)
    {
        invoke::assemble(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// provide conclusive info to the objects with matching arguments. This is typically
    /// used by components to extract data from another component it has been bundled
    /// alongside, e.g. the cpu_util component can extract data from \ref
    /// tim::component::wall_clock and \ref tim::component::cpu_clock
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
    /// allow the components to inspect the incoming arguments before start
    /// or out-going return value before returning (typically using in GOTCHA components)
    //
    template <typename... Args>
    void audit(Args&&... _args)
    {
        invoke::audit(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// perform an add_secondary operation. This operation allows components to add
    /// additional entries to storage which are their direct descendant
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
        invoke::invoke<OpT, Tag>(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// generic member function for invoking user-provided operations on a specific
    /// set of component types
    /// \tparam OpT Operation struct
    //
    template <template <typename> class OpT, typename... Tp, typename... Args>
    void invoke(mpl::piecewise_select<Tp...>, Args&&... _args)
    {
        TIMEMORY_FOLD_EXPRESSION(operation::generic_operator<Tp, OpT<Tp>, Tag>(
            this->get<Tp>(), std::forward<Args>(_args)...));
    }

    //----------------------------------------------------------------------------------//
    // get member functions taking either a type
    //
    //
    //----------------------------------------------------------------------------------//
    //  exact type available
    //
    template <typename U, typename T = decay_t<U>,
              enable_if_t<is_one_of<T, data_type>::value, int> = 0>
    T* get()
    {
        return &(std::get<index_of<T, data_type>::value>(m_data));
    }

    template <typename U, typename T = decay_t<U>,
              enable_if_t<is_one_of<T, data_type>::value, int> = 0>
    const T* get() const
    {
        return &(std::get<index_of<T, data_type>::value>(m_data));
    }
    //
    //----------------------------------------------------------------------------------//
    //  type available with add_pointer
    //
    template <typename U, typename T = decay_t<U>,
              enable_if_t<is_one_of<T*, data_type>::value, int> = 0>
    T* get()
    {
        return std::get<index_of<T*, data_type>::value>(m_data);
    }

    template <typename U, typename T = decay_t<U>,
              enable_if_t<is_one_of<T*, data_type>::value, int> = 0>
    const T* get() const
    {
        return std::get<index_of<T*, data_type>::value>(m_data);
    }
    //
    //----------------------------------------------------------------------------------//
    //  type available with remove_pointer
    //
    template <
        typename U, typename T = decay_t<U>, typename R = remove_pointer_t<T>,
        enable_if_t<!is_one_of<T, data_type>::value && !is_one_of<T*, data_type>::value &&
                        is_one_of<R, data_type>::value,
                    int> = 0>
    T* get()
    {
        return &std::get<index_of<R, data_type>::value>(m_data);
    }

    template <
        typename U, typename T = decay_t<U>, typename R = remove_pointer_t<T>,
        enable_if_t<!is_one_of<T, data_type>::value && !is_one_of<T*, data_type>::value &&
                        is_one_of<R, data_type>::value,
                    int> = 0>
    const T* get() const
    {
        return &std::get<index_of<R, data_type>::value>(m_data);
    }
    //
    //----------------------------------------------------------------------------------//
    ///  type is not explicitly listed so redirect to opaque search
    ///
    template <
        typename U, typename T = decay_t<U>, typename R = remove_pointer_t<T>,
        enable_if_t<!is_one_of<T, data_type>::value && !is_one_of<T*, data_type>::value &&
                        !is_one_of<R, data_type>::value,
                    int> = 0>
    T* get() const
    {
        void* ptr = nullptr;
        get(ptr, typeid_hash<T>());
        return static_cast<T*>(ptr);
    }

    /// performs an opaque search. Opaque searches are generally provided by user_bundles
    /// with a functor such as this:
    ///
    /// \code{.cpp}
    /// auto _get = [=](void* v_this, void*& ptr, size_t _hash) {
    /// {
    ///     if(!ptr && v_this && _hash == typeid_hash<Tp>())
    ///     {
    ///         Tp* _this = static_cast<Tp*>(v_this);
    ///         _this->get(ptr, _hash);
    ///     }
    ///     return ptr;
    /// };
    /// \endcode
    ///
    /// And the component provides this function:
    ///
    /// \code{.cpp}
    /// template <typename Tp, typename Value>
    /// void
    /// base<Tp, Value>::get(void*& ptr, size_t _hash) const
    /// {
    ///     if(!ptr && _hash == typeid_hash<Tp>())
    ///         ptr = reinterpret_cast<void*>(const_cast<base_type*>(this));
    /// }
    /// \endcode
    ///
    void get(void*& ptr, size_t _hash) const
    {
        using get_t = operation_t<operation::get>;
        apply_v::access<get_t>(m_data, ptr, _hash);
    }

    //----------------------------------------------------------------------------------//
    /// this is a simple alternative to get<T>() when used from SFINAE in operation
    /// namespace which has a struct get also templated. Usage there can cause error
    /// with older compilers
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              enable_if_t<trait::is_available<T>::value && is_one_of<T, data_type>::value,
                          int> = 0>
    auto get_component()
    {
        return get<T>();
    }

    template <
        typename U, typename T = std::remove_pointer_t<decay_t<U>>,
        enable_if_t<trait::is_available<T>::value && is_one_of<T*, data_type>::value,
                    int> = 0>
    auto get_component()
    {
        return get<T>();
    }

    /// returns a reference from a stack component instead of a pointer
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              enable_if_t<trait::is_available<T>::value && is_one_of<T, data_type>::value,
                          int> = 0>
    auto& get_reference()
    {
        return std::get<index_of<T, data_type>::value>(m_data);
    }

    /// returns a reference from a heap component instead of a pointer
    template <
        typename U, typename T = std::remove_pointer_t<decay_t<U>>,
        enable_if_t<trait::is_available<T>::value && is_one_of<T*, data_type>::value,
                    int> = 0>
    auto& get_reference()
    {
        return std::get<index_of<T*, data_type>::value>(m_data);
    }

    //----------------------------------------------------------------------------------//
    /// create an optional type that is in variadic list AND is available AND
    /// accepts arguments
    ///
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args>
    enable_if_t<trait::is_available<T>::value && is_one_of<T*, data_type>::value &&
                    !is_one_of<T, data_type>::value &&
                    std::is_constructible<T, Args...>::value,
                bool>
    init(Args&&... _args)
    {
        T*& _obj = std::get<index_of<T*, data_type>::value>(m_data);
        if(!_obj)
        {
            if(settings::debug())
            {
                printf("[component_bundle::init]> initializing type '%s'...\n",
                       demangle<T>().c_str());
            }
            _obj = new T(std::forward<Args>(_args)...);
            set_prefix(_obj);
            set_scope(_obj);
            return true;
        }

        static std::atomic<int> _count(0);
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            std::string _id = demangle<T>();
            fprintf(stderr,
                    "[component_bundle::init]> skipping re-initialization of type"
                    " \"%s\"...\n",
                    _id.c_str());
        }

        return false;
    }

    //----------------------------------------------------------------------------------//
    /// create an optional type that is in variadic list AND is available but is not
    /// constructible with provided arguments
    ///
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args>
    enable_if_t<trait::is_available<T>::value && is_one_of<T*, data_type>::value &&
                    !is_one_of<T, data_type>::value &&
                    !std::is_constructible<T, Args...>::value &&
                    std::is_default_constructible<T>::value,
                bool>
    init(Args&&...)
    {
        T*& _obj = std::get<index_of<T*, data_type>::value>(m_data);
        if(!_obj)
        {
            if(settings::debug())
            {
                fprintf(stderr, "[component_bundle::init]> initializing type '%s'...\n",
                        demangle<T>().c_str());
            }
            _obj = new T{};
            set_prefix(_obj);
            set_scope(_obj);
            return true;
        }

        static std::atomic<int> _count(0);
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            fprintf(stderr,
                    "[component_bundle::init]> skipping re-initialization of type "
                    "\"%s\"...\n",
                    demangle<T>().c_str());
        }

        return false;
    }

    //----------------------------------------------------------------------------------//
    /// try to re-create a stack object with provided arguments
    ///
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args>
    enable_if_t<trait::is_available<T>::value && !is_one_of<T*, data_type>::value &&
                    is_one_of<T, data_type>::value,
                bool>
    init(Args&&... _args)
    {
        T& _obj = std::get<index_of<T, data_type>::value>(m_data);
        operation::construct<T>{ _obj, std::forward<Args>(_args)... };
        set_prefix(&_obj);
        set_scope(&_obj);
        return true;
    }

    //----------------------------------------------------------------------------------//
    /// try to re-create a stack object with provided arguments (ignore heap type)
    ///
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args>
    enable_if_t<trait::is_available<T>::value && is_one_of<T*, data_type>::value &&
                    is_one_of<T, data_type>::value,
                bool>
    init(Args&&... _args)
    {
        static std::atomic<int> _count(0);
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            fprintf(stderr,
                    "[component_bundle::init]> type exists as a heap-allocated instance "
                    "and stack-allocated instance: \"%s\"...\n",
                    demangle<T>().c_str());
        }
        T& _obj = std::get<index_of<T, data_type>::value>(m_data);
        operation::construct<T>{ _obj, std::forward<Args>(_args)... };
        set_prefix(&_obj);
        set_scope(&_obj);
        return true;
    }

    //----------------------------------------------------------------------------------//
    /// if a type is not in variadic list but a \ref tim::component::user_bundle is
    /// available, add it in there
    ///
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args>
    enable_if_t<trait::is_available<T>::value && !is_one_of<T, data_type>::value &&
                    !is_one_of<T*, data_type>::value && has_user_bundle_v,
                bool>
    init(Args&&...)
    {
        using bundle_t =
            decay_t<decltype(std::get<0>(std::declval<user_bundle_types>()))>;
        static_assert(trait::is_user_bundle<bundle_t>::value, "Error! Not a user_bundle");
        this->init<bundle_t>();
        auto* _bundle = this->get<bundle_t>();
        if(_bundle)
        {
            _bundle->insert(component::factory::get_opaque<T>(m_scope),
                            component::factory::get_typeids<T>());
            return true;
        }
        return false;
    }

    //----------------------------------------------------------------------------------//
    /// do nothing if type not available, not one of the variadic types, and there
    /// is no user bundle available
    ///
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args>
    enable_if_t<!trait::is_available<T>::value ||
                    !(is_one_of<T*, data_type>::value || is_one_of<T, data_type>::value ||
                      has_user_bundle_v),
                bool>
    init(Args&&...)
    {
        return false;
    }

    //----------------------------------------------------------------------------------//
    /// \brief variadic initialization
    /// \tparam T components to initialize
    /// \tparam Args arguments to pass to the construction of the component
    //
    template <typename... T, typename... Args>
    auto initialize(Args&&... args)
    {
        constexpr auto N = sizeof...(T);
        return TIMEMORY_FOLD_EXPANSION(bool, N, init<T>(std::forward<Args>(args)...));
    }

    /// delete any optional types currently allocated
    template <typename... Tail>
    void disable()
    {
        TIMEMORY_FOLD_EXPRESSION(operation::generic_deleter<remove_pointer_t<Tail>>{
            this->get_reference<Tail>() });
    }

    //----------------------------------------------------------------------------------//
    /// apply a member function to a stack type that is in variadic list AND is available
    ///
    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T, data_type>::value, int> = 0>
    void type_apply(Func&& _func, Args&&... _args)
    {
        auto* _obj = get<T>();
        ((*_obj).*(_func))(std::forward<Args>(_args)...);
    }

    /// apply a member function to either a heap type or a type that is in a user_bundle
    template <typename T, typename Func, typename... Args,
              enable_if_t<trait::is_available<T>::value, int> = 0>
    void type_apply(Func&& _func, Args&&... _args)
    {
        auto* _obj = get<T>();
        if(_obj)
            ((*_obj).*(_func))(std::forward<Args>(_args)...);
    }

    /// ignore applying a member function because the type is not present
    template <typename T, typename Func, typename... Args,
              enable_if_t<!trait::is_available<T>::value, int> = 0>
    void type_apply(Func&&, Args&&...)
    {}

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
        using printer_t = typename bundle_type::print_type;
        if(size() == 0 || m_hash == 0)
            return;
        std::stringstream ss_data;
        apply_v::access_with_indices<printer_t>(m_data, std::ref(ss_data), false);
        if(PrintPrefix)
        {
            update_width();
            std::stringstream ss_prefix;
            std::stringstream ss_id;
            ss_id << get_prefix() << " " << std::left << key();
            ss_prefix << std::setw(output_width()) << std::left << ss_id.str() << " : ";
            os << ss_prefix.str();
        }
        os << ss_data.str();
        if(m_laps > 0 && PrintLaps)
            os << " [laps: " << m_laps << "]";
    }

    //----------------------------------------------------------------------------------//
    //
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        obj.print<true, true>(os);
        return os;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        std::string _key   = {};
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

protected:
    static int64_t output_width(int64_t w = 0) { return bundle_type::output_width(w); }
    void           update_width() const { bundle_type::update_width(); }
    void compute_width(const string_t& _key) const { bundle_type::compute_width(_key); }

protected:
    // protected member functions
    data_type&       get_data();
    const data_type& get_data() const;

    template <typename T>
    void set_scope(T* obj) const;
    void set_scope(scope::config);

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
    mutable data_type m_data = data_type{};
};
//
//======================================================================================//
//
template <typename... Types>
auto
get(const component_bundle<Types...>& _obj)
    -> decltype(std::declval<component_bundle<Types...>>().get())
{
    return _obj.get();
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get_labeled(const component_bundle<Types...>& _obj)
    -> decltype(std::declval<component_bundle<Types...>>().get_labeled())
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

template <std::size_t N, typename Tag, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(::tim::component_bundle<Tag, Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename Tag, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const ::tim::component_bundle<Tag, Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename Tag, typename... Types>
auto
get(::tim::component_bundle<Tag, Types...>&& obj)
    -> decltype(get<N>(std::forward<::tim::component_bundle<Tag, Types...>>(obj).data()))
{
    using obj_type = ::tim::component_bundle<Tag, Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std

//--------------------------------------------------------------------------------------//
