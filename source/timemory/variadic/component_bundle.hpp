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
 * This is the C++ class that bundles together components and enables
 * operation on the components as a single entity
 *
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/general/source_location.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
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

namespace tim
{
//
template <typename Tag, typename... Types>
class component_bundle : public api_bundle<Tag, implemented_t<Types...>>
{
    // manager is friend so can use above
    friend class manager;

    template <typename TagT, typename... Tp>
    friend class auto_bundle;

public:
    using bundle_type         = api_bundle<Tag, implemented_t<Types...>>;
    using this_type           = component_bundle<Tag, Types...>;
    using captured_location_t = source_location::captured;

    using data_type         = typename bundle_type::data_type;
    using impl_type         = typename bundle_type::impl_type;
    using tuple_type        = typename bundle_type::tuple_type;
    using sample_type       = typename bundle_type::sample_type;
    using reference_type    = typename bundle_type::reference_type;
    using user_bundle_types = typename bundle_type::user_bundle_types;
    using value_type        = data_type;

    using apply_v     = apply<void>;
    using size_type   = typename bundle_type::size_type;
    using string_t    = typename bundle_type::string_t;
    using string_hash = typename bundle_type::string_hash;

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

    static constexpr bool is_component      = false;
    static constexpr bool has_gotcha_v      = bundle_type::has_gotcha_v;
    static constexpr bool has_user_bundle_v = bundle_type::has_user_bundle_v;

public:
    //
    //----------------------------------------------------------------------------------//
    //
    static initializer_type& get_initializer()
    {
        static initializer_type _instance = [](this_type& cl) {
            static auto env_enum = []() {
                auto _tag = demangle<Tag>();
                for(const auto& itr : { string_t("tim::"), string_t("api::") })
                {
                    auto _pos = _tag.find(itr);
                    do
                    {
                        if(_pos != std::string::npos)
                            _tag = _tag.erase(_pos, itr.length());
                        _pos = _tag.find(itr);
                    } while(_pos != std::string::npos);
                }

                for(const auto& itr : { string_t("::"), string_t("<"), string_t(">"),
                                        string_t(" "), string_t("__") })
                {
                    auto _pos = _tag.find(itr);
                    do
                    {
                        if(_pos != std::string::npos)
                            _tag = _tag.replace(_pos, itr.length(), "_");
                        _pos = _tag.find(itr);
                    } while(_pos != std::string::npos);
                }

                if(_tag.length() > 0 && _tag.at(0) == '_')
                    _tag = _tag.substr(1);
                if(_tag.length() > 0 && _tag.at(_tag.size() - 1) == '_')
                    _tag = _tag.substr(0, _tag.size() - 1);

                for(auto& itr : _tag)
                    itr = toupper(itr);
                auto env_var = string_t("TIMEMORY_") + _tag + "_COMPONENTS";
                if(settings::debug() || settings::verbose() > 0)
                    PRINT_HERE("%s is using environment variable: '%s'",
                               demangle<this_type>().c_str(), env_var.c_str());

                // get environment variable
                return enumerate_components(
                    tim::delimit(tim::get_env<string_t>(env_var, "")));
            }();
            ::tim::initialize(cl, env_enum);
        };
        return _instance;
    }

public:
    template <typename T, typename... U>
    struct quirk_config
    {
        static constexpr bool value =
            is_one_of<T,
                      contains_one_of_t<quirk::is_config, concat<Types..., U...>>>::value;
    };

public:
    component_bundle();

    template <typename... T, typename Func = initializer_type>
    explicit component_bundle(const string_t& key, quirk::config<T...>,
                              const Func& = get_initializer());

    template <typename... T, typename Func = initializer_type>
    explicit component_bundle(const captured_location_t& loc, quirk::config<T...>,
                              const Func& = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(const string_t& key, const bool& store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&          = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(const captured_location_t& loc, const bool& store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&          = get_initializer());

    template <typename Func = initializer_type>
    explicit component_bundle(size_t _hash, const bool& store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&          = get_initializer());

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
    static constexpr std::size_t size() { return std::tuple_size<data_type>::value; }
    static void                  print_storage();
    static void                  init_storage();

    //----------------------------------------------------------------------------------//
    // public member functions
    //
    void push();
    void pop();
    template <typename... Args>
    void measure(Args&&...);
    template <typename... Args>
    void sample(Args&&...);
    template <typename... Args>
    void start(Args&&...);
    template <typename... Args>
    void stop(Args&&...);
    template <typename... Args>
    this_type& record(Args&&...);
    template <typename... Args>
    void reset(Args&&...);
    template <typename... Args>
    auto get(Args&&...) const;
    template <typename... Args>
    auto             get_labeled(Args&&...) const;
    data_type&       data();
    const data_type& data() const;

    // lightweight variants which exclude push/pop and assemble/derive
    template <typename... Args>
    void start(mpl::lightweight, Args&&...);
    template <typename... Args>
    void stop(mpl::lightweight, Args&&...);

    template <typename... Tp, typename... Args>
    void start(mpl::piecewise_select<Tp...>, Args&&...);
    template <typename... Tp, typename... Args>
    void stop(mpl::piecewise_select<Tp...>, Args&&...);

    using bundle_type::hash;
    using bundle_type::key;
    using bundle_type::laps;
    using bundle_type::rekey;
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
    // construct the objects that have constructors with matching arguments
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
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... Args>
    void mark_begin(Args&&... _args)
    {
        invoke::mark_begin(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... Args>
    void mark_end(Args&&... _args)
    {
        invoke::mark_end(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // store a value
    //
    template <typename... Args>
    void store(Args&&... _args)
    {
        invoke::store(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // perform a audit operation (typically for GOTCHA)
    //
    template <typename... Args>
    void audit(Args&&... _args)
    {
        invoke::audit(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // perform an add_secondary operation
    //
    template <typename... Args>
    void add_secondary(Args&&... _args)
    {
        invoke::add_secondary(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class OpT, typename... Args>
    void invoke(Args&&... _args)
    {
        invoke::invoke<OpT, Tag>(m_data, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

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
    //  type is not explicitly listed
    //
    template <
        typename U, typename T = decay_t<U>, typename R = remove_pointer_t<T>,
        enable_if_t<!is_one_of<T, data_type>::value && !is_one_of<T*, data_type>::value &&
                        !is_one_of<R, data_type>::value,
                    int> = 0>
    T* get() const
    {
        void*       ptr   = nullptr;
        static auto _hash = std::hash<std::string>()(demangle<T>());
        get(ptr, _hash);
        return static_cast<T*>(ptr);
    }

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

    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              enable_if_t<trait::is_available<T>::value && is_one_of<T, data_type>::value,
                          int> = 0>
    auto& get_reference()
    {
        return std::get<index_of<T, data_type>::value>(m_data);
    }

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
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args,
              enable_if_t<trait::is_available<T>::value == true &&
                              is_one_of<T*, data_type>::value == true &&
                              is_one_of<T, data_type>::value == false,
                          char> = 0>
    void init(Args&&... _args)
    {
        T*& _obj = std::get<index_of<T*, data_type>::value>(m_data);
        if(!_obj)
        {
            if(settings::debug())
            {
                printf("[component_bundle::init]> initializing type '%s'...\n",
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
                printf("[component_bundle::init]> skipping re-initialization of type"
                       " \"%s\"...\n",
                       _id.c_str());
            }
        }
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args,
              enable_if_t<trait::is_available<T>::value == true &&
                              is_one_of<T*, data_type>::value == false &&
                              is_one_of<T, data_type>::value == true,
                          char> = 0>
    void init(Args&&... _args)
    {
        T&                      _obj = std::get<index_of<T, data_type>::value>(m_data);
        operation::construct<T> _tmp(_obj, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              typename... Args,
              enable_if_t<trait::is_available<T>::value == true &&
                              is_one_of<T, data_type>::value == false &&
                              is_one_of<T*, data_type>::value == false &&
                              has_user_bundle_v == true,
                          int> = 0>
    void init(Args&&...)
    {
        using bundle_t =
            decay_t<decltype(std::get<0>(std::declval<user_bundle_types>()))>;
        static_assert(trait::is_user_bundle<bundle_t>::value, "Error! Not a user_bundle");
        this->init<bundle_t>();
        auto* _bundle = this->get<bundle_t>();
        if(_bundle)
            _bundle->insert(component::factory::get_opaque<T>(m_scope),
                            component::factory::get_typeids<T>());
    }

    //----------------------------------------------------------------------------------//

    template <typename T, typename... Args,
              enable_if_t<trait::is_available<T>::value == false ||
                              (is_one_of<T*, data_type>::value == false &&
                               is_one_of<T, data_type>::value == false &&
                               has_user_bundle_v == false),
                          int> = 0>
    void init(Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    //  variadic initialization
    //
    template <typename... T, typename... Args>
    void initialize(Args&&... args)
    {
        TIMEMORY_FOLD_EXPRESSION(this->init<T>(std::forward<Args>(args)...));
    }

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
              enable_if_t<is_one_of<T, data_type>::value == true, int> = 0>
    void type_apply(Func&& _func, Args&&... _args)
    {
        auto* _obj = get<T>();
        ((*_obj).*(_func))(std::forward<Args>(_args)...);
    }

    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T*, data_type>::value == true, int> = 0>
    void type_apply(Func&& _func, Args&&... _args)
    {
        auto* _obj = get<T*>();
        if(_obj)
            ((*_obj).*(_func))(std::forward<Args>(_args)...);
    }

    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T, data_type>::value == false &&
                              is_one_of<T*, data_type>::value == false,
                          int> = 0>
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
        using printer_t = typename bundle_type::print_t;
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
        std::string _key   = "";
        auto        keyitr = get_hash_ids()->find(m_hash);
        if(keyitr != get_hash_ids()->end())
            _key = keyitr->second;

        ar(cereal::make_nvp("hash", m_hash), cereal::make_nvp("key", _key),
           cereal::make_nvp("laps", m_laps));

        if(keyitr == get_hash_ids()->end())
        {
            auto _hash = add_hash_id(_key);
            if(_hash != m_hash)
                PRINT_HERE("Warning! Hash for '%s' (%llu) != %llu", _key.c_str(),
                           (unsigned long long) _hash, (unsigned long long) m_hash);
        }

        ar(cereal::make_nvp("data", m_data));
    }

public:
    int64_t         laps() const { return bundle_type::laps(); }
    std::string     key() const { return bundle_type::key(); }
    uint64_t        hash() const { return bundle_type::hash(); }
    void            rekey(const string_t& _key) { bundle_type::rekey(_key); }
    bool&           store() { return bundle_type::store(); }
    const bool&     store() const { return bundle_type::store(); }
    const string_t& prefix() const { return bundle_type::prefix(); }
    const string_t& get_prefix() const { return bundle_type::get_prefix(); }

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
    using bundle_type::m_hash;
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
