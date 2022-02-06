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

#include "timemory/components/base/types.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/units.hpp"

#include <type_traits>

namespace tim
{
namespace component
{
template <typename Tp, typename ValueT>
struct base_units
{
    static constexpr bool is_assignable_v = trait::assignable_units<Tp>::value;
    using value_type                      = typename trait::units<Tp>::type;
    using display_type                    = typename trait::units<Tp>::display_type;
    using value_return_type =
        std::conditional_t<is_assignable_v, value_type&, value_type>;
    using display_return_type =
        std::conditional_t<is_assignable_v, display_type&, display_type>;

    template <typename Up = Tp, typename UnitT = typename trait::units<Up>::type,
              enable_if_t<std::is_arithmetic<UnitT>::value, int> = 0>
    static value_return_type unit();

    template <typename Up = Tp, typename UnitT = typename trait::units<Up>::display_type,
              enable_if_t<std::is_same<UnitT, std::string>::value, int> = 0>
    static display_return_type display_unit();

    template <typename Up = Tp, typename UnitT = typename trait::units<Up>::type,
              enable_if_t<std::is_arithmetic<UnitT>::value, int> = 0>
    static value_type get_unit();

    template <typename Up = Tp, typename UnitT = typename trait::units<Up>::display_type,
              enable_if_t<std::is_same<UnitT, std::string>::value, int> = 0>
    static display_type get_display_unit();

    template <typename Up = Tp, enable_if_t<trait::assignable_units<Up>::value, int> = 0>
    static void set_unit(value_type);

    template <typename Up                                            = Tp,
              enable_if_t<!trait::assignable_units<Up>::value, long> = 0>
    static void set_unit(value_type);

    template <typename Up = Tp, enable_if_t<trait::assignable_units<Up>::value, int> = 0>
    static void set_display_unit(display_type);

    template <typename Up                                            = Tp,
              enable_if_t<!trait::assignable_units<Up>::value, long> = 0>
    static void set_display_unit(display_type);

private:
    template <typename Up = Tp, typename UnitT = typename trait::units<Up>::type,
              enable_if_t<std::is_arithmetic<UnitT>::value, int> = 0>
    static value_type& value_impl();

    template <typename Up = Tp, typename UnitT = typename trait::units<Up>::display_type,
              enable_if_t<std::is_same<UnitT, std::string>::value, int> = 0>
    static display_type& display_impl();
};

//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, typename UnitT, enable_if_t<std::is_arithmetic<UnitT>::value, int>>
typename base_units<Tp, ValueT>::value_return_type
base_units<Tp, ValueT>::unit()
{
    return value_impl();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, typename UnitT,
          enable_if_t<std::is_same<UnitT, std::string>::value, int>>
typename base_units<Tp, ValueT>::display_return_type
base_units<Tp, ValueT>::display_unit()
{
    return display_impl();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, typename UnitT, enable_if_t<std::is_arithmetic<UnitT>::value, int>>
typename base_units<Tp, ValueT>::value_type
base_units<Tp, ValueT>::get_unit()
{
    static bool _updated  = false;
    static auto _settings = settings::shared_instance();

    if(_updated)
        return Tp::unit();
    if(!_settings)
        return Tp::unit();
    if(_settings->get_initialized())
        _updated = true;

    IF_CONSTEXPR(trait::uses_timing_units<Tp>::value)
    {
        set_unit(std::get<1>(units::get_timing_unit(_settings->get_timing_units())));
    }
    IF_CONSTEXPR(trait::uses_memory_units<Tp>::value)
    {
        set_unit(std::get<1>(units::get_memory_unit(_settings->get_memory_units())));
    }

    return Tp::unit();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, typename UnitT,
          enable_if_t<std::is_same<UnitT, std::string>::value, int>>
typename base_units<Tp, ValueT>::display_type
base_units<Tp, ValueT>::get_display_unit()
{
    static bool _updated  = false;
    static auto _settings = settings::shared_instance();

    if(_updated)
        return Tp::display_unit();
    if(!_settings)
        return Tp::display_unit();
    if(_settings->get_initialized())
        _updated = true;

    IF_CONSTEXPR(trait::uses_timing_units<Tp>::value)
    {
        set_display_unit(
            std::get<0>(units::get_timing_unit(_settings->get_timing_units())));
    }
    IF_CONSTEXPR(trait::uses_memory_units<Tp>::value)
    {
        set_display_unit(
            std::get<0>(units::get_memory_unit(_settings->get_memory_units())));
    }

    return Tp::display_unit();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, enable_if_t<trait::assignable_units<Up>::value, int>>
void
base_units<Tp, ValueT>::set_unit(value_type _v)
{
    Tp::unit() = std::move(_v);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, enable_if_t<!trait::assignable_units<Up>::value, long>>
void
base_units<Tp, ValueT>::set_unit(value_type _v)
{
    value_impl() = std::move(_v);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, enable_if_t<trait::assignable_units<Up>::value, int>>
void
base_units<Tp, ValueT>::set_display_unit(display_type _v)
{
    Tp::display_unit() = std::move(_v);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, enable_if_t<!trait::assignable_units<Up>::value, long>>
void
base_units<Tp, ValueT>::set_display_unit(display_type _v)
{
    display_impl() = std::move(_v);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, typename UnitT, enable_if_t<std::is_arithmetic<UnitT>::value, int>>
typename base_units<Tp, ValueT>::value_type&
base_units<Tp, ValueT>::value_impl()
{
    static value_type _v = []() -> value_type {
        IF_CONSTEXPR(trait::uses_timing_units<Tp>::value) { return units::sec; }
        IF_CONSTEXPR(trait::uses_memory_units<Tp>::value) { return units::megabyte; }
        IF_CONSTEXPR(trait::uses_temperature_units<Tp>::value) { return 1; }
        IF_CONSTEXPR(trait::uses_power_units<Tp>::value) { return units::watt; }
        IF_CONSTEXPR(trait::uses_percent_units<Tp>::value) { return 1; }
        return 1;
    }();
    return _v;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename ValueT>
template <typename Up, typename UnitT,
          enable_if_t<std::is_same<UnitT, std::string>::value, int>>
typename base_units<Tp, ValueT>::display_type&
base_units<Tp, ValueT>::display_impl()
{
    static display_type _v = {};
    if(_v.empty())
    {
        IF_CONSTEXPR(trait::uses_timing_units<Tp>::value)
        {
            _v = units::time_repr(unit());
        }
        IF_CONSTEXPR(trait::uses_memory_units<Tp>::value)
        {
            _v = units::mem_repr(unit());
        }
        IF_CONSTEXPR(trait::uses_temperature_units<Tp>::value) { _v = "degC"; }
        IF_CONSTEXPR(trait::uses_power_units<Tp>::value)
        {
            _v = units::power_repr(unit());
        }
        IF_CONSTEXPR(trait::uses_percent_units<Tp>::value) { _v = "%"; }
    }
    return _v;
}
}  // namespace component
}  // namespace tim
