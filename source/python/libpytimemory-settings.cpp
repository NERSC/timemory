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

#if !defined(TIMEMORY_PYSETTINGS_SOURCE)
#    define TIMEMORY_PYSETTINGS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/settings.hpp"

using string_t = std::string;

#define SETTING_PROPERTY(TYPE, FUNC)                                                     \
    settings.def_property_static(                                                        \
        TIMEMORY_STRINGIZE(FUNC), [](py::object) { return tim::settings::FUNC(); },      \
        [](py::object, TYPE v) { tim::settings::FUNC() = v; },                           \
        "Binds to 'tim::settings::" TIMEMORY_STRINGIZE(FUNC) "()'")

//======================================================================================//
//
namespace pysettings
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Up, typename... Tail,
          tim::enable_if_t<(sizeof...(Tail) == 0), int> = 0>
auto
add_property(py::class_<tim::settings>& _class, std::shared_ptr<tim::vsettings> _obj)
{
    using Tp   = tim::decay_t<Up>;
    auto _tidx = std::type_index(typeid(Tp));

    if(_obj && _obj->get_type_index() == _tidx)
    {
        bool _is_ref = dynamic_cast<tim::tsettings<Tp, Tp&>*>(_obj.get()) != nullptr;
        /*
        auto _env    = _obj->get_env_name();
        // member property
        _class.def_property(
            _obj->get_name().c_str(),
            [_env](tim::settings* _object) { return _object->get<Tp>(_env); },
            [_env](tim::settings* _object, Tp v) { return _object->set(_env, v); },
            _obj->get_description().c_str());
        */
        // static property
        if(!_is_ref)
        {
            _class.def_property_static(
                _obj->get_name().c_str(),
                [_obj](py::object) {
                    return (_obj) ? static_cast<tim::tsettings<Tp>*>(_obj.get())->get()
                                  : Tp{};
                },
                [_obj](py::object, Tp v) {
                    if(_obj)
                        static_cast<tim::tsettings<Tp>*>(_obj.get())->get() = v;
                },
                _obj->get_description().c_str());
            return true;
        }
        else
        {
            _class.def_property_static(
                _obj->get_name().c_str(),
                [_obj](py::object) {
                    return (_obj)
                               ? static_cast<tim::tsettings<Tp, Tp&>*>(_obj.get())->get()
                               : Tp{};
                },
                [_obj](py::object, Tp v) {
                    if(_obj)
                        static_cast<tim::tsettings<Tp, Tp&>*>(_obj.get())->get() = v;
                },
                _obj->get_description().c_str());
            return true;
        }
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename... Tail, tim::enable_if_t<(sizeof...(Tail) > 0), int> = 0>
auto
add_property(py::class_<tim::settings>& _class, std::shared_ptr<tim::vsettings> _obj)
{
    auto ret = add_property<Tp>(_class, _obj);
    if(!ret)
        return add_property<Tail...>(_class, _obj);
    else
        return ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tail>
auto
add_property(py::class_<tim::settings>& _class, std::shared_ptr<tim::vsettings> _obj,
             tim::type_list<Tail...>)
{
    return add_property<Tail...>(_class, _obj);
}
//
//--------------------------------------------------------------------------------------//
//
py::class_<tim::settings>
generate(py::module& _pymod)
{
    // auto pyargparse = py::module::import("argparse");
    py::class_<tim::settings> settings(_pymod, "settings",
                                       "Global configuration settings for timemory");

    auto _init = []() {
        auto ret = new tim::settings{};
        *ret     = *tim::settings::instance<tim::api::native_tag>();
        return ret;
    };
    auto _parse = [](py::object _instance) {
        auto* _obj = _instance.cast<tim::settings*>();
        if(_instance.is_none() || _obj == nullptr)
            tim::settings::parse();
        else
            tim::settings::parse(_obj);
    };
    auto _read = [](py::object _instance, std::string inp) {
        auto* _obj = _instance.cast<tim::settings*>();
        if(_instance.is_none() || _obj == nullptr)
            tim::settings::instance()->read(inp);
        else
            _obj->read(inp);
    };

    // create an instance
    settings.def(py::init(_init), "Create a copy of the global settings");
    // to parse changes in env vars
    settings.def_static("parse", _parse,
                        "Update the values of the settings from the current environment",
                        py::arg("instance") = py::none{});
    settings.def_static("read", _read, "Read the settings from JSON or text file");

    std::set<std::string> names;
    auto                  _settings = tim::settings::instance<tim::api::native_tag>();
    if(_settings)
    {
        for(auto& itr : *_settings)
        {
            if(!add_property(settings, itr.second, tim::settings::data_type_list_t{}))
                names.insert(itr.second->get_name());
        }
        if(names.size() > 0)
        {
            std::stringstream ss, msg;
            for(auto& itr : names)
                ss << ", " << itr;
            msg << "Warning! The following settings were not added to python: "
                << ss.str().substr(2);
            std::cerr << msg.str() << std::endl;
        }

        using strvector_t = std::vector<std::string>;
        SETTING_PROPERTY(strvector_t, command_line);
        SETTING_PROPERTY(strvector_t, environment);
    }

    return settings;
}
}  // namespace pysettings
//
//======================================================================================//
