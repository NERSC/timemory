#pragma once

#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

/// Alias template for enable_if
// template <bool B, typename T>
// using enable_if_t = typename std::enable_if<B, T>::type;

namespace impl
{
template <std::size_t _N, typename _Func, typename... _Args,
          typename std::enable_if<(_N == 1), char>::type = 0>
inline void
unroll(_Func&& __func, _Args&&... __args)
{
    std::forward<_Func>(__func)(std::forward<_Args>(__args)...);
}

template <std::size_t _N, typename _Func, typename... _Args,
          typename std::enable_if<(_N > 1), char>::type = 0>
inline void
unroll(_Func&& __func, _Args&&... __args)
{
    std::forward<_Func>(__func)(std::forward<_Args>(__args)...);
    impl::unroll<_N - 1, _Func, _Args...>(std::forward<_Func>(__func),
                                          std::forward<_Args>(__args)...);
}
}  // namespace impl

template <std::size_t _N, typename _Func, typename... _Args>
static void
unroll(_Func&& __func, _Args&&... __args)
{
    impl::unroll<_N, _Func, _Args...>(std::forward<_Func>(__func),
                                      std::forward<_Args>(__args)...);
}

#define REP2(S)                                                                          \
    S;                                                                                   \
    S
#define REP4(S)                                                                          \
    REP2(S);                                                                             \
    REP2(S)
#define REP8(S)                                                                          \
    REP4(S);                                                                             \
    REP4(S)
#define REP16(S)                                                                         \
    REP8(S);                                                                             \
    REP8(S)
#define REP32(S)                                                                         \
    REP16(S);                                                                            \
    REP16(S)
#define REP64(S)                                                                         \
    REP32(S);                                                                            \
    REP32(S)
#define REP128(S)                                                                        \
    REP64(S);                                                                            \
    REP64(S)
#define REP256(S)                                                                        \
    REP128(S);                                                                           \
    REP128(S)
#define REP512(S)                                                                        \
    REP256(S);                                                                           \
    REP256(S)
