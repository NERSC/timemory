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

#include "timemory/macros/os.hpp"

#if defined(_WINDOWS)
#    include <WinSock2.h>
#    include <Ws2tcpip.h>
#else
#    include <arpa/inet.h>
#    include <netdb.h>
#    include <sys/socket.h>
#    include <unistd.h>
#endif

#include <atomic>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

namespace tim
{
namespace socket
{
//
#if defined(_WINDOWS)
using socket_t                      = SOCKET;
static constexpr int invalid_socket = INVALID_SOCKET;
static constexpr int socket_error   = SOCKET_ERROR;
#else
using socket_t                      = int;
static constexpr int invalid_socket = 0;
static constexpr int socket_error   = -1;
#endif
//
class manager
{
public:
    static constexpr int buffer_size = 4096;
    using socket_map_t               = std::unordered_map<std::string, socket_t>;

public:
    manager()               = default;
    ~manager()              = default;
    manager(const manager&) = default;
    manager(manager&&)      = default;

    manager& operator=(const manager&) = default;
    manager& operator=(manager&&) = default;

public:
    template <typename CallbackT>
    auto listen(const std::string& _channel_name, int _port, CallbackT&& callback)
        -> decltype(callback(_channel_name), void())
    {
        if(this->init() != 0)
        {
            std::cerr << "Can't start socket!" << std::endl;
            return;
        }

        socket_t _listening = ::socket(AF_INET, SOCK_STREAM, 0);
        if(_listening == invalid_socket)
        {
            std::cerr << "Can't create a socket!" << std::endl;
            return;
        }

        sockaddr_in _hint;
        _hint.sin_family = AF_INET;
        _hint.sin_port   = htons(_port);
#if defined(_WINDOWS)
        _hint.sin_addr.S_un.S_addr = INADDR_ANY;
#else
        _hint.sin_addr.s_addr     = INADDR_ANY;
#endif

        ::bind(_listening, (sockaddr*) &_hint, sizeof(_hint));
        ::listen(_listening, SOMAXCONN);

        sockaddr_in _client;
#if defined(_WINDOWS)
        int _client_size = sizeof(_client);
#else
        unsigned int _client_size = sizeof(_client);
#endif
        socket_t _client_socket = accept(_listening, (sockaddr*) &_client, &_client_size);
        m_server_sockets[_channel_name] = _client_socket;
        char _host[NI_MAXHOST];
        char _service[NI_MAXSERV];

        memset(_host, 0, NI_MAXHOST);
        memset(_service, 0, NI_MAXSERV);
        if(getnameinfo((sockaddr*) &_client, sizeof(_client), _host, NI_MAXHOST, _service,
                       NI_MAXSERV, 0) == 0)
        {
            std::cout << _host << " connected on port " << _service << std::endl;
        }
        else
        {
            inet_ntop(AF_INET, &_client.sin_addr, _host, NI_MAXHOST);
            std::cout << _host << " connected on port " << ntohs(_client.sin_port)
                      << std::endl;
        }
        this->close(_listening);
        char _buff[buffer_size];
        while(true)
        {
            memset(_buff, 0, buffer_size);

            int _bytes_recv = recv(_client_socket, _buff, buffer_size, 0);
            if(_bytes_recv == socket_error)
            {
                std::cerr << "Error in recv(). Quitting" << std::endl;
                break;
            }
            if(_bytes_recv > 0)
            {
                callback(std::string(_buff, 0, _bytes_recv));
            }
            else
            {
                break;
            }
        }
        this->close(_client_socket);
        this->quit();
    }

    void send(const std::string& _channel_name, const std::string& _data)
    {
        if(m_client_sockets.find(_channel_name) != m_client_sockets.end())
        {
            socket_t _sock        = m_client_sockets.at(_channel_name);
            int      _send_result = ::send(_sock, _data.c_str(), _data.size() + 1, 0);
            if(_send_result == socket_error)
            {
                std::cerr << "Can't create socket!" << std::endl;
            }
        }
    }

    void connect(const std::string& _channel_name, const std::string& _ip, int _port)
    {
        if(this->init() != 0)
        {
            std::cerr << "Can't start socket!" << std::endl;
            return;
        }

        socket_t _sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if(_sock == invalid_socket)
        {
            std::cerr << "Can't create socket!" << std::endl;
            this->quit();
            return;
        }

        sockaddr_in _hint;
        _hint.sin_family = AF_INET;
        _hint.sin_port   = htons(_port);
        inet_pton(AF_INET, _ip.c_str(), &_hint.sin_addr);

        int _conn_result = ::connect(_sock, (sockaddr*) &_hint, sizeof(_hint));
        if(_conn_result == socket_error)
        {
            std::cerr << "Can't connect to server!" << std::endl;
            this->close(_sock);
            this->quit();
            return;
        }
        m_client_sockets[_channel_name] = _sock;
    }

    void close(const std::string& _channel_name)
    {
        if(m_client_sockets.find(_channel_name) != m_client_sockets.end())
        {
            socket_t s = m_client_sockets.at(_channel_name);
            this->close(s);
            this->quit();
        }
    }

private:
    int init()
    {
#if defined(_WINDOWS)
        WSADATA _wsa_data;
        return WSAStartup(MAKEWORD(1, 1), &_wsa_data);
#else
        return 0;
#endif
    }

    int quit()
    {
#if defined(_WINDOWS)
        return WSACleanup();
#else
        return 0;
#endif
    }

    int close(socket_t _sock)
    {
        int status = 0;
#if defined(_WINDOWS)
        if((status = ::shutdown(_sock, SD_BOTH)) == 0)
            return ::closesocket(_sock);
#else
        if((status = ::shutdown(_sock, SHUT_RDWR)) == 0)
            return ::close(_sock);
#endif
        return status;
    }

private:
    socket_map_t m_client_sockets = {};
    socket_map_t m_server_sockets = {};
};
//
}  // namespace socket
}  // namespace tim
