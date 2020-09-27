
#include "timemory/utility/socket.hpp"

#include <string>
#include <vector>

int
main()
{
    auto socket_manager = tim::socket::manager{};
    socket_manager.connect("test", "127.0.0.1", 8080);
    std::string input;
    while(true)
    {
        std::cout << "> ";
        std::getline(std::cin, input);
        if(input.empty())
            break;
        socket_manager.send("test", input);
    }
    socket_manager.close("test");

    return 0;
}