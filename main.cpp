#include "buff.hpp"
#include <iostream>


using namespace std;

int main()
{
    BUFF test("./video/test.mov","red");
    test.process();
    return 0;
}