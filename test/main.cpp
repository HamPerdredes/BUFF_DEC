#include "buff.hpp"
#include <iostream>

using namespace std;

int main()
{
    BUFF test("./test.mov","red");
    test.process();
    cout<<"done"<<endl;
    return 0;
}