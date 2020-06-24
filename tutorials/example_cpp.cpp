#include <iostream>
#include <vector>
#include <string>

using namespace std;

int test_f()
{
    return 8;
}

int main()
{
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};
    
    // for (const string& word : msg)
    // {
    //     cout << word << " ";
    // }
    // cout << endl;
    std::string s = "asdf";
    
    std::string token = s.substr(0, s.find("d"));
    return 0;
}

