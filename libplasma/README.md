# C wrapper for Plasma

First install apache arrow C++ bindings, either globally, or locally:


```bash
git clone https://github.com/apache/arrow.git
cd arrow/cpp
```

Get your system setup, as documented in the `README.md` in that folder.

Then create a release build:


```bash
mkdir release
cd release
cmake .. -DCMAKE_BUILD_TYPE=Release -DARROW_PLASMA=ON
make unittest
```

Then compile:

```fish-shell
clang++ -Wall -std=c++11 -c -I arrow/cpp/src/ plasma.cc
clang++ -Wall -std=c++11 -o libplasma.so -I arrow/cpp/src/ plasma.cc arrow/cpp/release/release/{libplasma.a,libarrow.a}
```
