#!/bin/bash
rm -fr CMakeFiles
rm CMakeCache.txt
cmake .
make clean
make
sudo make install