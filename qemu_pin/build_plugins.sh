#!/bin/bash

gcc -g -shared -fPIC -o fault_injector.so fault_injector.c $(pkg-config --cflags --libs glib-2.0)
gcc -g -shared -fPIC -o profiler.so profiler.c $(pkg-config --cflags --libs glib-2.0)