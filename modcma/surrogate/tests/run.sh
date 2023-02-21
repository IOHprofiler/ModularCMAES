#!/bin/bash
# Parallel testing of python files

unittest-parallel -j 32 -v -t . --class-fixtures --disable-process-pooling
