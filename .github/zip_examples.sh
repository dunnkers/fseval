#!/bin/bash

mkdir -p examples_zipped
cd examples

for f in $(find "." -mindepth 1 -maxdepth 1 -type d ); do
    zip -r "../examples_zipped/$f" "$f"
done