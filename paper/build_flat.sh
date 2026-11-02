#!/bin/sh
# Copy the manuscript, tables and figures into one flat folder (paper/build/),
# the layout the journal requires; compile there with pdflatex and bibtex.
set -e
cd "$(dirname "$0")"
rm -rf build && mkdir build
cp manuscript.tex references.bib sn-jnl.cls sn-basic.bst build/
cp tables/*.tex build/
for n in 1 2 3 4 5; do cp "figures/Figure_$n.pdf" "build/Fig$n.pdf"; done
echo "flat source in paper/build/"
