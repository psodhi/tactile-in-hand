#!/bin/bash

PKG_DIR="."
wget https://www.dropbox.com/s/19jmxxoobe3ixpl/local.zip -P $PKG_DIR

unzip $PKG_DIR/local.zip -d $PKG_DIR
rm -r $PKG_DIR/local.zip