#!/usr/bin/env bash

ipython notebook -c "import os,sys; sys.path.append(os.path.abspath('..'))" --pylab=inline --notebook-dir=notebooks/ --port=8889