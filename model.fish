#!/usr/bin/fish

rm -rf model/
uv run make_model.py
unzip -q model.pt2