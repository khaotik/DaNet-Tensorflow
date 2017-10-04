#!/usr/bin/env bash

open -a 'Google Chrome' http://localhost:6008/#scalars
tensorboard --logdir=logs --port 6008 --reload_interval 1
