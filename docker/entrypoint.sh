#!/bin/bash

# Start vfb
# Start vnc

if [ -z "$VNC_PASSWORD" ]
then
  echo '[-] Please inform a password via VNC_PASSWORD variable'
  exit -1
fi

# This cannot be done during install, except if we want a static password
if [ ! -f /root/.vnc/passwd ]
then
  mkdir /root/.vnc
  x11vnc -storepasswd $VNC_PASSWORD /root/.vnc/passwd
fi

if [ -f /tmp/.X0-lock ]
then
  rm /tmp/.X0-lock
fi

Xvfb -screen 0 1024x768x16 -ac &
env DISPLAY=:0.0 x11vnc -noxrecord -noxfixes -noxdamage -forever -display :0 &
env DISPLAY=:0.0 fluxbox
# exec "$@"
