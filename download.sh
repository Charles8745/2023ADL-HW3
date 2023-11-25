#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SkgUdM4GcFcmm3DiRwXdud1Wj9i1H2bA" -O adapter_checkpoint.zip && rm -rf /tmp/cookies.txt

unzip adapter_checkpoint

rm -rf adapter_checkpoint.zip


