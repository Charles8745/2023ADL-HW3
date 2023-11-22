#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-8F74EFp7x7t6EjnokAlitieAspipU63" -O peft_config.zip && rm -rf /tmp/cookies.txt

unzip peft_config

rm -rf peft_config.zip


