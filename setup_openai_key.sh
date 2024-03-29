#!/bin/bash

read -p "Enter your OpenAI API Key: " openai_key

if [ -z "$openai_key" ]; then
  echo "No OpenAI API Key entered. Exiting..."
  exit 1
fi

echo "export OPENAI_API_KEY='$openai_key'" >> ~/.bash_profile
echo "OpenAI API Key has been added to your .bash_profile."

source ~/.bash_profile
echo "Your current OpenAI API Key is: $OPENAI_API_KEY"

exec bash