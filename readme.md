download the repo like this:
gh repo clone lyrarobinson/clipcat_tool

download the model from here, dont unzip just add it to the directory:
https://huggingface.co/lyrarobinson/ppo_imageglitch_latest

first, create a new environment (doesn't have to be conda but i think its easier):
conda create -n clipcat_env python=3.11.7

activate the new environment:
conda activate clipcat_env

cd into it the directory you cloned this repo into:
cd C:\Users\lyra\Desktop\clipcat-tool

if necessary:
pip install --upgrade virtualenv

install the dependencies from the setup.py file (there's a few, the environment ends up being just over 2 gigabytes):
pip install .

make sure everything is in the system path properly:
set PYTHONPATH=%PYTHONPATH%;C:\Users\lyra\Desktop\clipcat-tool
set PATH=%PATH%;C:\Users\lyra\.conda\envs\clipcat_env\Scripts
set PYTHONPATH=%PYTHONPATH%;C:\path\to\directory\with\loop3

usage (it only takes 2 arguments, can use --help):
clipcat "./sampleimages" "./processed_images"

you can use it from anywhere, as long as you have the environment activated and use absolute paths :)

Also, it helps to add the environment variables to the activate.bat script, like this:
rem Add custom environment variables
set PYTHONPATH=C:\Users\lyra\Desktop\clipcat-tool;%PYTHONPATH%
set PATH=C:\Users\lyra\.conda\envs\clipcat_env\Scripts;%PATH%
set PYTHONPATH=%PYTHONPATH%;C:\path\to\directory\with\loop3

for me, the script was located at C:\ProgramData\anaconda3\envs\clipcat_env\Lib\venv\scripts\nt

if you make any changes to the code, refresh the environemnt installation with this:
pip install -e .
