```
brew install pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
source ~/.bash_profile

pyenv install 3.8.10
pyenv virtualenv 3.8.10 env-3.8.10

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate env-3.8.10
pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name env-3.8.10
```
