echo 开始下载：https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh
#curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
curl -o- -L -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36" https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash

echo 'export NVM_NODEJS_ORG_MIRROR=https://npmmirror.com/mirrors/node/' >> ~/.bashrc
source ~/.bashrc

nvm install 22.22.0
nvm use 22.22.0
nvm alias default 22.22.0

node -v
