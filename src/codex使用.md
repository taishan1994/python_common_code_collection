# 安装

```shell
# 安装 nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 重启终端或执行以下命令使配置生效
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 安装最新 LTS 版本的 Node.js
nvm install --lts

# 验证
node -v
npm -v

# 安装 codex
npm i -g @openai/codex
```

使用虚拟手机号注册一个openai账号并登录chatgpt。

使用代充的方式升级账号为plus：https://ai.muooy.com/#faq

使用codex device在服务器上登录该账号：

- ChatGPT → 设置 → 安全（Security）
- 找到与 Codex CLI 相关的设置项
- 开启「为 Codex 启用设备码授权」（类似 wording）

`codex login --device-auth`
