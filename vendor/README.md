# vendor/

外部项目放这里，**不提交进 git**（已在 `.gitignore` 中排除，只保留本文件）。

## 拉取

```powershell
git clone https://github.com/karpathy/nanoGPT.git  vendor\nanoGPT
git clone https://github.com/karpathy/nanochat.git vendor\nanochat
```

国内网络需要代理：

```powershell
git config --global http.proxy http://127.0.0.1:7890
```

## 约定

- `vendor/` 里的代码**只读**，不改、不提交
- 要改配置，复制到 `stage*/` 目录里再改
- 想锁定版本，可以改用 submodule：

```powershell
git submodule add https://github.com/karpathy/nanoGPT.git vendor/nanoGPT
```

（submodule 会把 commit 号记进主仓库，比直接 clone 更可复现，但日常用起来麻烦一些。）
