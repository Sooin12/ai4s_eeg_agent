# GitHub 推送故障诊断

> 日期：2026-08-12  
> 远端：`https://github.com/Sooin12/ai4s_eeg_agent.git`

## 结论

仓库地址、提交身份和 GitHub 凭据均已配置。之前无法推送的直接原因是 Codex 默认
沙箱禁止连接 `github.com:443`，不是远端仓库不存在，也不是 GitHub Desktop 登录
失效。

## 证据

- `origin` 的 fetch/push URL 均正确；
- 仓库提交身份为 `Sooin12 <256255154+Sooin12@users.noreply.github.com>`；
- Git Credential Manager 已列出 GitHub 账户 `Sooin12`，未输出任何token；
- 沙箱内执行 `git ls-remote origin HEAD`：连接 GitHub 443 失败；
- 经用户授权在沙箱外执行同一类HTTPS读取：成功返回远端HEAD
  `a94d80046a020ba563985fae9fbd18ee8f106370`。

## 修复方式

后续 fetch/push 必须允许 Git 命令在 Codex 沙箱外联网。无需改成SSH，也无需重建
GitHub账户或重填提交邮箱。若系统弹出Git Credential Manager窗口，选择已登录的
`Sooin12` 即可。

沙箱内创建的工作区在沙箱外会触发 Git `dubious ownership` 检查。已采用仅对单条
命令生效的 `-c safe.directory=D:/桌面/ai4s`，没有修改用户全局Git配置。

目前本地分支是 `feature/research-design-agent`。推送命令：

```powershell
git push -u origin feature/research-design-agent
```

最终推送已成功：远端分支从 `e9e6226` 更新到 `4c1a4f8`。
