# Trae IDE 配置 PowerShell 7 指南

## ✅ 已完成配置

已将 PowerShell 7 添加到系统环境变量 PATH：
- **PowerShell 7 路径**: `C:\Users\jinzhenfeng\AppData\Local\Microsoft\WindowsApps\pwsh.exe`
- **配置时间**: 2026-05-12

---

## 🎯 在 Trae IDE 中使用 PowerShell 7

### 方法 1：重启 Trae IDE（推荐）

1. **完全关闭 Trae IDE**
2. **重新打开 Trae IDE**
3. **打开终端**
   - 快捷键：`Ctrl + \``（反引号）
   - 或菜单：`查看` → `终端`
4. **验证版本**
   ```powershell
   $PSVersionTable.PSVersion
   ```
   应该显示 `7.x.x`

### 方法 2：在 Trae IDE 中手动选择终端

1. **打开 Trae IDE 设置**
   - 快捷键：`Ctrl + ,`
   - 或菜单：`文件` → `首选项` → `设置`

2. **搜索终端配置**
   - 搜索关键词：`terminal` 或 `shell`

3. **选择 PowerShell 7**
   - 找到终端配置文件选项
   - 选择 **PowerShell 7** 或 **pwsh**

### 方法 3：使用终端配置文件（如果支持）

如果 Trae IDE 支持 `.trae/settings.json` 配置文件：

1. **在项目目录创建 `.trae` 文件夹**
2. **创建 `.trae/settings.json` 文件**
3. **添加以下配置**：
   ```json
   {
       "terminal.integrated.defaultProfile.windows": "pwsh",
       "terminal.integrated.profiles.windows": {
           "PowerShell 7": {
               "path": "pwsh.exe",
               "icon": "terminal-powershell"
           }
       }
   }
   ```

---

## 🔍 验证配置成功

### 步骤 1：检查终端版本

在 Trae IDE 终端中输入：

```powershell
$PSVersionTable.PSVersion
```

**成功标志**：
```
Major  Minor  Build  Revision
-----  -----  -----  --------
7      6      1      0        ← PowerShell 7 ✅
```

**失败标志**（仍是 5.1）：
```
Major  Minor  Build  Revision
-----  -----  -----  --------
5      1      19041  6456     ← PowerShell 5.1 ❌
```

### 步骤 2：测试中文显示

```powershell
Write-Host "✅ Trae IDE 中文测试成功！" -ForegroundColor Green
Write-Host "PowerShell 版本：$($PSVersionTable.PSVersion)" -ForegroundColor Cyan
Write-Host "输出编码：$([Console]::OutputEncoding.EncodingName)" -ForegroundColor Cyan
```

如果显示正常中文（不是问号），说明 UTF-8 编码已生效。

---

## 🛠️ 如果配置不生效

### 问题 1：Trae IDE 没有终端配置选项

**解决方法**：
1. 关闭 Trae IDE
2. 重新打开 Trae IDE
3. 环境变量已更新，新启动的 Trae IDE 应该会自动使用 PowerShell 7

### 问题 2：终端仍是 PowerShell 5.1

**解决方法**：
1. 完全关闭 Trae IDE（包括所有窗口）
2. 重新打开 Trae IDE
3. 打开**新的**终端窗口（不要使用已打开的）

### 问题 3：找不到任何终端设置

**可能原因**：
- Trae IDE 可能使用系统默认的 PowerShell
- 或者配置方式与 VS Code 不同

**解决方法**：
1. 查看 Trae IDE 的官方文档
2. 联系 Trae IDE 技术支持
3. 或者在 Trae IDE 中直接使用 `pwsh` 命令启动 PowerShell 7

---

## 📋 手动启动 PowerShell 7（临时方案）

如果自动配置不生效，可以在 Trae IDE 终端中手动输入：

```powershell
pwsh
```

这会启动 PowerShell 7，然后您就可以使用 UTF-8 编码了。

---

## 🎉 配置完成后的效果

配置成功后，您在 Trae IDE 中：

✅ **中文显示正常**
```powershell
PS> Write-Host "什么是 LangChain？"
什么是 LangChain？
```

✅ **API 调用正常**
```powershell
PS> $body = '{"message":"什么是 LangChain？","thread_id":"test"}'
PS> Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat -Body $body
```

✅ **日志显示正常**
```
2026-05-12 10:23:01 [INFO] - 输入参数：'什么是 LangChain？'
```

---

## 📞 需要帮助？

如果以上方法都不生效，可以：

1. **检查 Trae IDE 版本**
   - 查看是否为最新版本
   - 旧版本可能不支持自定义终端

2. **查看 Trae IDE 文档**
   - 搜索 "Trae IDE 终端配置"
   - 或 "Trae IDE terminal configuration"

3. **使用外部终端**
   - 在 Trae IDE 中打开外部终端
   - 使用 PowerShell 7 运行命令

---

## 🔗 相关资源

- [PowerShell 7 官方文档](https://docs.microsoft.com/powershell/)
- [Trae IDE 使用指南](https://trae.cn/docs)
- [UTF-8 编码配置指南](./VSCode_PowerShell7_Setup_Guide.md)

---

**最后更新**: 2026-05-12  
**配置状态**: ✅ PowerShell 7 已添加到环境变量
