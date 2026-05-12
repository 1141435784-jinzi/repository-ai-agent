# VS Code 配置 PowerShell 7 为默认终端指南

## 🎯 配置目标

将 VS Code 的默认终端设置为 **PowerShell 7 (pwsh)**，以获得更好的 UTF-8 中文支持。

---

## 📝 方法一：通过 VS Code 设置界面（推荐）

### 步骤 1：打开 VS Code 设置

1. 打开 VS Code
2. 按快捷键 `Ctrl + ,`（逗号）打开设置
3. 或者点击菜单：`文件` → `首选项` → `设置`

### 步骤 2：搜索终端配置

在设置搜索框中输入：
```
terminal.integrated.defaultProfile.windows
```

### 步骤 3：选择 PowerShell 7

1. 在下拉菜单中选择 **`pwsh`** 或 **`PowerShell 7`**
2. 如果没有看到 `pwsh` 选项，选择 **`PowerShell`** 即可

### 步骤 4：验证配置

1. 打开新的终端：`Ctrl + \``（反引号）或 `查看` → `终端`
2. 查看终端标题，应该显示 **PowerShell 7.x.x** 或 **pwsh**
3. 输入以下命令验证：
   ```powershell
   $PSVersionTable.PSVersion
   ```
   应该显示 `7.x.x` 而不是 `5.1.x`

---

## 📝 方法二：直接编辑 settings.json

### 步骤 1：打开 settings.json

1. 按 `Ctrl + Shift + P` 打开命令面板
2. 输入：`Preferences: Open Workspace Settings (JSON)`
3. 或输入：`Preferences: Open User Settings (JSON)`

### 步骤 2：添加配置

在 `settings.json` 中添加以下内容：

```json
{
    "terminal.integrated.defaultProfile.windows": "pwsh",
    "terminal.integrated.profiles.windows": {
        "PowerShell 7": {
            "source": "PowerShell",
            "icon": "terminal-powershell",
            "path": "pwsh.exe"
        }
    }
}
```

### 步骤 3：保存文件

按 `Ctrl + S` 保存文件

---

## 🔍 验证配置成功

### 检查终端版本

打开 VS Code 终端，输入：

```powershell
$PSVersionTable.PSVersion
```

**成功标志**：
```
Major  Minor  Build  Revision
-----  -----  -----  --------
7      6      1      0        ← 这是 PowerShell 7
```

**失败标志**（仍是 5.1）：
```
Major  Minor  Build  Revision
-----  -----  -----  --------
5      1      19041  6456     ← 这是 PowerShell 5.1
```

### 测试中文显示

在终端中输入：

```powershell
Write-Host "✅ PowerShell 7 中文测试成功！" -ForegroundColor Green
```

如果显示正常的中文（不是问号），说明 UTF-8 编码已生效。

---

## 🛠️ 常见问题

### 问题 1：下拉菜单中没有 `pwsh` 选项

**解决方法**：
1. 确保已安装 PowerShell 7（运行 `winget list Microsoft.PowerShell` 检查）
2. 重启 VS Code
3. 如果仍没有，手动编辑 `settings.json`（见方法二）

### 问题 2：配置后仍是 PowerShell 5.1

**解决方法**：
1. 完全关闭 VS Code
2. 重新打开 VS Code
3. 打开新终端（不要使用已打开的终端）

### 问题 3：只想在当前项目使用 PowerShell 7

**解决方法**：
1. 在项目目录创建 `.vscode\settings.json`
2. 添加上述配置
3. VS Code 会自动使用工作区设置

---

## 📋 完整配置示例（可选）

如果您想要更完整的配置，可以使用以下内容：

```json
{
    // 终端配置
    "terminal.integrated.defaultProfile.windows": "pwsh",
    "terminal.integrated.profiles.windows": {
        "PowerShell 7": {
            "source": "PowerShell",
            "icon": "terminal-powershell",
            "path": "pwsh.exe"
        },
        "PowerShell 5.1": {
            "source": "PowerShell",
            "icon": "terminal-powershell"
        },
        "Command Prompt": {
            "path": ["cmd.exe"],
            "icon": "terminal-cmd"
        }
    },
    "terminal.integrated.fontSize": 14,
    "terminal.integrated.fontFamily": "Cascadia Code, Consolas, monospace",
    
    // 文件编码
    "files.encoding": "utf8",
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000,
    
    // Python 配置
    "python.defaultInterpreterPath": "${workspaceFolder}\\agent-lab4Tare\\.venv\\Scripts\\python.exe",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    
    // 编辑器配置
    "editor.formatOnSave": true,
    "editor.tabSize": 4,
    "editor.insertSpaces": true
}
```

---

## 🎉 配置完成！

现在您的 VS Code 终端将默认使用 **PowerShell 7**，享受：
- ✅ 默认 UTF-8 编码
- ✅ 完美的中文显示
- ✅ 更多现代 PowerShell 功能
- ✅ 与 PowerShell 5.1 并存，互不影响

---

## 📞 需要帮助？

如果配置过程中遇到问题，可以：
1. 检查 PowerShell 7 是否正确安装：`pwsh -Version`
2. 查看 VS Code 版本是否最新
3. 重启 VS Code 后重试
