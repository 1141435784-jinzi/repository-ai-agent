# PowerShell UTF-8 自动配置
# 此文件会在每次启动 PowerShell 时自动执行

# 设置控制台编码为 UTF-8
if ($PSVersionTable.PSVersion.Major -le 5) {
    # PowerShell 5.1 及以下版本
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8
    chcp 65001 | Out-Null
    Write-Host "✅ 已自动配置 UTF-8 编码 (PowerShell $($PSVersionTable.PSVersion))" -ForegroundColor Green
} else {
    # PowerShell 7+ 默认就是 UTF-8
    Write-Host "✅ PowerShell 7+ 默认使用 UTF-8 (版本：$($PSVersionTable.PSVersion))" -ForegroundColor Green
}
