param(
    [string]$RepoRoot = '\\wsl$\Ubuntu-20.04\usr\local\python_projects\new_androserum',
    [string]$OutDir = 'C:\Users\Lenovo\Downloads\lamda240_apks',
    [int]$Timeout = 300,
    [int]$Retries = 5
)

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

$csvPath = Join-Path $RepoRoot "configs\lamda_binary_eval_240_candidates.csv"
$envPath = Join-Path $RepoRoot ".env.local"
$shaDir = Join-Path $RepoRoot "data_lamda_eval"
$shaFile = Join-Path $shaDir "lamda_full_240.sha.txt"

if (-not (Test-Path $csvPath)) {
    throw "Missing CSV: $csvPath"
}

Step "1/4 build lamda240 SHA list"
New-Item -ItemType Directory -Force -Path $shaDir | Out-Null
$rows = Import-Csv $csvPath
$shaList = @($rows.sha256 | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ })
$shaList | Set-Content -Encoding ASCII $shaFile
Write-Host "sha file: $shaFile"
Write-Host "count   : $($shaList.Count)"

Step "2/4 load ANDROZOO_APIKEY from .env.local"
if (-not (Test-Path $envPath)) {
    throw "Missing .env.local: $envPath"
}
$apikeyLine = Get-Content $envPath | Where-Object { $_ -match '^\s*ANDROZOO_APIKEY=' } | Select-Object -First 1
if (-not $apikeyLine) {
    throw "ANDROZOO_APIKEY not found in $envPath"
}
$apikey = ($apikeyLine -split '=', 2)[1].Trim().Trim('"').Trim("'")
if (-not $apikey) {
    throw "Empty ANDROZOO_APIKEY in $envPath"
}
$env:ANDROZOO_APIKEY = $apikey
Write-Host "apikey loaded"

Step "3/4 create output dir"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "out dir: $OutDir"

Step "4/4 download lamda240 via native Windows HTTP"
$failureLog = Join-Path $OutDir "download_failures_windows.log"
if (Test-Path $failureLog) {
    Remove-Item $failureLog -Force
}

$ok = 0
$skip = 0
$fail = 0
$total = $shaList.Count

for ($i = 0; $i -lt $total; $i++) {
    $sha = $shaList[$i]
    $target = Join-Path $OutDir "$sha.apk"

    Write-Progress -Activity "Downloading lamda240" -Status "$($i+1) / $total : $sha" -PercentComplete ((($i + 1) / $total) * 100)

    if ((Test-Path $target) -and ((Get-Item $target).Length -gt 0)) {
        $skip++
        continue
    }

    $success = $false
    $lastErrorText = ""
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            $uri = "https://androzoo.uni.lu/api/download?apikey=$apikey&sha256=$sha"
            Invoke-WebRequest -Uri $uri -OutFile $target -TimeoutSec $Timeout | Out-Null
            if ((Test-Path $target) -and ((Get-Item $target).Length -gt 0)) {
                $success = $true
                break
            }
            $lastErrorText = "empty file after download"
        }
        catch {
            $lastErrorText = $_.Exception.Message
            Start-Sleep -Seconds 2
        }
    }

    if ($success) {
        $ok++
    }
    else {
        $fail++
        Add-Content -Path $failureLog -Value "[FAIL] $sha after $Retries retries: $lastErrorText"
        if (Test-Path $target) {
            Remove-Item $target -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host ""
Write-Host "Done. OK=$ok SKIP=$skip FAIL=$fail"
if (Test-Path $failureLog) {
    Write-Host "failure log: $failureLog"
}

Write-Host ""
Write-Host "done" -ForegroundColor Green
Write-Host "downloaded dir: $OutDir"
