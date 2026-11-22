param(
    [string]$Python = "E:\codex\sgdiff-part2-aligned\.venv-cuda\Scripts\python.exe",
    [string]$Peer = "E:\codex\lin-eval",
    [string]$PeerEaAssets = "E:\codex\lin-peer-assets\experiments\new_link_models\benchmarks\ea_graphs"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $repo "runs\part2-peer\driver"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$log = Join-Path $logRoot "sgdiff-local.log"
$status = Join-Path $logRoot "status.json"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
Set-Location $repo

$common = @(
    "--peer", $Peer,
    "--peer-ea-assets", $PeerEaAssets,
    "--device", "cuda",
    "--weights", "runs\part2-peer\weights",
    "--output", "runs\part2-peer\evaluation",
    "--tables", "results\part2-peer-cn-gcn-sgdiff",
    "--sgdiff-batch-size", "32",
    "--sgdiff-eval-batch-size", "64",
    "--sgdiff-finetune-epochs", "5",
    "--max-padded-pairs", "8192",
    "--num-workers", "0"
)

try {
    & $Python "scripts\run_part2_peer_models.py" "train" `
        "--models" "sgdiff" `
        "--datasets" "cora" "citeseer" "pubmed" `
        "--seeds" "0" "1" "2" @common 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "SGDiff training exited with code $LASTEXITCODE" }

    & $Python "scripts\run_part2_peer_models.py" "evaluate" `
        "--models" "cn" "gcn" "sgdiff" `
        "--datasets" "cora" "citeseer" "pubmed" `
        "--seeds" "0" "1" "2" @common 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "Aligned evaluation exited with code $LASTEXITCODE" }

    @{state="complete"; finished=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content -LiteralPath $status -Encoding utf8
}
catch {
    @{state="failed"; finished=(Get-Date).ToString("o"); error=$_.Exception.Message} |
        ConvertTo-Json | Set-Content -LiteralPath $status -Encoding utf8
    throw
}
