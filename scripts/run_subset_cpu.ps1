# Subset run on CPU: data prep, zero-shot baseline, LoRA fine-tuning, evaluation.
# First run takes ~4 h on a laptop CPU, mostly data prep. Safe to re-run: finished data-prep work is skipped.
#
#   .venv\Scripts\activate
#   powershell -ExecutionPolicy Bypass -File scripts\run_subset_cpu.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_subset_cpu.ps1 -BaseModel openai/whisper-small
#   powershell -ExecutionPolicy Bypass -File scripts\run_subset_cpu.ps1 -SkipDataPrep -Epochs 3   # re-train only

param(
    [string]$BaseModel = "openai/whisper-tiny",
    [int]$Epochs = 5,
    [int]$BatchSize = 1,
    [int]$GradAccum = 1,
    [int]$WarmupSteps = 20,
    [double]$LearningRate = 5e-4,
    [switch]$SkipDataPrep
)

$ErrorActionPreference = "Stop"
$model = ($BaseModel -split "/")[-1]
$run = "$model-subset-e$Epochs"
$hf = "data/subset/manifests/hf"

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n== $name ==" -ForegroundColor Cyan
    $start = Get-Date
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Write-Host ("   done in {0:N0} min" -f ((Get-Date) - $start).TotalMinutes)
}

if (-not $SkipDataPrep) {
    Step "data prep" { sc-asr data-prep --config configs/data_prep_subset.yaml }
}
$baseline = "results/$model-subset-zeroshot.json"
if (-not (Test-Path $baseline)) {
    Step "baseline" { sc-asr evaluate --manifest "$hf/test.jsonl" --base-model $BaseModel --output $baseline | Out-Null }
}
Step "fine-tune" {
    sc-asr train --train "$hf/train.jsonl" --validation "$hf/validation.jsonl" --output "models/$run" `
        --base-model $BaseModel --epochs $Epochs --batch-size $BatchSize --gradient-accumulation-steps $GradAccum `
        --warmup-steps $WarmupSteps --learning-rate $LearningRate
}
Step "evaluate" { sc-asr evaluate --manifest "$hf/test.jsonl" --base-model $BaseModel --model "models/$run" --output "results/$run.json" | Out-Null }

$before = Get-Content $baseline -Raw | ConvertFrom-Json
$after = Get-Content "results/$run.json" -Raw | ConvertFrom-Json
Write-Host ("`n{0,-12} WER {1:N3}  CER {2:N3}" -f "zero-shot", $before.wer, $before.cer)
Write-Host ("{0,-12} WER {1:N3}  CER {2:N3}   ({3} test chunks)" -f "fine-tuned", $after.wer, $after.cer, $after.examples)
