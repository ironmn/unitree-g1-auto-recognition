param(
    [ValidateSet('Collect', 'Replay', 'Validate')]
    [string]$Mode = 'Collect',
    [string]$RunDirectory = '',
    [ValidateSet('press','rotate','toggle')]
    [string]$Action = 'press',
    [string]$WaypointOverride = '',
    [string]$Python = 'python',
    [string]$OfficialRoot = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not $OfficialRoot) { $OfficialRoot = Join-Path (Split-Path $projectRoot -Parent) 'Binjiang_Competition' }
$officialRoot = [System.IO.Path]::GetFullPath($OfficialRoot)
$collectionPython = (Get-Command $Python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $officialRoot)) { throw 'Binjiang_Competition checkout is missing.' }
if (-not $RunDirectory) {
    if ($Mode -ne 'Collect') { throw 'Replay/Validate requires -RunDirectory pointing to an existing run.' }
    $RunDirectory = Join-Path $projectRoot ('data\official_demo_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$RunDirectory = [System.IO.Path]::GetFullPath($RunDirectory)
$datasetPath = Join-Path $RunDirectory 'dataset_verified'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
$configPath = Join-Path $projectRoot 'configs\official_g1_buttons.yaml'
$wrapperPath = Join-Path $PSScriptRoot 'run_official_demo.py'
$entryDirectory = Join-Path $officialRoot 'src\examples\dataCollection\unitree_g1'
$waypointFile = "my_waypoint_button/marked/my_waypoint_${Action}_01.yaml"
if ($WaypointOverride) { $waypointFile = (Resolve-Path -LiteralPath $WaypointOverride).Path }
$repoId = "local/g1_${Action}_24fps"

if ($Mode -eq 'Validate') {
    & $collectionPython (Join-Path $PSScriptRoot 'validate_official_dataset.py') $datasetPath --output (Join-Path $RunDirectory 'validation')
    if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed.' }
    return
}

Push-Location $entryDirectory
try {
    if ($Mode -eq 'Collect') {
        if (Test-Path -LiteralPath $RunDirectory) { throw 'Collect requires a new run directory; existing data will not be overwritten.' }
        New-Item -ItemType Directory -Path $RunDirectory | Out-Null
        $auditPath = Join-Path $RunDirectory 'capture_verified_audit'
        & $collectionPython $wrapperPath --official-root $officialRoot --audit-output $auditPath -- --task_config $configPath --agent_name g1_pick --waypoint_files $waypointFile --lerobot_out $datasetPath --repo_id $repoId --fps 24 --num_episodes 1 --clock sim --cameras head,wrist_r --cam_resolution 960x1280 --joint_strip on --strip_col off --time_step 0.001 --frame_skip 5 --track_log_every 300 *> (Join-Path $RunDirectory 'capture.log')
        if ($LASTEXITCODE -ne 0) { throw 'Capture failed; inspect capture.log.' }
    } else {
        $auditPath = Join-Path $RunDirectory ('replay_audit_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
        $replayLogPath = Join-Path $RunDirectory ((Split-Path $auditPath -Leaf) + '.log')
        & $collectionPython $wrapperPath --official-root $officialRoot --entry replay --audit-output $auditPath -- --dataset_dir $datasetPath --task_config $configPath --agent_name g1_pick --episode 1 --episodes 1 --replay_fps 24 --joint_strip on --strip_col off --time_step 0.001 --frame_skip 5 --track_log_every 50 *> $replayLogPath
        if ($LASTEXITCODE -ne 0) { throw 'Replay failed; inspect replay.log.' }
    }
} finally {
    Pop-Location
}
if ($Mode -eq 'Collect') {
    & $collectionPython (Join-Path $PSScriptRoot 'validate_official_dataset.py') $datasetPath --output (Join-Path $RunDirectory 'validation')
    if ($LASTEXITCODE -ne 0) { throw 'Captured dataset did not pass validation.' }
}
Write-Output $RunDirectory
