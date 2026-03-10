<#
 tools/optimize-models.ps1
 Пример пакетной оптимизации .glb файлов в папке 3d/ с помощью gltf-pipeline (DRACO compression).

 Требования:
 - Node.js/npm
 - Установить gltf-pipeline глобально: npm i -g gltf-pipeline
   или использовать npx (в примере используется npx)

 Запуск (PowerShell):
   .\tools\optimize-models.ps1

#>

$inputDir = Join-Path $PSScriptRoot "..\3d"
$outDir = Join-Path $PSScriptRoot "..\backups\optimized"

if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

Get-ChildItem -Path $inputDir -Filter *.glb | ForEach-Object {
    $in = $_.FullName
    $out = Join-Path $outDir $_.Name
    Write-Host "Optimizing $($_.Name) -> $out"
    # Пример с draco сжатием (потребуется установленный gltf-pipeline)
    npx gltf-pipeline -i "$in" -o "$out" -d --draco.compressMeshes
}

Write-Host "Готово. Оптимизированные файлы в: $outDir"

