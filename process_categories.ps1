$categories = @("Arson","Burglary","Explosion","Fighting","Normal","RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism")
$downloadDir = "$env:USERPROFILE\Downloads"
$destDir = "D:\Research_Internship\data\features\ucf-crime\i3d\deepmil_train"

foreach ($cat in $categories) {
    $zipPath = Join-Path $downloadDir "$cat.zip"
    if (-not (Test-Path $zipPath)) {
        Write-Host "SKIP: $cat.zip not found in Downloads" -ForegroundColor Yellow
        continue
    }
    Write-Host "Checking $cat.zip..." -ForegroundColor Cyan
    $result = python -c "
import zipfile
z = zipfile.ZipFile(r'$zipPath')
bad = z.testzip()
print('BAD' if bad else 'OK')
print(len(z.namelist()))
"
    $lines = $result -split "`n"
    if ($lines[0] -eq "OK") {
        Write-Host "  -> OK, $($lines[1]) entries. Extracting..." -ForegroundColor Green
        python -c "import zipfile; zipfile.ZipFile(r'$zipPath').extractall(r'$destDir')"
        Write-Host "  -> Done." -ForegroundColor Green
    } else {
        Write-Host "  -> CORRUPTED. Re-download $cat.zip" -ForegroundColor Red
    }
}
