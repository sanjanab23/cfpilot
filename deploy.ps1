# Azure Deployment Script for CF Pilot
# Usage: ./deploy.ps1

# --- CONFIGURATION (EDIT THESE IF NEEDED) ---
$location = "centralindia"        # ci = Central India
$resourceGroup = "rg-ci-cfpilot-dev" 
# Note: ACR names must be globally unique and alphanumeric only (no hyphens).
$acrName = "acrciacfpilotdev"     
$appServicePlan = "asp-ci-cfpilot-dev"
$webappName = "app-ci-cfpilot-dev"
$imageName = "cfpilot-backend"
$tagName = "latest"

# --- 1. LOGIN & PRE-CHECKS ---
Write-Host "Checking Azure CLI..." -ForegroundColor Cyan
$azInstalled = Get-Command az -ErrorAction SilentlyContinue
if (-not $azInstalled) {
    Write-Error "Azure CLI (az) is not installed. Please install it first."
    exit 1
}

# Login check
$account = az account show --output json | ConvertFrom-Json
if (-not $account) {
    Write-Host "Logging into Azure..." -ForegroundColor Yellow
    az login
} else {
    Write-Host "Already logged in as $($account.user.name)" -ForegroundColor Green
}

# --- 2. CREATE RESOURCE GROUP ---
Write-Host "Creating Resource Group '$resourceGroup' in '$location'..." -ForegroundColor Cyan
az group create --name $resourceGroup --location $location

# --- 3. CREATE & LOGIN TO ACR ---
Write-Host "Checking/Creating ACR '$acrName'..." -ForegroundColor Cyan
$acrExists = az acr show --name $acrName --resource-group $resourceGroup --output none 2>$null
if (-not $lastExitCode -eq 0) {
    az acr create --resource-group $resourceGroup --name $acrName --sku Basic --admin-enabled true
}
Write-Host "Logging into ACR..." -ForegroundColor Cyan
az acr login --name $acrName

# --- 4. BUILD & PUSH DOCKER IMAGE ---
$acrServer = az acr show --name $acrName --resource-group $resourceGroup --query "loginServer" --output tsv
$fullImageName = "$acrServer/$imageName`:$tagName"

Write-Host "Building Docker Image '$fullImageName'..." -ForegroundColor Cyan
docker build -t $fullImageName .

Write-Host "Pushing Docker Image to ACR..." -ForegroundColor Cyan
docker push $fullImageName

# --- 5. CREATE APP SERVICE PLAN (LINUX) ---
Write-Host "Checking/Creating App Service Plan '$appServicePlan'..." -ForegroundColor Cyan
az appservice plan create --name $appServicePlan --resource-group $resourceGroup --sku B1 --is-linux

# --- 6. CREATE & CONFIGURE WEB APP ---
Write-Host "Checking/Creating Web App '$webappName'..." -ForegroundColor Cyan
az webapp create --resource-group $resourceGroup --plan $appServicePlan --name $webappName --deployment-container-image-name $fullImageName

# Configure to pull from ACR
$acrPassword = az acr credential show --name $acrName --resource-group $resourceGroup --query "passwords[0].value" --output tsv
az webapp config container set --name $webappName --resource-group $resourceGroup `
    --docker-custom-image-name $fullImageName `
    --docker-registry-server-url "https://$acrServer" `
    --docker-registry-server-user $acrName `
    --docker-registry-server-password $acrPassword

# --- 7. CONFIGURE ENVIRONMENT VARIABLES (APP SETTINGS) ---
Write-Host "Configuring App Settings..." -ForegroundColor Cyan
# IMPORTANT: DB_PATH set to /app/data/cfpilot.db to use volume (configured later manually or via script)
# You should add your specific secrets here or via Portal
az webapp config appsettings set --name $webappName --resource-group $resourceGroup --settings `
    DB_PATH="/app/data/cfpilot.db" `
    WEBSITES_PORT="8000" `
    WEB_CONCURRENCY="2" `
    ALLOWED_ORIGINS="http://localhost:8000,http://127.0.0.1:8000"

# --- 8. MOUNT STORAGE FOR DB PERSISTENCE ---
# Step 8.1: Create Storage Account
$storageName = "stcfpilot" + (Get-Random -Minimum 1000 -Maximum 9999)
Write-Host "Creating Storage Account '$storageName' for DB persistence..." -ForegroundColor Cyan
# Check if we already have a storage account stored in a local file or tag? 
# For simplicity, creates a new one if not tracking. 
# Ideally, you'd check resources in the group.
# Let's try to find an existing one to avoid duplicates on re-run
$existingStorage = az storage account list --resource-group $resourceGroup --query "[0].name" --output tsv
if ($existingStorage) {
    $storageName = $existingStorage
    Write-Host "Using existing storage account: $storageName"
} else {
    az storage account create --resource-group $resourceGroup --name $storageName --sku Standard_LRS
}

# Step 8.2: Create File Share
$shareName = "cfpilot-data"
$storageKey = az storage account keys list --resource-group $resourceGroup --account-name $storageName --query "[0].value" --output tsv
az storage share create --account-name $storageName --account-key $storageKey --name $shareName

# Step 8.3: Mount to Web App
Write-Host "Mounting Azure File Share to /app/data..." -ForegroundColor Cyan
# Define a custom id for the mount
$mountId = "cfpilot-db-mount"
# Check if already mounted to avoid error
$mounts = az webapp config storage-account list --resource-group $resourceGroup --name $webappName --output json | ConvertFrom-Json
if (-not ($mounts | Where-Object name -eq $mountId)) {
    az webapp config storage-account add --resource-group $resourceGroup --name $webappName `
        --custom-id $mountId `
        --storage-type AzureFiles `
        --share-name $shareName `
        --account-name $storageName `
        --access-key $storageKey `
        --mount-path "/app/data"
}

# --- 9. RESTART & URL ---
Write-Host "Restarting Web App..." -ForegroundColor Cyan
az webapp restart --name $webappName --resource-group $resourceGroup

$appUrl = "https://$webappName.azurewebsites.net"
Write-Host "------------------------------------------------------" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "App URL: $appUrl" -ForegroundColor Green
Write-Host "NOTE: You must manually add your secrets (OPENROUTER_API_KEY, DB_ENCRYPTION_KEY) in the Azure Portal > Environment Variables." -ForegroundColor Yellow
Write-Host "------------------------------------------------------" -ForegroundColor Green
