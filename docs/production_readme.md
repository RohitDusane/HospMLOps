## END-TO-END PRODUCTION MLOPS IMPLEMENTATION PLAN
### Hospital Readmission Prediction System - GCP + DVC + GitHub Actions + GKE

### 🔧 PHASE 1: GCP INFRASTRUCTURE SETUP

'''text
- GCP project creation & billing
- Service accounts & IAM
- Cloud Storage buckets
- Artifact Registry setup
- GKE cluster provisioning
'''

#### Step 1.1: GCP Project & Billing Setup

:: Login to GCP
gcloud auth login

:: Set variables
set PROJECT_ID=hospital-readmission-ml
set REGION=us-central1
set ZONE=us-central1-a

:: Create new project (if not exists already)
gcloud projects create %PROJECT_ID% --name="Hospital Readmission ML"

:: Set default project
gcloud config set project %PROJECT_ID%

:: Enable APIs
gcloud services enable compute.googleapis.com ^
  container.googleapis.com ^
  storage.googleapis.com ^
  artifactregistry.googleapis.com ^
  cloudbuild.googleapis.com ^
  run.googleapis.com

The GCP project is Created, Acitve and Accessible.

:: Set deafult region/zone
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a


#### Step 1.2: Create Service Account
We can create a servie account, role assignement and KEY download.
1. Create Service ACCOUNT - IAM & SA --> CREATE SA --> SA name: mlops-service-account --> Create & Continue
2. Grant ROles(permissions): Roles add Storage Admin, Kubernetes Engine Admin, Container Admin, Artifact Registry Admin, Cloud Run Admin --> Continue
Skip USER ACCESS --> Done
3. Create & download SA-KEY
 go to SA page. find mlops-service-account --> click `three-dot-menu` --> Manage Keys --> Add Key --> Create new key --> JSON format --> 'CREATE'

 STORE THE KEY IN GITHUB REPO SECRETS - Repo Settings -> Secrets -> Actions -> New Repo secret -> GCP_SA_KEY -> 'PASTE KEY'

#### Step 1.3: Create Cloud Storage Buckets
1. IN you project --> Cloud Storage --> Buckets --> 'Create Bucket' 
    a. **DVC data storage** → `hospital-readmission-ml-dvc-data`
    b. **MLflow artifacts** → `hospital-readmission-ml-mlflow-artifacts`
    c. **Model Registry** → `hospital-readmission-ml-models`

2. Model Versioning for Model Registry
If your goal is **ML model versioning**:

* Keep **all noncurrent versions** until you decide they’re no longer needed.
* Then, add a **lifecycle rule** to delete older versions after, e.g., 90 days, to save costs:

Steps:

1. Go to your bucket → **Lifecycle** → **Add Rule**
2. **Select action:** `Delete object`
3. **Select conditions:**

   * `Age in days` → 90 (or whatever makes sense for your project)
   * `Matches storage class` → Optional
   * If versioning is enabled, this applies to **noncurrent versions**
4. **Save rule**

> ⚠️ Reminder: Lifecycle rules **may take up to 24 hours** to take effect.


#### Step 1.4: Setup Artifact Registry for Docker Images
## **1️⃣ Create a Docker repository in Artifact Registry**

1. Go to Artifact Registry:
   [https://console.cloud.google.com/artifacts](https://console.cloud.google.com/artifacts)

2. Make sure your project is set to `hospital-readmission-ml`.

3. Click **+ CREATE REPOSITORY**.

4. Fill in the details:

| Field           | Value                                             |
| --------------- | ------------------------------------------------- |
| **Name**        | `hospital-readmission-repo`                       |
| **Format**      | Docker                                            |
| **Location**    | `us-central1` (or your region)                    |
| **Description** | Docker images for hospital readmission ML service |

5. Click **Create** ✅

---

## **2️⃣ Configure Docker authentication**

Artifact Registry requires authentication for `docker push` and `docker pull`.
### Using **Service Account** (optional for CI/CD)

1. Go to **IAM & Admin → Service Accounts**.
2. Create a service account (or use `mlops-service-account`) with the role: **Artifact Registry Reader/Writer**.
3. Download the JSON key.
4. Authenticate Docker:

```bash
gcloud artifacts repositories list --location=us-central1
```

```bash
cat mlops-sa-key.json | docker login -u _json_key --password-stdin https://us-central1-docker.pkg.dev
```

✅ After this, your Docker repo is ready to **store ML service images**, and Docker is configured to push/pull images.

#### Step 1.5: Provision GKE Cluster
We can create a Kubernetes CLUSTER.
1. GOTO GKE --> CLUSTERS --> CREATE CLUSTER --> name: hospital-readmission-cluster --> Create

'''bash
:: ===============================
:: Step 1: Set environment variables
:: ===============================
set CLUSTER_NAME=hospital-readmission-cluster

:: ===============================
:: Step 2: Create GKE cluster
:: ===============================
gcloud container clusters create %CLUSTER_NAME% ^
    --zone %ZONE% ^
    --num-nodes 3 ^
    --machine-type n1-standard-2 ^
    --disk-size 50GB ^
    --enable-autoscaling ^
    --min-nodes 2 ^
    --max-nodes 5 ^
    --enable-autorepair ^
    --enable-autoupgrade ^
    --maintenance-window-start="2024-01-01T00:00:00Z" ^
    --maintenance-window-duration=4h ^
    --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver ^
    --workload-pool=%PROJECT_ID%.svc.id.goog


:: ===============================
:: Step 3: Get cluster credentials
:: ===============================
gcloud container clusters get-credentials %CLUSTER_NAME% --zone %ZONE%


:: ===============================
:: Step 4: Verify cluster
:: ===============================
kubectl cluster-info
kubectl get nodes

'''



### 📦 PHASE 2: DVC DATA VERSIONING

#### Step 2.1: Initialize DVC in Project
<!-- 
# Install DVC with GCS support
pip install dvc[gs] -->

# Initialize DVC
dvc init

# Add DVC to git
git add .dvc .dvcignore
git commit -m "Initialize DVC"