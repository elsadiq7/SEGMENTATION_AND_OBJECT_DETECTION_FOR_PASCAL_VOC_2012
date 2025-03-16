import os
import subprocess

# Project Configuration
PROJECT_NAME = "this_studio"  # Use your project folder name
PORT = 5000  # Update if needed
GHCR_USERNAME = "YOUR_GITHUB_USERNAME"  
GHCR_REPO = f"ghcr.io/{GHCR_USERNAME}/{PROJECT_NAME}"

# Build Docker image
print("🚀 Building Docker image...")
subprocess.run(["docker", "build", "-t", PROJECT_NAME, "."], check=True)

# Run the container
print("▶️ Running the Docker container...")
subprocess.run([
    "docker", "run", "-d", "-p", f"{PORT}:{PORT}",
    "--name", PROJECT_NAME, PROJECT_NAME
], check=True)

# Save Docker image as tar.gz
print("📦 Exporting Docker image...")
subprocess.run(["docker", "save", "-o", f"{PROJECT_NAME}.tar", PROJECT_NAME], check=True)
subprocess.run(["gzip", f"{PROJECT_NAME}.tar"], check=True)

print(f"✅ Docker image saved as {PROJECT_NAME}.tar.gz")

# Optionally push to GitHub Container Registry (GHCR)
push_to_ghcr = input("Do you want to push this image to GitHub Container Registry? (yes/no): ").strip().lower()
if push_to_ghcr == "yes":
    subprocess.run(["docker", "tag", PROJECT_NAME, GHCR_REPO], check=True)
    subprocess.run(["docker", "push", GHCR_REPO], check=True)
    print(f"🚀 Image pushed to {GHCR_REPO}")

print("✅ All done! 🚀")