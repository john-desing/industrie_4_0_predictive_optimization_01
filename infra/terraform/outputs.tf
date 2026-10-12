output "cluster_name" {
  description = "GKE cluster name used by the deployment workflow."
  value       = google_container_cluster.main.name
}

output "cluster_location" {
  description = "GKE cluster region used by the deployment workflow."
  value       = google_container_cluster.main.location
}

output "artifact_registry_url" {
  description = "Docker registry URL for the application images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
