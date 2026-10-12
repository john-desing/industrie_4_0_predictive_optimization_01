variable "project_id" {
  description = "Google Cloud project that hosts the GKE cluster."
  type        = string
}

variable "region" {
  description = "Google Cloud region for GKE and Artifact Registry."
  type        = string
  default     = "europe-west1"
}

variable "cluster_name" {
  description = "Name of the GKE Autopilot cluster."
  type        = string
  default     = "industrie-4-0"
}

variable "artifact_repository_id" {
  description = "Artifact Registry Docker repository name."
  type        = string
  default     = "industrie-4-0"
}
