"""
Production deployment automation script for AlphaCare Insurance Analytics.
Handles model deployment, scaling, health checks, and rollback capabilities.
"""

import os
import subprocess
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
import logging
import docker
import boto3
from kubernetes import client, config
import yaml

class ProductionDeployment:
    """
    Production deployment manager for insurance analytics platform.
    
    Features:
    - Docker containerization
    - Kubernetes orchestration
    - AWS deployment
    - Health monitoring
    - Rollback capabilities
    - Blue-green deployment
    """
    
    def __init__(self, deployment_config: Dict):
        """Initialize deployment manager."""
        self.config = deployment_config
        self.logger = self._setup_logging()
        self.docker_client = docker.from_env()
        self.deployment_history = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup deployment logger."""
        logger = logging.getLogger('production_deploy')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler('logs/deployment.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def build_docker_images(self) -> Dict[str, str]:
        """Build Docker images for all services."""
        
        images = {}
        
        # Build main application image
        self.logger.info("Building main application Docker image...")
        
        try:
            # Build Streamlit dashboard
            dashboard_image = self.docker_client.images.build(
                path=".",
                dockerfile="Dockerfile",
                tag=f"alphacare/dashboard:{self.config['version']}",
                rm=True
            )
            images['dashboard'] = dashboard_image[0].id
            self.logger.info("Dashboard image built successfully")
            
            # Build API service
            api_image = self.docker_client.images.build(
                path=".",
                dockerfile="api/Dockerfile",
                tag=f"alphacare/api:{self.config['version']}",
                rm=True
            )
            images['api'] = api_image[0].id
            self.logger.info("API image built successfully")
            
            # Build monitoring service
            monitoring_image = self.docker_client.images.build(
                path=".",
                dockerfile="monitoring/Dockerfile",
                tag=f"alphacare/monitoring:{self.config['version']}",
                rm=True
            )
            images['monitoring'] = monitoring_image[0].id
            self.logger.info("Monitoring image built successfully")
            
        except Exception as e:
            self.logger.error(f"Docker build failed: {e}")
            raise
        
        return images
    
    def push_to_registry(self, images: Dict[str, str]) -> bool:
        """Push Docker images to container registry."""
        
        registry = self.config.get('registry', 'docker.io')
        
        try:
            for service, image_id in images.items():
                tag = f"{registry}/alphacare/{service}:{self.config['version']}"
                
                # Tag image
                image = self.docker_client.images.get(image_id)
                image.tag(tag)
                
                # Push to registry
                self.logger.info(f"Pushing {service} image to registry...")
                self.docker_client.images.push(tag)
                self.logger.info(f"{service} image pushed successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Registry push failed: {e}")
            return False
    
    def deploy_to_kubernetes(self) -> bool:
        """Deploy to Kubernetes cluster."""
        
        try:
            # Load Kubernetes config
            config.load_incluster_config() if self.config.get('in_cluster') else config.load_kube_config()
            
            # Create Kubernetes clients
            apps_v1 = client.AppsV1Api()
            core_v1 = client.CoreV1Api()
            
            # Deploy services
            self._deploy_dashboard_k8s(apps_v1, core_v1)
            self._deploy_api_k8s(apps_v1, core_v1)
            self._deploy_monitoring_k8s(apps_v1, core_v1)
            
            self.logger.info("Kubernetes deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def _deploy_dashboard_k8s(self, apps_v1, core_v1):
        """Deploy dashboard service to Kubernetes."""
        
        # Dashboard deployment
        dashboard_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "alphacare-dashboard"},
            "spec": {
                "replicas": self.config.get('dashboard_replicas', 3),
                "selector": {"matchLabels": {"app": "alphacare-dashboard"}},
                "template": {
                    "metadata": {"labels": {"app": "alphacare-dashboard"}},
                    "spec": {
                        "containers": [{
                            "name": "dashboard",
                            "image": f"alphacare/dashboard:{self.config['version']}",
                            "ports": [{"containerPort": 8501}],
                            "env": [
                                {"name": "ENVIRONMENT", "value": "production"},
                                {"name": "LOG_LEVEL", "value": "INFO"}
                            ],
                            "resources": {
                                "requests": {"memory": "512Mi", "cpu": "250m"},
                                "limits": {"memory": "1Gi", "cpu": "500m"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/_stcore/health", "port": 8501},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }]
                    }
                }
            }
        }
        
        # Dashboard service
        dashboard_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "alphacare-dashboard-service"},
            "spec": {
                "selector": {"app": "alphacare-dashboard"},
                "ports": [{"port": 80, "targetPort": 8501}],
                "type": "LoadBalancer"
            }
        }
        
        # Apply configurations
        apps_v1.create_namespaced_deployment(
            namespace="default",
            body=dashboard_deployment
        )
        
        core_v1.create_namespaced_service(
            namespace="default",
            body=dashboard_service
        )
    
    def _deploy_api_k8s(self, apps_v1, core_v1):
        """Deploy API service to Kubernetes."""
        
        # API deployment
        api_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "alphacare-api"},
            "spec": {
                "replicas": self.config.get('api_replicas', 5),
                "selector": {"matchLabels": {"app": "alphacare-api"}},
                "template": {
                    "metadata": {"labels": {"app": "alphacare-api"}},
                    "spec": {
                        "containers": [{
                            "name": "api",
                            "image": f"alphacare/api:{self.config['version']}",
                            "ports": [{"containerPort": 8000}],
                            "env": [
                                {"name": "ENVIRONMENT", "value": "production"},
                                {"name": "API_WORKERS", "value": "4"}
                            ],
                            "resources": {
                                "requests": {"memory": "1Gi", "cpu": "500m"},
                                "limits": {"memory": "2Gi", "cpu": "1000m"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }]
                    }
                }
            }
        }
        
        # API service
        api_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "alphacare-api-service"},
            "spec": {
                "selector": {"app": "alphacare-api"},
                "ports": [{"port": 80, "targetPort": 8000}],
                "type": "LoadBalancer"
            }
        }
        
        # Apply configurations
        apps_v1.create_namespaced_deployment(
            namespace="default",
            body=api_deployment
        )
        
        core_v1.create_namespaced_service(
            namespace="default",
            body=api_service
        )
    
    def _deploy_monitoring_k8s(self, apps_v1, core_v1):
        """Deploy monitoring service to Kubernetes."""
        
        # Monitoring deployment
        monitoring_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "alphacare-monitoring"},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "alphacare-monitoring"}},
                "template": {
                    "metadata": {"labels": {"app": "alphacare-monitoring"}},
                    "spec": {
                        "containers": [{
                            "name": "monitoring",
                            "image": f"alphacare/monitoring:{self.config['version']}",
                            "env": [
                                {"name": "ENVIRONMENT", "value": "production"},
                                {"name": "MONITORING_INTERVAL", "value": "300"}
                            ],
                            "resources": {
                                "requests": {"memory": "256Mi", "cpu": "100m"},
                                "limits": {"memory": "512Mi", "cpu": "200m"}
                            }
                        }]
                    }
                }
            }
        }
        
        # Apply configuration
        apps_v1.create_namespaced_deployment(
            namespace="default",
            body=monitoring_deployment
        )
    
    def deploy_to_aws(self) -> bool:
        """Deploy to AWS using ECS or EKS."""
        
        try:
            if self.config.get('aws_service') == 'ecs':
                return self._deploy_to_ecs()
            elif self.config.get('aws_service') == 'eks':
                return self._deploy_to_eks()
            else:
                self.logger.error("AWS service not specified (ecs or eks)")
                return False
                
        except Exception as e:
            self.logger.error(f"AWS deployment failed: {e}")
            return False
    
    def _deploy_to_ecs(self) -> bool:
        """Deploy to AWS ECS."""
        
        ecs_client = boto3.client('ecs', region_name=self.config['aws_region'])
        
        # Create task definition
        task_definition = {
            'family': 'alphacare-insurance-analytics',
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '1024',
            'memory': '2048',
            'executionRoleArn': self.config['ecs_execution_role'],
            'containerDefinitions': [
                {
                    'name': 'dashboard',
                    'image': f"alphacare/dashboard:{self.config['version']}",
                    'portMappings': [{'containerPort': 8501, 'protocol': 'tcp'}],
                    'essential': True,
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': '/ecs/alphacare-dashboard',
                            'awslogs-region': self.config['aws_region'],
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                },
                {
                    'name': 'api',
                    'image': f"alphacare/api:{self.config['version']}",
                    'portMappings': [{'containerPort': 8000, 'protocol': 'tcp'}],
                    'essential': True,
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': '/ecs/alphacare-api',
                            'awslogs-region': self.config['aws_region'],
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                }
            ]
        }
        
        # Register task definition
        response = ecs_client.register_task_definition(**task_definition)
        task_def_arn = response['taskDefinition']['taskDefinitionArn']
        
        # Create or update service
        service_name = 'alphacare-service'
        
        try:
            ecs_client.update_service(
                cluster=self.config['ecs_cluster'],
                service=service_name,
                taskDefinition=task_def_arn,
                desiredCount=self.config.get('desired_count', 2)
            )
            self.logger.info("ECS service updated successfully")
        except ecs_client.exceptions.ServiceNotFoundException:
            ecs_client.create_service(
                cluster=self.config['ecs_cluster'],
                serviceName=service_name,
                taskDefinition=task_def_arn,
                desiredCount=self.config.get('desired_count', 2),
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': self.config['subnets'],
                        'securityGroups': self.config['security_groups'],
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )
            self.logger.info("ECS service created successfully")
        
        return True
    
    def perform_health_checks(self, endpoints: List[str]) -> Dict[str, bool]:
        """Perform health checks on deployed services."""
        
        health_status = {}
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{endpoint}/health", timeout=30)
                health_status[endpoint] = response.status_code == 200
                
                if health_status[endpoint]:
                    self.logger.info(f"Health check passed for {endpoint}")
                else:
                    self.logger.warning(f"Health check failed for {endpoint}: {response.status_code}")
                    
            except Exception as e:
                health_status[endpoint] = False
                self.logger.error(f"Health check error for {endpoint}: {e}")
        
        return health_status
    
    def rollback_deployment(self, previous_version: str) -> bool:
        """Rollback to previous deployment version."""
        
        try:
            self.logger.info(f"Rolling back to version {previous_version}")
            
            # Update deployment config
            rollback_config = self.config.copy()
            rollback_config['version'] = previous_version
            
            # Redeploy with previous version
            if self.config.get('platform') == 'kubernetes':
                return self._rollback_kubernetes(previous_version)
            elif self.config.get('platform') == 'aws':
                return self._rollback_aws(previous_version)
            else:
                self.logger.error("Unknown deployment platform for rollback")
                return False
                
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def _rollback_kubernetes(self, previous_version: str) -> bool:
        """Rollback Kubernetes deployment."""
        
        try:
            config.load_kube_config()
            apps_v1 = client.AppsV1Api()
            
            # Rollback deployments
            deployments = ['alphacare-dashboard', 'alphacare-api', 'alphacare-monitoring']
            
            for deployment in deployments:
                apps_v1.patch_namespaced_deployment(
                    name=deployment,
                    namespace='default',
                    body={
                        'spec': {
                            'template': {
                                'spec': {
                                    'containers': [{
                                        'name': deployment.split('-')[1],
                                        'image': f"alphacare/{deployment.split('-')[1]}:{previous_version}"
                                    }]
                                }
                            }
                        }
                    }
                )
            
            self.logger.info("Kubernetes rollback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Kubernetes rollback failed: {e}")
            return False
    
    def generate_deployment_report(self) -> Dict:
        """Generate deployment report."""
        
        return {
            'deployment_timestamp': datetime.now().isoformat(),
            'version': self.config['version'],
            'platform': self.config.get('platform', 'unknown'),
            'services_deployed': ['dashboard', 'api', 'monitoring'],
            'deployment_status': 'completed',
            'health_checks': self.perform_health_checks(self.config.get('health_check_endpoints', [])),
            'rollback_available': len(self.deployment_history) > 0
        }

# Configuration and usage
def create_deployment_config() -> Dict:
    """Create deployment configuration."""
    return {
        'version': 'v1.0.0',
        'platform': 'kubernetes',  # or 'aws'
        'registry': 'your-registry.com',
        'dashboard_replicas': 3,
        'api_replicas': 5,
        'aws_region': 'us-east-1',
        'ecs_cluster': 'alphacare-cluster',
        'ecs_execution_role': 'arn:aws:iam::account:role/ecsTaskExecutionRole',
        'subnets': ['subnet-12345', 'subnet-67890'],
        'security_groups': ['sg-12345'],
        'health_check_endpoints': [
            'http://dashboard-service/health',
            'http://api-service/health'
        ]
    }

# Example usage
if __name__ == "__main__":
    config = create_deployment_config()
    deployer = ProductionDeployment(config)
    
    print("Starting production deployment...")
    
    # Build and push images
    images = deployer.build_docker_images()
    deployer.push_to_registry(images)
    
    # Deploy to platform
    if config['platform'] == 'kubernetes':
        success = deployer.deploy_to_kubernetes()
    elif config['platform'] == 'aws':
        success = deployer.deploy_to_aws()
    
    if success:
        # Generate deployment report
        report = deployer.generate_deployment_report()
        print(f"Deployment completed successfully: {report}")
    else:
        print("Deployment failed. Check logs for details.")
