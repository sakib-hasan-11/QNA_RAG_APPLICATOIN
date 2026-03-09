"""
Docker and ECR utilities for building and pushing images.

This module provides functions to:
1. Authenticate with AWS ECR
2. Build Docker images
3. Tag and push images to ECR
"""

import os
import subprocess
import sys

import boto3
from botocore.exceptions import ClientError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger

logger = get_logger(__name__)


class DockerECRManager:
    """
    Manages Docker image building and ECR pushing operations.
    """

    def __init__(self, aws_region=None, ecr_repo_name="rag-images-container"):
        """
        Initialize Docker ECR Manager.

        Args:
            aws_region (str): AWS region (defaults to AWS_REGION env var or us-east-1)
            ecr_repo_name (str): ECR repository name (default: rag-images-container)
        """
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.ecr_repo_name = ecr_repo_name
        self.ecr_client = boto3.client("ecr", region_name=self.aws_region)
        self.account_id = None
        self.ecr_uri = None
        logger.info(f"DockerECRManager initialized for region: {self.aws_region}")

    def get_ecr_login_credentials(self):
        """
        Get ECR login password and authenticate Docker.

        Returns:
            str: ECR repository URI
        """
        try:
            logger.info("Getting ECR login credentials...")

            # Get authorization token
            response = self.ecr_client.get_authorization_token()
            auth_data = response["authorizationData"][0]

            # Extract registry URL and credentials
            registry_url = auth_data["proxyEndpoint"]
            self.account_id = registry_url.split("//")[1].split(".")[0]
            self.ecr_uri = f"{self.account_id}.dkr.ecr.{self.aws_region}.amazonaws.com/{self.ecr_repo_name}"

            # Get password and login
            import base64

            auth_token = base64.b64decode(auth_data["authorizationToken"]).decode(
                "utf-8"
            )
            username, password = auth_token.split(":")

            # Docker login
            login_command = f"docker login --username {username} --password-stdin {registry_url.split('//')[1]}"
            process = subprocess.Popen(
                login_command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input=password)

            if process.returncode != 0:
                logger.error(f"Docker login failed: {stderr}")
                raise Exception(f"Docker login failed: {stderr}")

            logger.info("✓ Successfully authenticated with ECR")
            logger.info(f"ECR URI: {self.ecr_uri}")
            return self.ecr_uri

        except ClientError as e:
            logger.error(f"AWS ECR authentication error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during ECR login: {e}")
            raise

    def build_docker_image(self, dockerfile_path, image_name, tag="latest"):
        """
        Build a Docker image.

        Args:
            dockerfile_path (str): Path to Dockerfile
            image_name (str): Name for the image
            tag (str): Image tag (default: latest)

        Returns:
            str: Full image name with tag
        """
        try:
            logger.info(f"Building Docker image: {image_name}:{tag}")
            logger.info(f"Using Dockerfile: {dockerfile_path}")

            # Get the directory containing the Dockerfile
            build_context = os.path.dirname(os.path.abspath(dockerfile_path))
            if build_context == "":
                build_context = "."

            # Build command
            build_command = [
                "docker",
                "build",
                "-f",
                dockerfile_path,
                "-t",
                f"{image_name}:{tag}",
                build_context,
            ]

            logger.info(f"Build command: {' '.join(build_command)}")

            # Execute build
            process = subprocess.Popen(
                build_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output
            for line in process.stdout:
                logger.info(line.strip())

            process.wait()

            if process.returncode != 0:
                raise Exception(
                    f"Docker build failed with return code {process.returncode}"
                )

            logger.info(f"✓ Successfully built image: {image_name}:{tag}")
            return f"{image_name}:{tag}"

        except Exception as e:
            logger.error(f"Error building Docker image: {e}")
            raise

    def tag_image(self, source_image, target_image):
        """
        Tag a Docker image.

        Args:
            source_image (str): Source image name with tag
            target_image (str): Target image name with tag
        """
        try:
            logger.info(f"Tagging image: {source_image} -> {target_image}")

            tag_command = ["docker", "tag", source_image, target_image]

            subprocess.run(tag_command, capture_output=True, text=True, check=True)

            logger.info(f"✓ Successfully tagged image: {target_image}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error tagging image: {e.stderr}")
            raise

    def push_image(self, image_name):
        """
        Push a Docker image to ECR.

        Args:
            image_name (str): Full image name with tag to push
        """
        try:
            logger.info(f"Pushing image to ECR: {image_name}")

            push_command = ["docker", "push", image_name]

            process = subprocess.Popen(
                push_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output
            for line in process.stdout:
                logger.info(line.strip())

            process.wait()

            if process.returncode != 0:
                raise Exception(
                    f"Docker push failed with return code {process.returncode}"
                )

            logger.info(f"✓ Successfully pushed image to ECR: {image_name}")

        except Exception as e:
            logger.error(f"Error pushing image to ECR: {e}")
            raise

    def build_and_push_images(self, root_dir=None):
        """
        Build and push both FastAPI and Streamlit images to ECR.

        Args:
            root_dir (str): Root directory of the project (defaults to current directory)

        Returns:
            dict: Contains pushed image URIs
        """
        try:
            # Determine root directory
            if root_dir is None:
                # Assume we're in src/scripts, go up two levels
                root_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

            logger.info("\n" + "=" * 80)
            logger.info("STARTING DOCKER BUILD AND ECR PUSH")
            logger.info("=" * 80 + "\n")
            logger.info(f"Root directory: {root_dir}")





            # Step 1: Authenticate with ECR
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: Authenticating with ECR")
            logger.info("=" * 80)
            self.get_ecr_login_credentials()






            # Step 2: Build FastAPI image
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: Building FastAPI Docker Image")
            logger.info("=" * 80)

            dockerfile_api = os.path.join(root_dir, "Dockerfile.api")
            api_image_name = "rag-fastapi"
            self.build_docker_image(dockerfile_api, api_image_name, "latest")

            # Tag for ECR
            api_ecr_image = f"{self.ecr_uri}:api-latest"
            self.tag_image(f"{api_image_name}:latest", api_ecr_image)





            # Step 3: Build Streamlit image
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Building Streamlit Docker Image")
            logger.info("=" * 80)

            dockerfile_streamlit = os.path.join(root_dir, "Dockerfile.streamlit")
            streamlit_image_name = "rag-streamlit"
            self.build_docker_image(
                dockerfile_streamlit, streamlit_image_name, "latest"
            )

            # Tag for ECR
            streamlit_ecr_image = f"{self.ecr_uri}:streamlit-latest"
            self.tag_image(f"{streamlit_image_name}:latest", streamlit_ecr_image)


            # Step 4: Push FastAPI image to ECR
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: Pushing FastAPI Image to ECR")
            logger.info("=" * 80)
            self.push_image(api_ecr_image)


            # Step 5: Push Streamlit image to ECR
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: Pushing Streamlit Image to ECR")
            logger.info("=" * 80)
            self.push_image(streamlit_ecr_image)

            logger.info("\n" + "=" * 80)
            logger.info("✓ DOCKER BUILD AND ECR PUSH COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80 + "\n")

            result = {
                "api_image": api_ecr_image,
                "streamlit_image": streamlit_ecr_image,
                "ecr_repo": self.ecr_repo_name,
                "region": self.aws_region,
            }

            logger.info(f"API Image URI: {api_ecr_image}")
            logger.info(f"Streamlit Image URI: {streamlit_ecr_image}")

            return result

        except Exception as e:
            logger.error(f"Error in build_and_push_images: {e}")
            logger.error("✗ DOCKER BUILD AND ECR PUSH FAILED")
            raise


def build_and_push_to_ecr(
    ecr_repo_name="rag-images-container", aws_region=None, root_dir=None
):
    """
    Convenience function to build and push Docker images to ECR.

    Args:
        ecr_repo_name (str): ECR repository name
        aws_region (str): AWS region
        root_dir (str): Root directory of the project

    Returns:
        dict: Contains pushed image URIs
    """
    manager = DockerECRManager(aws_region=aws_region, ecr_repo_name=ecr_repo_name)
    return manager.build_and_push_images(root_dir=root_dir)


if __name__ == "__main__":
    """
    Test Docker ECR Manager standalone.
    """
    print("Building and pushing Docker images to ECR...")
    result = build_and_push_to_ecr()
    print("\nCompleted!")
    print(f"Results: {result}")
