import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def download_model_from_s3(bucket_name: str, s3_key: str, local_path: str, logger):
    """
    Downloads the model file from S3 to the specified local path.
    Checks os.path.exists to only download ONCE per container startup.
    """
    if os.path.exists(local_path):
        logger.info(f"Model already found in temp storage at {local_path}. Skipping S3 sync.")
        return

    logger.info(f"Fresh startup detected. Downloading model from S3://{bucket_name}/{s3_key} to {local_path}...")

    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info(f"Model successfully cached in temporary storage: {local_path}")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Ensure your EC2 instance has an IAM role attached.")
        raise RuntimeError("AWS credentials missing.")
    except ClientError as e:
        logger.error(f"Failed to download model from S3: {e}")
        raise e