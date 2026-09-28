"""Deploy BrandBrain landing page to AWS S3 static website hosting."""

import mimetypes
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"


def deploy_to_s3(bucket_name: str | None = None) -> str:
    region = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
    s3 = boto3.client("s3", region_name=region)
    sts = boto3.client("sts", region_name=region)

    if not bucket_name:
        account_id = sts.get_caller_identity()["Account"]
        bucket_name = f"brandbrain-mcp-{account_id}"

    print(f"Deploying landing page to S3 bucket: {bucket_name} in {region}...")

    # Create bucket if it does not exist
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"Created bucket '{bucket_name}'.")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"Bucket '{bucket_name}' already exists. Updating contents...")
        else:
            raise

    # Configure static website hosting
    s3.put_bucket_website(
        Bucket=bucket_name,
        WebsiteConfiguration={
            "IndexDocument": {"Suffix": "index.html"},
            "ErrorDocument": {"Key": "index.html"},
        },
    )

    # Disable Block Public Access for static website hosting
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": False,
            "IgnorePublicAcls": False,
            "BlockPublicPolicy": False,
            "RestrictPublicBuckets": False,
        },
    )

    # Attach public read policy
    policy = f"""{{
        "Version": "2012-10-17",
        "Statement": [
            {{
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::{bucket_name}/*"
            }}
        ]
    }}"""
    s3.put_bucket_policy(Bucket=bucket_name, Policy=policy)

    # Upload files
    for file_path in PUBLIC_DIR.glob("**/*"):
        if file_path.is_file():
            key = str(file_path.relative_to(PUBLIC_DIR)).replace("\\", "/")
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"

            with open(file_path, "rb") as f:
                s3.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=f,
                    ContentType=mime_type,
                )
            print(f"Uploaded: {key} ({mime_type})")

    website_url = f"http://{bucket_name}.s3-website.{region}.amazonaws.com"
    print("\nDeployment successful!")
    print(f"Public URL: {website_url}\n")
    return website_url


if __name__ == "__main__":
    b_name = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        deploy_to_s3(b_name)
    except Exception as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        sys.exit(1)
