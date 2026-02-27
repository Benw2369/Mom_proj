#!/bin/bash
set -e

DB_DIR="/var/app/data"
S3_BUCKET="elasticbeanstalk-eu-north-1-197034345303"

mkdir -p "$DB_DIR"

aws s3 cp "s3://$S3_BUCKET/DJI_data.db" "$DB_DIR/DJI_data.db"
aws s3 cp "s3://$S3_BUCKET/FTSE100_data.db" "$DB_DIR/FTSE100_data.db"
