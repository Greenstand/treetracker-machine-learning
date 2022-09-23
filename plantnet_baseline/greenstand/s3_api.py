# Imports
import boto3
from pathlib import Path

# Functions
def get_missing_local_files(bucket, prefix, local_dir, species_limit=None):
    # Note any corrupted files in S3 to ignore
    blacklist = ['2021.10.14.13.04.09_8.418876039795578_-13.165308712050319_504fb6d8-cbf3-4196-a21f-29cf027dc712_IMG_20211014_125152_8666198052150270767.jpg']
    
    print(f"Checking missing local files based on bucket: {bucket} at prefix {prefix}...")
    # Get S3 objects
    s3 = S3()
    keys = s3.list_objects(bucket, prefix, subprefix_limit=species_limit)
    
    # Get local objects
    missing_keys = []
    for key in keys:
        fname = f"{local_dir}/{key}"
        path = Path(fname)
        if not path.is_file() and not path.is_dir() and key not in blacklist:
            missing_keys.append(key)
            
    if len(missing_keys) > 0:
        print(f"Found {len(missing_keys)} missing items locally. Downloading...")
        download_objects_locally(bucket, prefix, local_dir, missing_keys)
        return True
    else:
        print("No missing items locally.")
        return False
    
    
def download_objects_locally(bucket, prefix, local_dir, keys=None):
    s3 = S3()
    if keys is None:
        keys = s3.list_objects(bucket, prefix)
    print(f"Found {len(keys)} objects...")
    for i in range(len(keys)):
        key = keys[i]
        
        if i%100==0:
            print(f"Writing file {i}/{len(keys)}")
        
        # Create directory if not exist
        fname = f"{local_dir}/{key}"
        output_file = Path(fname)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        if '.' in key:
            output_file.write_bytes(s3.get_object(bucket, key))
    

# Classes
class S3:
    def __init__(self, client=None):
        if client is None: 
            client = boto3.client('s3')
        self.client = client
        
    def get_object(self, bucket, key):
        return self.client.get_object(Bucket=bucket, Key=key)['Body'].read()
    
    def list_objects(self, bucket, key_prefix=None, limit=None, subprefix_limit=None, delimiter=None):
        truncated, marker, keys, kwargs = (True, None, [], {'Bucket': bucket})
        search_keys = ('Contents','Key')
        
        if key_prefix:
            kwargs['Prefix'] = key_prefix
            
        if delimiter:
            kwargs['Delimiter'] = delimiter
            search_keys = ('CommonPrefixes','Prefix')
            
        if subprefix_limit: # For each 'folder' under the prefix, list at most x num objects
            # Get each subprefix
            subprefixes = self.list_objects(bucket, key_prefix=key_prefix, subprefix_limit=None, delimiter='/')
            # Get all keys for each subprefix with limit specified
            keys = []
            for subprefix in subprefixes:
                keys += self.list_objects(bucket, subprefix, limit=subprefix_limit)
            return keys

        while truncated:
            if marker:
                kwargs['Marker'] = marker
            response = self.client.list_objects(**kwargs)
            if search_keys[0] not in response:
                raise Exception("Nothing found under that key prefix")
            for item in response[search_keys[0]]:
                keys.append(item[search_keys[1]])
            truncated = response['IsTruncated']
            # Leave loop if exceeded limit
            if limit and len(keys) >= limit:
                keys = keys[:limit]
                break
            marker = response.get('NextMarker', item[search_keys[1]])

        return keys

    def put_object(self, bucket, key, contents, check_encryption=False):
        if not isinstance(contents, bytes):
            contents = contents.encode('utf-8')
        
        # Gen params
        kwargs = {'Bucket': bucket, 'Key': key, 'Body': contents}
        if check_encryption:
            kwargs.update(self.__check_encryption(bucket))
        
            
        self.client.put_object(**kwargs)
        return True
        
        
    def delete_object(self, bucket, key):
        self.client.delete_object(Bucket=bucket, Key=key)
        return True
        
        
    def __check_encryption(self, bucket):
        # Check encryption settings
        response = self.client.get_bucket_encryption(Bucket=bucket)
        sse = response.get('ServerSideEncryptionConfiguration', {})
        rule = sse.get('Rules', [{}])[0]
        apply = rule.get('ApplyServerSideEncryptionByDefault', {})
        algorithm = apply.get('SSEAlgorithm', None)
        kms = apply.get('KMSMasterKeyID', None)
        
        # Set args
        args = {}
        if algorithm:
            args['ServerSideEncryption'] = algorithm
            if kms:
                args['SSEKMSId'] = kms
        
        return args
