"""Check AWS Braket device availability."""
from dotenv import load_dotenv
load_dotenv()

import boto3
from braket.aws import AwsDevice, AwsSession

print("AWS BRAKET AVAILABLE DEVICES")
print("="*90)

# Create session with explicit region
boto_session = boto3.Session(region_name='us-east-1')
aws_session = AwsSession(boto_session=boto_session)

# Get all devices
devices = AwsDevice.get_devices(aws_session=aws_session)

# Categorize
qpus = [d for d in devices if d.type.name == 'QPU']
simulators = [d for d in devices if d.type.name == 'SIMULATOR']

print("\n--- QUANTUM PROCESSORS (QPUs) ---")
print(f"{'Name':35} | {'Provider':15} | {'Status':12} | Region")
print("-"*90)
for d in qpus:
    region = d.arn.split(':')[3] if ':' in d.arn else 'N/A'
    status_icon = "[OK]" if d.status == "ONLINE" else "[--]"
    print(f"{status_icon} {d.name:32} | {d.provider_name:15} | {d.status:12} | {region}")

print("\n--- SIMULATORS ---")
print(f"{'Name':35} | {'Provider':15} | {'Status':12} | Region")
print("-"*90)
for d in simulators:
    region = d.arn.split(':')[3] if ':' in d.arn else 'N/A'
    status_icon = "[OK]" if d.status == "ONLINE" else "[--]"
    print(f"{status_icon} {d.name:32} | {d.provider_name:15} | {d.status:12} | {region}")

print("\n" + "="*90)
print("ONLINE QPUs READY FOR SUBMISSION:")
print("="*90)
online_qpus = [d for d in qpus if d.status == "ONLINE"]
if online_qpus:
    for d in online_qpus:
        print(f"\n  * {d.name} ({d.provider_name})")
        print(f"    ARN: {d.arn}")
        try:
            if hasattr(d.properties, 'paradigm'):
                print(f"    Qubits: {d.properties.paradigm.qubitCount}")
            elif hasattr(d.properties, 'provider'):
                print(f"    Provider: {d.properties.provider}")
        except:
            pass
else:
    print("  No QPUs currently online")

print("\n" + "="*90)
