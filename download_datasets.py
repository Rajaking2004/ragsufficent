"""
Script to download HotPotQA and Musique datasets.
"""

from datasets import load_dataset
import os

# Create cache directory
cache_dir = "./data_cache"
os.makedirs(cache_dir, exist_ok=True)

print("=" * 60)
print("DOWNLOADING DATASETS")
print("=" * 60)

# Download HotPotQA
print("\n[1/2] Downloading HotPotQA (distractor) - ~600 MB...")
print("This may take a few minutes...")
try:
    hotpotqa = load_dataset("hotpot_qa", "distractor", cache_dir=cache_dir)
    print(f"✅ HotPotQA downloaded successfully!")
    print(f"   Train: {len(hotpotqa['train']):,} samples")
    print(f"   Validation: {len(hotpotqa['validation']):,} samples")
    
    # Show sample
    sample = hotpotqa["validation"][0]
    print(f"\n   Sample question: {sample['question'][:80]}...")
    print(f"   Sample answer: {sample['answer']}")
except Exception as e:
    print(f"❌ Error downloading HotPotQA: {e}")

# Download Musique
print("\n[2/2] Downloading Musique - ~150 MB...")
print("This may take a few minutes...")
try:
    musique = load_dataset("dgslibiern/MuSiQue", cache_dir=cache_dir)
    print(f"✅ Musique downloaded successfully!")
    for split in musique.keys():
        print(f"   {split}: {len(musique[split]):,} samples")
    
    # Show sample
    sample = musique[list(musique.keys())[0]][0]
    print(f"\n   Sample question: {sample['question'][:80]}...")
except Exception as e:
    print(f"❌ Error downloading Musique: {e}")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
print(f"Datasets cached in: {os.path.abspath(cache_dir)}")
