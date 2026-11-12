#!/usr/bin/env python


import subprocess
import argparse
import os

print("""
==========================================================================================
                                    FUNLINK V.2
==========================================================================================


Developed by Nattapong Langsiri, Ph.D.

usage:
--protein_fasta: input your reference protein file (should be from UniProt in this version)
--output_db: input your path to BLASTP database
--query_fasta: input your path to query protein file (should be from funannotate pipeline)
--annotation_file: input your path to annotation file (should be from funannotate pipelie)


============================================================================================
""")

program_path = input("Please enter the directory to funlink: ")  
immediate_path = os.path.join(program_path, "immediate_output")
final_path = os.path.join(program_path, "final_output")

os.makedirs(immediate_path, exist_ok=True)  
print(f"Directory created: {immediate_path}")
os.makedirs(final_path, exist_ok=True)  
print(f"Directory created: {final_path}")

# Set up argument parser
parser = argparse.ArgumentParser(description="Protein Analysis Pipeline with ML", 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--protein_fasta", type=str, required=True, help="Path to reference protein FASTA file")
parser.add_argument("--output_db", type=str, required=True, help="Path to output BLAST database")
parser.add_argument("--query_fasta", type=str, required=True, help="Path to query protein FASTA file")
parser.add_argument("--annotation_file", type=str, required=True, help="Path to annotation file")

# Parse arguments
args = parser.parse_args()

# 1. Create BLAST database and run search (with added annotation file)
uniq_id_command = [
    "python", os.path.join(program_path, "uniq_id.py"),
    "--protein_fasta", args.protein_fasta,
    "--output_db", args.output_db,
    "--query_fasta", args.query_fasta, 
    "--blast_output", os.path.join(immediate_path, "blast_results.tsv"),
    "--merged_output", os.path.join(immediate_path, "merged_results.tsv"),
    "--annotation_file", args.annotation_file,
    "--uniprot_output", os.path.join(immediate_path, "uniprot_ids.txt")
]

# Run the command
subprocess.run(uniq_id_command, check=True)

# 2. Link to other db using API
fetch_command = [
    "python", os.path.join(program_path, "fetch.py"), os.path.join(immediate_path, "uniprot_ids.txt"), os.path.join(immediate_path, "uniprot_map.txt"),

]

# Run the command
subprocess.run(fetch_command, check=True)

# 3. Merge the linked output with previous annnotation
merge_command = [
    "python", os.path.join(program_path, "merge.py"), os.path.join(immediate_path, "merged_results.tsv"), os.path.join(immediate_path, "uniprot_map.txt"), os.path.join(immediate_path, "draft_db.tsv"),

]   

# Run the command
subprocess.run(merge_command, check=True)

print(f"Draft database is established")
species = input("Please enter the species name for indexing the database: ")
condition = input(" Please enter the condition for indexing the database: ")
isolate = input("Please enter the ioslate name to be the key for searching database (should use the unique name without space): ")
 
# 4. Storing the database
storage_command = [
    "python", os.path.join(program_path, "storage.py"), 
    "--species", species,  # Use the variable from input()
    "--condition", condition,  # Use the variable from input()
    "--isolate", isolate,  # Use the variable from input()
    "--db_dir", final_path,  # Use the final_path variable
    os.path.join(immediate_path, "draft_db.tsv"),
]

# Run the command
subprocess.run(storage_command, check=True)


