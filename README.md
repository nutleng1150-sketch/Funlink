# Funlink
The linker between annotation result from fungal genome annotation tools (maker or funannotate) and other databases
using SQL hierrachical database system

usage:
--protein_fasta: input your reference protein file (should be from UniProt in this version)
--output_db: input your path to BLASTP database
--query_fasta: input your path to query protein file (should be from funannotate pipeline)
--annotation_file: input your path to annotation file (should be from funannotate pipelie)

# Funml
Preprocessing data module and incremental machine learning module using SGDClasasifier to generate the model for prediction of specific condition using the stored data in database
usage:
--db_bz2_file: input your database file
--output_dir input your output_dir
--metadata_file: input your metadata
--model_output: input where to stored your model


# Funvisual
Predict and make the visualization of the parameter for testing results
