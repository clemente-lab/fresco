#!/usr/bin/env python
from __future__ import division

import sys
from glob import glob
from os import makedirs
from os.path import abspath, exists, join
from subprocess import Popen, PIPE
from time import sleep

minerva_job_format = """#!/bin/sh -e
#PBS -A acc_15
#PBS -q small
#PBS -l nodes=1:ppn=%(num_processes)d
#PBS -l walltime=10:00:00
#PBS -l mem=8000mb
#PBS -N fresco-%(job_id)s
#PBS -o %(jobs_dir)s/%(job_id)s.log
#PBS -e %(jobs_dir)s/%(job_id)s.err

cd $PBS_O_WORKDIR
module load scikit-learn/0.14.1
rm -rf %(tmp_out_dir)s
fresco.py --group_map_files %(otu_map_list_fp)s --mapping_file %(map_fp)s --prediction_field %(category)s --start_level %(start_level)d --n_procs %(num_processes)d --model %(model)s --output_dir %(tmp_out_dir)s
mv %(tmp_out_dir)s %(out_dir)s
"""

class ExternalCommandFailedError(Exception):
    pass

def create_dir(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)

def run_command(cmd):
    # Taken from qiime.util.qiime_system_call
    proc = Popen(cmd, shell=True, universal_newlines=True, stdout=PIPE,
                 stderr=PIPE)
    # communicate pulls all stdout/stderr from the PIPEs to 
    # avoid blocking -- don't remove this line!
    stdout, stderr = proc.communicate()
    return_value = proc.returncode

    if return_value != 0:
        raise ExternalCommandFailedError("The command '%s' failed with exit "
                                         "status %d.\n\nStdout:\n\n%s\n\n"
                                         "Stderr:\n\n%s\n" % (cmd,
                                                              return_value,
                                                              stdout, stderr))


if len(sys.argv) != 4:
    sys.stderr.write("Usage: submit_fresco_jobs.py <input directory> "
                     "<output directory> <start level>\n")
    sys.exit(1)

in_dir, out_dir, start_level = sys.argv[1:]
start_level = int(start_level)

studies = {
    'study_451': ['DIET', 'TREATMENT']
}

models = ['lr', 'rf', 'sv']

jobs_dir = join(out_dir, 'jobs')
map_filename = 'map.txt'
otu_map_dir_wc = 'ucrc_*'
otu_map_dir_nested_dirname = 'uclust_ref_picked_otus'
otu_map_filename_wc = '*_seqs_otus.txt'
otu_map_list_filename = 'otu_maps.txt'
num_processes = 32
sleep_time = 1

create_dir(out_dir)
create_dir(jobs_dir)

for study in studies:
    study_dir = join(in_dir, study)
    map_fp = join(study_dir, map_filename)

    out_study_dir = join(out_dir, study)
    create_dir(out_study_dir)

    otu_map_fps = []
    for otu_map_dir in sorted(glob(join(study_dir, otu_map_dir_wc))):
        full_otu_map_dir = join(otu_map_dir, otu_map_dir_nested_dirname)
        otu_map_fp = glob(join(full_otu_map_dir, otu_map_filename_wc))

        if len(otu_map_fp) != 1:
            raise ValueError("Couldn't find exactly one OTU map in %s." %
                             full_otu_map_dir)

        otu_map_fps.append(abspath(otu_map_fp[0]))

    if start_level < 0 or start_level >= len(otu_map_fps):
        sys.stderr.write("Invalid start level: %d\n" % start_level)
        sys.exit(1)

    otu_map_list_fp = join(out_study_dir, otu_map_list_filename)

    if not exists(otu_map_list_fp):
        with open(otu_map_list_fp, 'w') as otu_map_list_f:
            for otu_map_fp in otu_map_fps:
                otu_map_list_f.write(otu_map_fp + '\n')

    for category in studies[study]:
        out_cat_dir = join(out_study_dir, category)
        create_dir(out_cat_dir)

        out_start_level_dir = join(out_cat_dir, '%d' % start_level)
        create_dir(out_start_level_dir)

        for model in models:
            out_model_dir = join(out_start_level_dir, model)
            tmp_out_model_dir = out_model_dir + '.tmp'
            job_id = '%s-%s-%d-%s' % (study, category, start_level, model)

            if not exists(out_model_dir):
                job_script_fp = join(jobs_dir, '%s-job.sh' % job_id)

                with open(job_script_fp, 'w') as job_script_f:
                    formatted_job_str = minerva_job_format % {
                            'num_processes': num_processes,
                            'jobs_dir': abspath(jobs_dir),
                            'job_id': job_id,
                            'otu_map_list_fp': abspath(otu_map_list_fp),
                            'map_fp': abspath(map_fp),
                            'category': category,
                            'start_level': start_level,
                            'model': model,
                            'tmp_out_dir': abspath(tmp_out_model_dir),
                            'out_dir': abspath(out_model_dir)
                    }

                    job_script_f.write(formatted_job_str)

                cmd = 'qsub %s' % job_script_fp
                print 'Running: %s' % cmd
                run_command(cmd)
                sleep(sleep_time)
