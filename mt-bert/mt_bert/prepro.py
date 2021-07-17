"""
Preprocessing BLUE dataset.

Usage:
    blue_prepro [options] --root_dir=<dir> --task_def=<file> --datasets=<str>

Options:
    --overwrite
"""
import os

import docopt
from data_utils.log_wrapper import create_logger
from blue_exp_def import BlueTaskDefs
from blue_utils import load_sts, load_mednli, \
    load_relation, load_ner, dump_rows, load_label_keywords
from data_utils.task_def import TaskType

def main(args):
    root = args['--root_dir']
    assert os.path.exists(root)

    log_file = os.path.join(root, 'blue_prepro.log')
    logger = create_logger(__name__, to_disk=True, log_file=log_file)

    task_defs = BlueTaskDefs(args['--task_def'])

    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    if args['--datasets'] == 'all':
        tasks = task_defs.tasks
    else:
        tasks = args['--datasets'].split(',')
    for task in tasks:
        logger.info("Task %s" % task)
        if task not in task_defs.task_def_dic:
            raise KeyError('%s: Cannot process this task' % task)

        if task in ['clinicalsts', 'biosses']:
            load = load_sts
        elif task == 'mednli':
            load = load_mednli
        elif task in ('chemprot', 'ddi', 'cdr', 'ppim', 'chr-document', 'chr-document-3', 'cdr-document',
                      'cdr-test', 'ppim-2', 'ade'):
            load = load_relation
        elif task in ('bc5cdr-disease', 'bc5cdr-chemical', 'shareclefe'):
            load = load_ner
        else:
            raise KeyError('%s: Cannot process this task' % task)

        data_format = task_defs.data_format_map[task]
        split_names = task_defs.split_names_map[task]
        task_type = task_defs.task_type_map[task]
        n_class = task_defs.n_class_map[task]
        for split_name in split_names:
            fin = os.path.join(root, f'{task}/{split_name}.tsv')
            fout = os.path.join(canonical_data_root, f'{task}_{split_name}.tsv')
            if os.path.exists(fout) and not args['--overwrite']:
                logger.warning('%s: Not overwrite %s: %s', task, split_name, fout)
                continue
            data = load(fin)
            logger.info('%s: Loaded %s %s samples', task, len(data), split_name)
            dump_rows(data, fout, data_format)
        if task_type == TaskType.REWithLabelEmbedding or task_type == TaskType.RE:
            keywords_file = os.path.join(root, f'{task}/keywords.tsv')
            out_path = os.path.join(canonical_data_root, f'{task}_keywords.json')
            assert os.path.exists(keywords_file)
            load_label_keywords(keywords_file, out_path, n_class)
            logger.info('%s: Loaded label embedding', task)
        logger.info('%s: Done', task)


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(args)
