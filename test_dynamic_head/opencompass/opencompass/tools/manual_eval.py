#!/usr/bin/env python
"""Lightweight evaluator for already generated predictions.

This script scans an OpenCompass work_dir/predictions folder, rebuilds the
original datasets defined in the config, and then reuses the built-in
OpenICLEvalTask utilities to score every finished (non-tmp) prediction file.

Example:
    python tools/manual_eval.py \
        --pred-dir outputs/llada_dynamic_head_ruler/20251109_113213/predictions
"""

from __future__ import annotations

import argparse
import copy
import os.path as osp
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mmengine
from mmengine.config import Config, ConfigDict

from opencompass.tasks.openicl_eval import OpenICLEvalTask
from opencompass.utils import (dataset_abbr_from_cfg, get_infer_output_path,
                               get_logger, model_abbr_from_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Score finished prediction files without rerunning eval')
    parser.add_argument('--pred-dir', required=True,
                        help='Path to the predictions directory produced by'
                        ' OpenCompass (…/<time_str>/predictions).')
    parser.add_argument('--config', default=None,
                        help='Path to the experiment config. If not set, the'
                        ' script will pick the latest file in the sibling'
                        ' configs/ folder of the work_dir.')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Optional list of model abbreviations to score.')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Optional list of dataset abbreviations to score.')
    parser.add_argument('--summary-dir', default=None,
                        help='Directory to dump the aggregated summary. '
                        'Defaults to <work_dir>/summary.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip evaluation if the corresponding results '
                        'JSON already exists.')
    parser.add_argument('--no-save-results', action='store_true',
                        help='Do not dump per-dataset scores under '
                        '<work_dir>/results. Handy when you only need the '
                        'aggregate summary.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print extra debugging information.')
    return parser.parse_args()


def auto_find_config(work_dir: Path) -> Path:
    cfg_dir = work_dir / 'configs'
    if not cfg_dir.exists():
        raise FileNotFoundError(f'Config directory not found: {cfg_dir}')
    candidates = sorted(cfg_dir.glob('*.py'))
    if not candidates:
        raise FileNotFoundError(f'No config *.py files found in {cfg_dir}')
    return candidates[-1]


def build_task_skeleton(reference_cfg: Config, work_dir: Path) -> OpenICLEvalTask:
    if not reference_cfg['models']:
        raise ValueError('Config does not contain any model definitions')
    if not reference_cfg['datasets']:
        raise ValueError('Config does not contain any dataset definitions')

    skeleton_cfg = ConfigDict(
        models=[copy.deepcopy(reference_cfg['models'][0])],
        datasets=[[copy.deepcopy(reference_cfg['datasets'][0])]],
        work_dir=str(work_dir),
        eval=dict(runner=dict(task=dict(dump_details=False,
                                        cal_extract_rate=False)))
    )
    return OpenICLEvalTask(skeleton_cfg)


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.pred_dir).expanduser().resolve()
    if not predictions_dir.exists():
        raise FileNotFoundError(f'Prediction dir not found: {predictions_dir}')
    work_dir = predictions_dir.parent
    cfg_path = Path(args.config).expanduser().resolve() if args.config else \
        auto_find_config(work_dir)

    cfg = Config.fromfile(str(cfg_path), format_python_code=False)
    cfg['work_dir'] = str(work_dir)

    logger = get_logger('manual_eval')
    logger.info('Using config %s', cfg_path)
    logger.info('Scanning predictions in %s', predictions_dir)

    helper = build_task_skeleton(cfg, work_dir)
    helper.logger = logger

    dataset_cache: Dict[str, object] = {}
    summary: Dict[str, Dict[str, dict]] = defaultdict(dict)
    total = 0
    evaluated = 0

    for model_cfg in cfg['models']:
        model_abbr = model_abbr_from_cfg(model_cfg)
        if args.models and model_abbr not in args.models:
            continue

        for dataset_cfg in cfg['datasets']:
            dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
            if args.datasets and dataset_abbr not in args.datasets:
                continue

            total += 1
            pred_file = get_infer_output_path(model_cfg, dataset_cfg,
                                              str(predictions_dir))
            if not osp.exists(pred_file):
                if args.verbose:
                    logger.info('Skip %s/%s (prediction missing)',
                                model_abbr, dataset_abbr)
                continue

            result_file = get_infer_output_path(
                model_cfg, dataset_cfg, osp.join(work_dir, 'results'))
            if args.skip_existing and osp.exists(result_file):
                if args.verbose:
                    logger.info('Skip %s/%s (result exists)',
                                model_abbr, dataset_abbr)
                continue

            helper.model_cfg = copy.deepcopy(model_cfg)
            helper.dataset_cfg = copy.deepcopy(dataset_cfg)
            helper.eval_cfg = copy.deepcopy(dataset_cfg.get('eval_cfg', {}))
            reader_cfg = dataset_cfg.get('reader_cfg', {})
            helper.output_column = reader_cfg.get('output_column')
            helper.dump_details = False
            helper.cal_extract_rate = False

            if dataset_abbr not in dataset_cache:
                dataset_cache[dataset_abbr] = helper._load_and_preprocess_test_data()
            test_set = dataset_cache[dataset_abbr]

            try:
                pred_dicts, pred_strs = helper._load_predictions()
            except FileNotFoundError:
                if args.verbose:
                    logger.info('Skip %s/%s (prediction not finalized)',
                                model_abbr, dataset_abbr)
                continue

            pred_strs = helper._process_predictions(pred_strs)
            result = helper._evaluate_predictions(pred_strs, test_set,
                                                  pred_dicts)
            if not args.no_save_results:
                helper._save_results(result)
            summary[model_abbr][dataset_abbr] = {
                k: v
                for k, v in result.items()
                if k != 'details'
            }
            evaluated += 1
            logger.info('Evaluated %s/%s -> %s', model_abbr, dataset_abbr,
                        summary[model_abbr][dataset_abbr])

    if evaluated == 0:
        logger.warning('No finished predictions found under %s', predictions_dir)
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_dir = Path(args.summary_dir or (work_dir / 'summary'))
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f'manual_summary_{timestamp}.json'
    mmengine.dump(summary, summary_path, indent=2)

    logger.info('Scored %d/%d model-dataset pairs. Summary saved to %s',
                evaluated, total, summary_path)


if __name__ == '__main__':
    main()
