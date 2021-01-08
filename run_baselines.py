import os
from functools import partial
import pandas as pd

from baselines.brightness import get_brightness_factor
from baselines.domain_mapping import get_z_score_encoder, get_domain_mapping_method
from baselines.evaluation import evaluate_roc, evaluate_threshold
from baselines.rule_based import run_rule_based_baseline
from baselines.trad_ml import train_trad_ml_baseline, evaluate_ml_baseline
from baselines.utils import export_selected_records, extract_y_true
from baselines.zscores import compute_EIZ_from_scratch

from utils.experiment_utils import get_default_set_spec_dict
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a full evaluation run of different baselines.')

    parser.add_argument('source_device', help='The source device, e.g. ESAOTE_6100.')
    parser.add_argument('target_device', help='The target device, e.g. Philips_iU22.')
    parser.add_argument('--export_path', help='The path to export results to.', default='roc_analysis')

    parser.add_argument('--evaluate_set',default='test', help='The section to perform the evaluation on.',
                        choices=['test', 'val', 'im_muscle_chart'])

    parser.add_argument('--no_rule_based',action='store_true', help='Whether to skip the rule-based method.')
    parser.add_argument('--no_ml',action='store_true', help='Whether to skip the ML-based method.')

    args = parser.parse_args()

    set_spec_dict = get_default_set_spec_dict()

    source = args.source_device
    target = args.target_device
    evaluate_on = args.evaluate_set

    eval_set_name = target + '_' + evaluate_on

    # get the correct sets
    target_set = set_spec_dict[eval_set_name]
    source_set = set_spec_dict[source + '_' + 'train']

    export_path = os.path.join(args.export_path, eval_set_name)
    os.makedirs(export_path, exist_ok=True)
    # extract ground truth
    y_true, patients = extract_y_true(eval_set_name)
    # export
    pd.DataFrame(y_true, columns=['true']).to_csv(os.path.join(export_path, 'y_true.csv'), index=False, header=True)
    patients.to_pickle(os.path.join(export_path, 'patients.pkl'))
    records = export_selected_records(eval_set_name)
    records.to_pickle(os.path.join(export_path, 'records.pkl'))
    # path for predictions
    proba_path = os.path.join(export_path, 'proba')
    indomain_path = os.path.join(proba_path, 'in_domain')
    outdomain_path = os.path.join(proba_path, 'out_domain')
    os.makedirs(indomain_path, exist_ok=True)
    os.makedirs(outdomain_path, exist_ok=True)
    z_score_encoder = get_z_score_encoder(source)

    # domain mapping methods to be experimented with
    ei_extraction_methods_full = ['original', 'recompute', 'regression', 'brightness']#, 'mapped_images']
    ei_extraction_methods_no_encoder = ['original']

    # if a z-score encoder exists for the source domain, we can do a lot more
    if z_score_encoder:
        ei_extraction_methods = ei_extraction_methods_full
    else:
        ei_extraction_methods = ei_extraction_methods_no_encoder


    # Rule-based method
    if not args.no_rule_based:
        # rule-based baseline
        for method_name in ei_extraction_methods:
            ei_extraction_method = get_domain_mapping_method(method_name, source_set, target_set, z_score_encoder)
            y_proba = run_rule_based_baseline(eval_set_name, ei_extraction_method)
            exp_name = 'rulebased' + '_' + method_name
            base_path = indomain_path if method_name == 'original' else outdomain_path
            csv_path = os.path.join(base_path, (exp_name + '.csv'))
            pd.DataFrame(y_proba, columns=['pred']).to_csv(csv_path, index=False, header=True)
            evaluate_roc(y_true, y_proba, exp_name)

    # Traditional ML
    if not args.no_ml:

        eiz_options = [False, True]
        demographic_features = False

        for use_eiz in eiz_options:
            print('Training model, please press enter once the features are shown to confirm.')
            # train on source and eval on original val set
            result_dict = train_trad_ml_baseline(source + '_train', source + '_' + 'val', use_eiz=use_eiz,
                                                 demographic_features=demographic_features)

            source_eval_set = source + '_' + evaluate_on
            # evaluate on source
            y_proba = evaluate_ml_baseline(result_dict['model_path'], source_eval_set, use_eiz=use_eiz,
                                           demographic_features=demographic_features)

            exp_name = result_dict['model_path'] + '_' + 'original'
            y_true_source, patients_source = extract_y_true(source_eval_set)
            evaluate_roc(y_true_source, y_proba, exp_name)

            export_path = os.path.join(args.export_path, source_eval_set, 'proba', 'in_domain')
            os.makedirs(export_path, exist_ok=True)
            csv_path = os.path.join(export_path, (exp_name + '.csv'))
            pd.DataFrame(y_proba, columns=['pred']).to_csv(csv_path, index=False, header=True)

            # evaluate on target
            if use_eiz:
                z_score_encoder = get_z_score_encoder(source)
            else:
                z_score_encoder = None

            if use_eiz and not z_score_encoder:
                ei_extraction_methods = ei_extraction_methods_no_encoder
            else:
                ei_extraction_methods = ei_extraction_methods_no_encoder

            for method_name in ei_extraction_methods:
                ei_extraction_method = get_domain_mapping_method(method_name, source_set, target_set, z_score_encoder)

                y_proba = evaluate_ml_baseline(result_dict['model_path'], eval_set_name, use_eiz=use_eiz,
                                               demographic_features=demographic_features,
                                               ei_extraction_method=ei_extraction_method)
                exp_name = result_dict['model_path'] + '_' + method_name
                # export
                csv_path = os.path.join(proba_path, 'out_domain', (exp_name + '.csv'))
                pd.DataFrame(y_proba, columns=['pred']).to_csv(csv_path, index=False, header=True)

                # performance
                evaluate_roc(y_true, y_proba, exp_name)

                # performance at best threshold during training
                previous_threshold = result_dict['best_threshold']
                if previous_threshold:
                    print('Performance at best previous threshold')
                    print(evaluate_threshold(y_true, y_proba, previous_threshold))

            # separate evaluation logic for GE is necessary
            # can only do GE evaluation if no z-scores required, as we lack required demographic attributes
            # for the GE patients
            if not use_eiz:
               ge_set_name = 'GE_Logiq_E_im_muscle_chart'
               factor = get_brightness_factor(source_set.device, 'GE_Logiq_E')

               spec = set_spec_dict[ge_set_name]
               ei_extraction_method = partial(compute_EIZ_from_scratch, z_score_encoder=z_score_encoder,
                                              root_path=spec.img_root_path,
                                              factor=factor)

               y_proba = evaluate_ml_baseline(result_dict['model_path'], ge_set_name, use_eiz=False,
                                              demographic_features=demographic_features,
                                              ei_extraction_method=ei_extraction_method)

               exp_name = result_dict['model_path']
               # export
               export_path_ge = os.path.join(args.export_path, ge_set_name, 'proba')
               os.makedirs(export_path_ge, exist_ok=True)
               csv_path = os.path.join(export_path_ge, 'out_domain', (exp_name + '_brightness' + '.csv'))
               # performance
               pd.DataFrame(y_proba, columns=['pred']).to_csv(csv_path, index=False, header=True)
