from pathhack import pkg_path
import argparse
from src.utils import load_ilstm_model_by_arguments

"""
# Added mifwlstm-parallelization branch. 
# Wrote hri_intention_application_interface.py. 
# Add ilstm loading functions in utils.py. 
# Move IntentionLstm class to ilstm/model.py.
#  Move 211106_115545 checkpoint to 211106_slow.pt. 
# Wrote load_ilstm.py for test.
"""

def arg_parse():

    parser = argparse.ArgumentParser()
    # Model Options
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--obs_seq_len', default=20, type=int)
    parser.add_argument('--pred_seq_len', default=20, type=int)
    parser.add_argument('--motion_dim', default=3, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    # Scene Options
    parser.add_argument('--num_intentions', default=2, type=int)
    return parser.parse_args()

args = arg_parse()
model = load_ilstm_model_by_arguments(args, pkg_path, args.device)
print(model)