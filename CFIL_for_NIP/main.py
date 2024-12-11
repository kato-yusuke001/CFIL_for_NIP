import argparse

from agent import CFIL_ABN, CoarseToFineImitation
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_epochs", "-e", type=int, default=10000)
    parser.add_argument("--collection_num", "-n", type=int, default=100)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--no_last_inch", action="store_false")
    parser.add_argument("--log_folder", "-f", type=str, default="")
    parser.add_argument("--use_previous_trajectory", "-u", action="store_true")
    parser.add_argument("--abn", action="store_true")
    args = parser.parse_args()
    
    if args.abn:
        cfil = CFIL_ABN(log_dir=args.log_folder)
    else:
        cfil = CoarseToFineImitation()

    if args.load:
        cfil.load_bottleneck(args.log_folder)
        cfil.load_trajectory(args.log_folder)
        cfil.load_memory(args.log_folder)    
    else:
        if args.use_previous_trajectory:
            cfil.use_previous_demo_traj()
        else:
            cfil.collect_demo_traj()

        for i in range(3):
            cfil.collect_approach_traj(num=args.collection_num, last_inch=args.no_last_inch)
            utils.wait_press_key()
            cfil.use_previous_demo_traj()
        cfil.save_memory()
    
    
    cfil.train(train_epochs=args.train_epochs)
