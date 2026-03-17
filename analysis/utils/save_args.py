import os
import json

def save_args(args, path):
    os.makedirs(path, exist_ok=True)
    args_file = os.path.join(path, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved args to {args_file}")