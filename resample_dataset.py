import os, torch, torchaudio, argparse, re, tqdm


parser = argparse.ArgumentParser()
parser.add_argument('input_folder', type=str)
parser.add_argument('output_folder', type=str)
parser.add_argument('sample_rate', type=int)
parser.add_argument('--target_ext', type=str, default=".wav")
args = parser.parse_args()

VALID_EXTS = [".wav", ".aif", ".aiff", ".mp3", ".m4a"]

valid_files = []
for root, directory, files in os.walk(args.input_folder):
    files_tmp = list(filter(lambda f: os.path.splitext(f)[1].lower() in VALID_EXTS, files))
    valid_files.extend([root + "/" + f for f in files_tmp])
    

for current_file in tqdm.tqdm(valid_files, total=len(valid_files), desc="slicing..."):
    target_file, ext = os.path.splitext(re.sub(args.input_folder, args.output_folder, current_file))
    dirpath = os.path.dirname(target_file)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    x, sr = torchaudio.load(current_file)
    if sr != args.sample_rate:
        res_obj = torchaudio.transforms.Resample(sr, args.sample_rate)
        x = res_obj(x)
    torchaudio.save(f"{target_file}{args.target_ext}", x, sample_rate=args.sample_rate, bits_per_sample=16)

