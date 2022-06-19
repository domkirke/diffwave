import os, torch, torchaudio, argparse, re, tqdm


parser = argparse.ArgumentParser()
parser.add_argument('input_folder', type=str)
parser.add_argument('output_folder', type=str)
parser.add_argument('duration', type=float)
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
    split_length = int(args.duration * sr)
    x_splitted = x.split(split_length, -1)
    for i, x_part in enumerate(x_splitted):
        current_filename = f"{target_file}_{i}{ext}"
        torchaudio.save(current_filename, x_part, sample_rate=sr, bits_per_sample=16)
    
    
