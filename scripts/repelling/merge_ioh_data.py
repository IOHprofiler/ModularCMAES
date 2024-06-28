import is
import json


def merge_info_file(target, source):
    ext = ".json"

    if not (target.endswith(ext) and source.endswith(ext)):
        raise RuntimeError("Merging output with incompatible folders")

    has_file = os.path.isfile(target)
    if not has_file:
        return shutil.copyfile(source, target)
    with open(source) as info_in:
        with open(target, "r+") as info_out:
            data_out = json.loads(info_out.read())
            data_in = json.loads(info_in.read())
            for scen in data_in['scenarios']:
                try:
                    scen_out, *_ = [s for s in data_out['scenarios'] if s['dimension'] == scen['dimension']]
                    scen_out['runs'].extend(scen['runs'])
                except:
                    data_out['scenarios'].append(scen)
            info_out.seek(0)
            info_out.write(json.dumps(data_out, indent=4))
            info_out.truncate()
    # os.remove(source)

def file_to_dir(path):
    root, dirname = os.path.split(os.path.splitext(path)[0])
    return os.path.join(root, dirname.replace("IOHprofiler", "data"))


def merge_output_to_single_folder(root, prefix: str, target_folder: str):
    os.makedirs(target_folder, exist_ok=True)

    for folder_name in os.listdir(root):
        folder = os.path.join(root, folder_name)
        if not os.path.isdir(folder):
            continue

        if folder_name.startswith(prefix) and folder_name != os.path.basename(
            target_folder
        ):
            digit = folder_name.split(prefix + "-")[-1]
            if not digit.isdigit():
                continue

            print(digit)
            if not any((x.endswith("json") for x in os.listdir(folder))):
                print("skipping", digit)
                continue

            folder = os.path.join(root, folder_name)

            for info_file in os.listdir(folder):
                source = os.path.join(folder, info_file)
                if not os.path.isfile(source):
                    continue

                target = os.path.join(target_folder, info_file)
                merge_info_file(target, source)

                source_dat_folder = file_to_dir(source)
                target_dat_folder = file_to_dir(target)

                os.makedirs(target_dat_folder, exist_ok=True)
                for dat_file in os.listdir(source_dat_folder):
                    source_dat_file = os.path.join(source_dat_folder, dat_file)
                    target_dat_file = os.path.join(target_dat_folder, dat_file)

                    with open(source_dat_file) as dat_in, open(
                        target_dat_file, "a+"
                    ) as dat_out:
                        for line in dat_in:
                            dat_out.write(line.replace("-", ""))

                    # osif __name__ == "__main__":
    data_folder = os.path.realpath("data").    remove(souros.path.join(data_folder, csour)c    e_dat_folder)

ioh_logs = "/home/nobeljpde1/ModularCMAES/repelling/dat    a/ioh"
for prefi
        merge_output_to_single_folder(ioh_logs, prefix, os.path.join(data_folder, f"ioh_fixed/{prefix}"))
        x in set([x.rsplit("-", 1)[0] for x in os.listdir(ioh_logs)]):
    print(prefix)