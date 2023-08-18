# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: David Yan, Lahav Lipson (SLURM job parsing)


import os
import sys
import argparse
import shutil
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import defaultdict
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import math
from scipy.signal import convolve2d
from skimage.restoration import estimate_sigma
from skimage.metrics import structural_similarity
import json 

@dataclass
class Job:
    job_id: str
    name: str
    current_status: str
    req_memory: str = None
    max_memory_gb: float = -1
    cpu: str = None
    gpu: str = None
    node: str = None
    start_time: datetime = None
    time_elapsed: timedelta = None

    def end_time(self):
        return self.start_time + self.time_elapsed

    def __lt__(self, other):
        return (int(self.job_id) < int(other.job_id))

    def __str__(self):
        if self.req_memory is not None:
            return f"{self.job_id}     {self.name.ljust(60)}     {self.gpu.ljust(10)} {self.cpu.ljust(5)}  {f'{self.max_memory_gb:.3f}G'.ljust(9)}  {self.start_time.strftime('%m/%d/%Y, %H:%M:%S')}  {str(self.time_elapsed).ljust(20)}  {self.current_status.ljust(10)}  {self.node}"
        else:
            return f"{self.job_id}     {self.name.ljust(60+73)}     {self.current_status.ljust(10)}"


sacct_line_regex = re.compile(r"([0-9]+) +(\S+) +(\S+) +([0-9]+) +([A-Z_]+) +(node[0-9]+) +(\S+).*").fullmatch
MEM_FACTOR = {"G": 1, "M": 1e3, "K": 1e6}

pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  
pd.options.display.width=None

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

all_data = defaultdict(dict)

def sizeof_fmt(num, suffix="B"): #https://stackoverflow.com/a/1094933
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def td_to_str(td): #https://stackoverflow.com/a/64662985
    """
    convert a timedelta object td to a string in HH:MM:SS format.
    """
    if (pd.isnull(td)):
        return td
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'


def parse_sacct_line(line):
    if sacct_line_regex(line) is None:
        return
    job_id, job_name, resources, elapsed_raw, current_status, node, start_time = sacct_line_regex(line).groups()
    request = dict(e.split('=') for e in resources.split(','))
    start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
    elapsed = timedelta(seconds=int(elapsed_raw))
    return Job(job_id=job_id, name=job_name, req_memory=request['mem'], cpu=request['cpu'], gpu=request.get('gpu', '0'), current_status=current_status, node=node, start_time=start_time, time_elapsed=elapsed)

def parse_scene_log(scene_path, step_times, asset_time_data, poly_data, asset_mem_data, obj_created_data, instance_created_data):
    log_folder = os.path.join(scene_path, "logs")
    coarse_folder = os.path.join(scene_path , "coarse") 
    fine_folder = next(Path(scene_path).glob("fine*"))
    seed = Path(scene_path).stem 
    scene_times = []
    if (os.path.isdir(log_folder)):
        for filepath in Path(log_folder).glob('*.err'):
            step = ""
            for stepName in step_times:
                if filepath.stem.startswith(stepName): 
                    step = stepName
                    break
            else: continue
            errFile = open(filepath)
            text = errFile.read()
            if "[MAIN TOTAL] finished in" not in text: continue
            search = re.search(r'\[MAIN TOTAL\] finished in ([0-9]+):([0-9]+):([0-9]+)', text)
            d = None
            if search is None: 
                search = re.search(r'\[MAIN TOTAL\] finished in ([0-9]) day.*, ([0-9]+):([0-9]+):([0-9]+)', text)
                d,h,m,s = search.group(1,2,3,4)
            else:
                h,m,s = search.group(1,2,3)
            if d is None: 
                step_timedelta = timedelta(hours=int(h),minutes=int(m),seconds=int(s))
            else: 
                step_timedelta = timedelta(days=int(d), hours=int(h),minutes=int(m),seconds=int(s))
            step_times[step].append(step_timedelta)
            scene_times.append(step_timedelta)
            all_data[seed]["[" + step + "] Step Time"] = step_timedelta

            # parse times < 1 day
            for name, h, m, s in re.findall(r'INFO:times:\[(.*?)\] finished in ([0-9]+):([0-9]+):([0-9]+)', text):
                timedelta_obj = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
                if (name == "MAIN TOTAL"): continue
                else:
                    if (timedelta_obj.total_seconds() < 1): continue
                    instance_dict = {}
                    instance_dict["stage_timedelta"] = timedelta_obj
                    instance_dict["step_timedelta"] = step_timedelta 
                    instance_dict["step_name"] = stepName # should be same for every instance of a given stage
                    instance_dict["seed"]  = seed
                    stage_key = "[" + stepName + "] " + name
                    asset_time_data[stage_key].append(instance_dict)
                    if stage_key in all_data[seed]:
                        all_data[seed]["[time] " + stage_key] += timedelta_obj
                    else:
                        all_data[seed]["[time] " + stage_key] = timedelta_obj

            # parse times > 1 day
            for name, d, h, m, s in re.findall(r'INFO:times:\[(.*?)\] finished in ([0-9]) day.*, ([0-9]+):([0-9]+):([0-9]+)', text):
                timedelta_obj = timedelta(days=int(d), hours=int(h),minutes=int(m),seconds=int(s))
                if (name == "MAIN TOTAL"): continue
                else:
                    if (timedelta_obj.total_seconds() < 1): continue
                    instance_dict = {}
                    instance_dict["stage_timedelta"] = timedelta_obj
                    instance_dict["step_timedelta"] = step_timedelta 
                    instance_dict["step_name"] = stepName # should be same for every instance of a given stage
                    instance_dict["seed"] = seed 
                    stage_key = "[" + stepName + "] " + name
                    asset_time_data[stage_key].append(instance_dict)
                    if stage_key in all_data[seed]:
                        all_data[seed]["[time] " + stage_key] += timedelta_obj
                    else:
                        all_data[seed]["[time] " + stage_key] = timedelta_obj

    scene_time = sum(scene_times, timedelta())
    all_data[seed]["Total Scene Time"] = scene_time

    for asset_step in asset_time_data:
        for stage_instance in asset_time_data[asset_step]:
            if (stage_instance["seed"] == seed):
                stage_instance["scene_time"] = scene_time
       
    coarse_poly = os.path.join(coarse_folder, "polycounts.txt")
    fine_poly = os.path.join(fine_folder, "polycounts.txt")
    if os.path.isfile(coarse_poly) and os.path.isfile(fine_poly):
        coarse_text = open(coarse_poly).read().replace(',', '')
        fine_text = open(fine_poly).read().replace(',', '')
        for faces, tris in re.findall("Faces:([0-9]+)Tris:([0-9]+)", coarse_text):
            poly_data["[Coarse] Faces"].append(int(faces))
            poly_data["[Coarse] Tris"].append(int(tris))
            all_data[seed]["[Polys] [Coarse] Faces"] = int(faces)
            all_data[seed]["[Polys] [Coarse] Tris"] = int(tris)

        for faces, tris in re.findall("Faces:([0-9]+)Tris:([0-9]+)", fine_text):
            poly_data["[Fine] Faces"].append(int(faces))
            poly_data["[Fine] Tris"].append(int(tris))
            all_data[seed]["[Polys] [Fine] Faces"] = int(faces)
            all_data[seed]["[Polys] [Fine] Tris"] = int(tris)

    coarse_stage_df = pd.read_csv(os.path.join(coarse_folder, "pipeline_coarse.csv"))
    coarse_stage_df["mem_delta"] = coarse_stage_df[coarse_stage_df['ran']==True]['mem_at_finish'].diff()
    coarse_stage_df["obj_delta"] = coarse_stage_df[coarse_stage_df['ran']==True]['obj_count'].diff()
    coarse_stage_df["instance_delta"] = coarse_stage_df[coarse_stage_df['ran']==True]['instance_count'].diff()
    for index, row in coarse_stage_df.iterrows():
        if row["mem_delta"] == 0 or math.isnan(float(row["mem_delta"])) or row["ran"] == False: continue
        asset_mem_data["[Coarse] " + row["name"]].append(row["mem_delta"])
        obj_created_data["[Coarse] " + row["name"]].append(row["obj_delta"])
        instance_created_data["[Coarse] " + row["name"]].append(row["instance_delta"])
        all_data[seed]["[Memory] [Coarse] " + row["name"]] = sizeof_fmt(row["mem_delta"])
        all_data[seed]["[Objects Generated] [Coarse] " + row["name"]] = row["obj_delta"]
        all_data[seed]["[Instances Generated] [Coarse] " + row["name"]] = row["instance_delta"]

    fine_stage_df = pd.read_csv(os.path.join(coarse_folder, "pipeline_fine.csv")) # this is supposed to be coarse folder
    fine_stage_df["mem_delta"] = fine_stage_df[fine_stage_df['ran']]['mem_at_finish'].diff()
    fine_stage_df["obj_delta"] = fine_stage_df[fine_stage_df['ran']]['obj_count'].diff()
    fine_stage_df["instance_delta"] = fine_stage_df[fine_stage_df['ran']]['instance_count'].diff()
    for index, row in fine_stage_df.iterrows():
        if row["mem_delta"] == 0 or math.isnan(float(row["mem_delta"])) or row["ran"] == False: continue
        asset_mem_data["[Fine] " + row["name"]].append(row["mem_delta"])
        obj_created_data["[Fine] " + row["name"]].append(row["obj_delta"])
        instance_created_data["[Fine] " + row["name"]].append(row["instance_delta"])
        all_data[seed]["[Memory] [Fine] " + row["name"]] = sizeof_fmt(row["mem_delta"])
        all_data[seed]["[Objects Generated] [Fine] " + row["name"]] = row["obj_delta"]
        all_data[seed]["[Instances Generated] [Fine] " + row["name"]] = row["instance_delta"]
    
def test_generation(dir):
    completed_seeds = os.path.join(dir, "finished_seeds.txt")
    num_lines = sum(1 for _ in open(completed_seeds))
    num_scenes = len(next(os.walk(dir))[1]) - 1
    print(f'{num_lines}/{num_scenes} succeeded scenes')
   # assert num_lines >= 0.8 * int(num_scenes), "Over 20% of scenes did not complete"

def make_stats(data_df):
    stats = pd.DataFrame()
    stats['mean'] = data_df.mean(axis=1)
    stats['median'] = data_df.min(axis=1)
    stats['90%'] = data_df.quantile(0.9, axis=1)
    stats['95%'] = data_df.quantile(0.95, axis=1)
    stats['99%'] = data_df.quantile(0.99, axis=1)
    return stats

def test_logs(dir):
    print('')
    asset_time_data = defaultdict(list) # data for individual asset stages
    asset_mem_data = defaultdict(list)
    obj_created_data = defaultdict(list)
    instance_created_data = defaultdict(list)


    step_times = {"fineterrain" : [], "coarse" : [], "populate" : [], "rendershort" : [], "blendergt": []}
    poly_data = {"[Coarse] Faces" : [], "[Coarse] Tris" : [], \
        "[Fine] Faces" : [], "[Fine] Tris" : []}
    completed_seeds = os.path.join(dir, "finished_seeds.txt")
    num_lines = sum(1 for _ in open(completed_seeds))
    for scene in os.listdir(dir):
        if scene not in open(completed_seeds).read(): continue
        scene_path = os.path.join(dir, scene)
        parse_scene_log(scene_path, step_times, asset_time_data, poly_data, asset_mem_data, obj_created_data, instance_created_data)

    step_df = pd.DataFrame.from_dict(step_times, orient='index')
    step_stats = make_stats(step_df)
    for column in step_stats:
        step_stats[column] = step_stats[column].dt.round('1s').map(lambda x: td_to_str(x))
    print("Time Logs by Step")
    print(tabulate(step_stats, headers='keys', tablefmt='fancy_grid'))
    
    asset_stats = defaultdict(list)
    for asset_name in asset_time_data:
        asset_times = pd.Series(instance["stage_timedelta"] for instance in asset_time_data[asset_name])
        chance = float(len(asset_time_data[asset_name]))/float(num_lines) 
        step_times = pd.Series(instance["step_timedelta"] for instance in asset_time_data[asset_name])
        scene_times = pd.Series(instance["scene_time"] for instance in asset_time_data[asset_name])
        step_percent = asset_times.sum() / step_times.sum() * 100
        scene_percent = asset_times.sum() / scene_times.sum() * 100
        asset_stats[asset_name] =\
             [asset_times.mean().round(freq ='s'), asset_times.median().round(freq ='s'),\
             asset_times.quantile(0.9).round(freq ='s'), asset_times.quantile(0.95).round(freq ='s'),\
             asset_times.quantile(0.99).round(freq ='s'), asset_times.sum().round(freq ='s'), \
             scene_percent, step_percent, asset_time_data[asset_name][0]["step_name"], chance]
    
    asset_stats = pd.DataFrame.from_dict(asset_stats,  orient='index')
    asset_stats.columns = ["mean time", "median time", "90%", "95%", "99%", "total time",  "% of scene time", "% of step time", "step",  "chance of ocurring"]
    
    for column in asset_stats:
        if column in ["mean time", "median time", "90%", "95%", "99%", "total time"]:
            asset_stats[column] = asset_stats[column].apply(td_to_str)
        
    print("\nTime Logs by Asset Stage")
    print(tabulate(asset_stats.sort_values("% of scene time", ascending=False), headers='keys', tablefmt='fancy_grid'))

    assset_mem_df = pd.DataFrame.from_dict(asset_mem_data, orient='index')
    assset_mem_stats = make_stats(assset_mem_df)
    assset_mem_stats = assset_mem_stats.sort_values("mean", ascending=False)   
    print("\nMemory Usage by Asset Stage")
    print(tabulate(assset_mem_stats.applymap(lambda x: sizeof_fmt(x)), headers='keys', tablefmt='fancy_grid'))

    obj_created_df = pd.DataFrame.from_dict(obj_created_data, orient='index')
    obj_created_stats = make_stats(obj_created_df)
    obj_created_stats = obj_created_stats.sort_values("mean", ascending=False)  
    print("\nObjects Generated by Asset Stage")
    print(tabulate(obj_created_stats, headers='keys', tablefmt='fancy_grid'))

    instance_created_df = pd.DataFrame.from_dict(instance_created_data, orient='index')
    instance_created_stats = make_stats(instance_created_df)
    instance_created_stats = instance_created_stats.sort_values("mean", ascending=False)   
    print("\nInstances Generated by Asset Stage")
    print(tabulate(instance_created_stats, headers='keys', tablefmt='fancy_grid'))

    poly_df = pd.DataFrame.from_dict(poly_data, orient='index')
    poly_stats = make_stats(poly_df)
    print("\nPolycount Statistics")
    print(tabulate(poly_stats, headers='keys', tablefmt='fancy_grid'))  


def test_step_memory(dir, days):
    days_since = int(days)
    sacct_start_date = (datetime.now() - timedelta(days=days_since)).strftime('%Y-%m-%d')
    sacct_command = f"sacct --starttime {sacct_start_date} -u {os.environ['USER']} --noheader -o jobid,jobname%80,AllocTRES%80,ElapsedRaw,stat%30,NodeList,Start,MaxRSS"
    print(f"Running + {sacct_command}")
    sacct_output = subprocess.check_output(sacct_command.split()).decode()
    relevant_started_jobs = []
    mem_dict = dict(re.findall(r"([0-9]+)\.0 +.* +([0-9]*.?[0-9]*[KMG])", sacct_output))

    completed_seeds = os.path.join(dir, "finished_seeds.txt")
    seeds = open(completed_seeds).read()

    for sacct_line in sacct_output.splitlines():
        parsed_job = parse_sacct_line(sacct_line)
        if (parsed_job is None):
            continue
        for name in re.findall(f"{Path(dir).stem}_([^ _]+)_.*", parsed_job.name):
            if name in seeds:
                if parsed_job.job_id in mem_dict:
                    max_memory = mem_dict[parsed_job.job_id]
                    parsed_job.max_memory_gb = float(max_memory[:-1]) / MEM_FACTOR[max_memory[-1]]
                relevant_started_jobs.append(parsed_job)
                
    step_mem = {"fineterrain" : [], "coarse" : [], "populate" : [], "rendershort" : [], "blendergt": []}

    for job in relevant_started_jobs:
        for step in step_mem:
            if (step in job.name):
                step_mem[step].append(job.max_memory_gb)
    
    step_mem_df = pd.DataFrame.from_dict(step_mem, orient='index')
    step_mem_stats = make_stats(step_mem_df)
    print("\nMemory Usage by Step")
    print(tabulate(step_mem_stats, headers='keys', tablefmt='fancy_grid'))
    

def test_brightness(dir):
    print('')
    completed_seeds = os.path.join(dir, "finished_seeds.txt")
    num_lines = sum(1 for _ in open(completed_seeds))
    numDark = 0
    for scene in os.listdir(dir):
        for filepath in Path(os.path.join(dir,scene)).rglob('Image*.png'):
            im = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE) #https://stackoverflow.com/a/52514730
            meanPercent = np.mean(im) * 100 / 255
            if meanPercent < 5:
                numDark+=1
                all_data[scene]["Dark"] = True
                print(f"{scene} is dark")
            else:
                all_data[scene]["Dark"] = False
            
    print(f"{numDark}/{num_lines} images are dark")


def test_noise(dir):
    noise_dict = {}
    for scene in os.listdir(dir):
        for filepath in Path(os.path.join(dir,scene)).rglob('Image*.png'):
            img = cv2.imread(str(filepath))
            sigma = estimate_sigma(img, channel_axis=-1, average_sigmas=True)
            noise_dict[scene] = sigma
            all_data[scene]["Noise Index"] = sigma
            break # there will be up to two renders but they're the same
    
    noise_df = pd.DataFrame.from_dict(noise_dict, orient='index')
    noise_df.columns = ["Noise Estimate"]
    print("\nNoise in Rendered Images")
    print(tabulate(noise_df.sort_values("Noise Estimate", ascending=False), headers='keys', tablefmt='fancy_grid'))


def test_gt(dir):
    completed_seeds = os.path.join(dir, "finished_seeds.txt")
    seeds = open(completed_seeds).read()
    print('')
    
    tag_seg_data = defaultdict(list)
    pix_sum_seg = 0

    obj_seg_data = defaultdict(list)
    pix_sum_obj = 0

    similarity = {}

    for scene in os.listdir(dir):
        if scene not in seeds: continue
        scene_path = os.path.join(dir,scene)
        blender_gt_search = list(Path(scene_path).glob('frames*'))
        opengl_gt_search = list(Path(scene_path).glob('opengl*'))
        
        if not blender_gt_search or not opengl_gt_search: continue

        blender_gt_folder = blender_gt_search[0] #should only be one occurrence of each
        opengl_gt_folder = opengl_gt_search[0]

        blender_depth_search = list(Path(blender_gt_folder).glob('Depth*.npy'))
        opengl_depth_search = list((opengl_gt_folder).glob('Depth*.npy'))
        
        if blender_depth_search and opengl_depth_search: 

            blender_depth = np.load(blender_depth_search[0])
            opengl_depth = np.load(opengl_depth_search[0])

            opengl_depth[opengl_depth == np.inf] = 10*10
            opengl_depth = cv2.resize(opengl_depth, dsize= (blender_depth.shape[1], blender_depth.shape[0]))

            score, diff = structural_similarity(blender_depth, opengl_depth, full=True)
            similarity[scene] = score * 100
    
        fine_folder = next(Path(scene_path).glob("fine*"))
        tags = json.load(open(os.path.join(fine_folder, "MaskTag.json")))
        tag_seg_search = list((opengl_gt_folder).glob('TagSegmentation*.npy'))

        if tag_seg_search:
            tag_seg = np.load(tag_seg_search[0])
            pix_sum_seg += tag_seg.shape[0] * tag_seg.shape[1]
            index, counts = np.unique(tag_seg, return_counts=True)
            tag_seg_dict = dict(zip(index, counts))
            for tag in tags.keys():
                if tags[tag] not in tag_seg_dict.keys():
                    tag_seg_data[tag].append(0)
                    all_data[scene]["[Tag Seg. Percent] " + tag] = 0
                else:
                    tag_seg_data[tag].append(tag_seg_dict[tags[tag]])
                    all_data[scene]["[Tag Seg. Percent] " + tag] = float(tag_seg_dict[tags[tag]])/float(tag_seg.shape[0] * tag_seg.shape[1]) * 100

        obj_json_search = list((blender_gt_folder).glob('Objects*.json'))
        obj_seg_search = list((blender_gt_folder).glob('ObjectSegmentation*.npy'))

        if (obj_json_search and obj_seg_search):
            obj_seg = np.load(obj_seg_search[0])
            objects = json.load(open(obj_json_search[0]))
            pix_sum_obj += obj_seg.shape[0] * obj_seg.shape[1]
            index, counts = np.unique(obj_seg, return_counts=True)
            obj_seg_dict = dict(zip(index, counts))
            for obj in objects.keys():
                concise_obj = obj.split('(')[0].split(':')[-1].split('.')[0]
                if objects[obj]["object_index"] not in obj_seg_dict.keys():
                   obj_seg_data[obj].append(0)
                   all_data[scene]["[Obj Seg. Percent] " + concise_obj] = 0
                else:
                    obj_seg_data[obj].append(obj_seg_dict[objects[obj]["object_index"]])
                    concise_obj = obj.split('(')[0].split(':')[-1].split('.')[0]
                    all_data[scene]["[Obj Seg. Percent] " + concise_obj] = float((obj_seg_dict[objects[obj]["object_index"]]))/float(obj_seg.shape[0] * obj_seg.shape[1]) * 100

    assert len(similarity) != 0
    similarity_df = pd.DataFrame.from_dict(similarity, orient='index')
    print("Comparison checking not fully working, proceed with caution")
    similarity_df.columns = ["Similarity Score (%)"]
    print(tabulate(similarity_df, headers='keys', tablefmt='fancy_grid'))

    tag_seg_data_df = pd.DataFrame.from_dict(tag_seg_data, orient='index')
    tag_seg_stats = pd.DataFrame()
    tag_seg_stats['Percent of Pixels'] = tag_seg_data_df.sum(axis=1).map(lambda x: float(x)/float(pix_sum_seg) * 100)

    print("\nTag Segmentation Pixel Sources")
    print("Percent of untagged pixels: " + str(100 - tag_seg_stats["Percent of Pixels"].sum()))
    print(tabulate(tag_seg_stats.sort_values("Percent of Pixels", ascending=False), headers='keys', tablefmt='fancy_grid'))
   

    obj_seg_data_df = pd.DataFrame.from_dict(obj_seg_data, orient='index')
    obj_seg_stats = pd.DataFrame()
    obj_seg_stats['Percent of Pixels'] = obj_seg_data_df.sum(axis=1).map(lambda x: float(x)/float(pix_sum_obj) * 100)
    obj_seg_stats = obj_seg_stats.groupby(obj_seg_stats.index.str.split('(').str[0]).sum()
    obj_seg_stats = obj_seg_stats.groupby(obj_seg_stats.index.str.split(':').str[-1]).sum()
    obj_seg_stats = obj_seg_stats.groupby(obj_seg_stats.index.str.split('.').str[0]).sum()

    print("\nObject Segmentation Pixel Sources")
    print("Percent of untagged pixels: " + str(100 - obj_seg_stats["Percent of Pixels"].sum()))
    print(tabulate(obj_seg_stats.sort_values("Percent of Pixels", ascending=False), headers='keys', tablefmt='fancy_grid'))


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=dir_path)
    parser.add_argument('-t', '--time', type=int, default=None)
    args = parser.parse_args()

    return args


def main(dir, time):
    if not os.path.isdir(f"{dir}/test_results"):
        os.mkdir(f"{dir}/test_results")
    if os.path.exists(f"{dir}/test_results/test_logs.log"):
        os.remove(f"{dir}/test_results/test_logs.log")
    sys.stdout = open(f"{dir}/test_results/test_logs.log", 'w')
    try:
        print("\nTesting scene success rate")
        test_generation(dir)
    except Exception as e: 
        print(e)

    try:
        print("\nTesting logs")
        test_logs(dir)
    except Exception as e: 
        print(e)

    if time is None:
        print("\nNo slurm time arg provided, skipping scene memory stats")
    else:
        try:
            print("\nTesting step memory")
            test_step_memory(dir, time)
        except Exception as e: 
            print(e)

    try:
        print("\nTesting scene brightness")
        test_brightness(dir)
    except Exception as e: 
        print(e)

    try:
        print("\nTesting scene noise")
        test_noise(dir)
    except Exception as e: 
        print(e)

    try:
        test_gt(dir)
    except Exception as e: 
        print(e)

    data_df = pd.DataFrame.from_dict(all_data, orient='index')
    data_df.to_csv(f"{dir}/test_results/data.csv")
    
if __name__ == '__main__':
    args = make_args()
    main(args.dir, args.time)

