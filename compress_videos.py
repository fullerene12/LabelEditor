import subprocess
from glob import glob
from tqdm import tnrange


file_list = glob('./videos/*.h264')

for i in tnrange(len(file_list)):
    fname = file_list[i]
    cmd = ['ffmpeg','-i',fname,'-vf','scale=646:482',fname[:-5]+'.mp4']
    try:
    	subprocess.call(cmd)
    except Exception as ex:
    	print(ex)
