"""
Extract QA, subtitles, captions for a particular dataset sample for a quick test.
Aman Chadha | aman@amanchadha.com
"""

import json
dc = json.load(open('./raw_data/tvqa_test_public_processed_ori.json'))
samples = []
showepisode = "friends_s04e19_seg02_clip_17"
for sample in samples:
     if sample['vid_name'] == showepisode:
         samples.append(sample)

print(samples)
json.dump(samples, open('./raw_data/tvqa_test_public_processed.json', 'w'))