set -e

mkdir -p bbox_frames
for i in $(seq -f "%05g" 1 200)
do
    python tools/ground_truth/bounding_boxes_3d.py worldgen/outputs/69cc719a $i --query veratrum --output bbox_frames
    rm -f bbox_frames/A.png
    mv bbox_frames/B.png bbox_frames/"bboxes_$i.png"
done
