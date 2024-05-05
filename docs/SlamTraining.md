# SLAM Camera Trajectories

In order to create dynamic and varied camera motion, use the `slam_training.gin` configuration when generating scenes. This will animate cameras using the animation policy found in [rrt.py](../infinigen/core/util/rrt.py/.py). There are three customizable components used to create such a motion:

 - `RRT`: This class can generate a path of nodes between pairs of start and goal nodes that avoids obstacles using the [RRT* algorithm](https://en.wikipedia.org/wiki/Rapidly_exploring_random_tree). 

 - `AnimPolicyRRT`: Initialized with an instance of RRT, this animation policy will use RRT to continually generate paths (using the previous goal as the next start node) until it reaches the scene's end frame. The nodes in an RRT path are used as keyframe positions, while the rotation and duration between keyframes are sampled from user specified distributions. If the animation policy fails to validate some pose, then it retries the rotation at the corresponding keyframe. If the animation policy does a full retry, then it regenerates paths with RRT. 

 - `validate_cam_pose_rrt`: This function will validate a camera pose based on two conditions: the percentage of sky and close-by pixels in the view frame. A pixel is considered sky if the raycast through it does not intersect an object. A pixel is considered close-by if the raycast intersects an object a distance less than the focal length away. A pose is invalid if the percentage of pixels checked that are sky or are close-by exceed certain specified percentages.

In order to make the resultant motion easier/harder, modify the distributions of `AnimPolicyRRT.speed` or `AnimPolicyRRT.rot` in `slam_training.gin`. For example, to restrict the rotation in the roll dimension, set `AnimPolicyRRT.rot = ('normal', 0, [0, 15, 15], 3)`
