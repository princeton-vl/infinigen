import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional


def generate_html(
    asset_name: str,
    base_dir: str = "./tmp",
    output_file: Optional[str] = None,
    seeds_to_show: Optional[List[int]] = [],
    failure_seeds: Optional[List[int]] = [],
) -> Optional[Path]:
    """Generate HTML visualization for joint animations"""
    base_dir = Path(base_dir).absolute()
    base_path = base_dir / "renders" / "mjcf" / asset_name

    if not base_path.exists():
        print(f"Error: Path not found: {base_path}")
        return None

    # Collect video data organized by seed and joint
    seeds_data: Dict[str, Dict[str, Dict[str, str]]] = {}

    for seed_dir in base_path.iterdir():
        if len(seeds_to_show) == 0:
            continue

        if not seed_dir.is_dir():
            continue

        if seeds_to_show and int(seed_dir.name) not in seeds_to_show:
            continue

        seed_name = seed_dir.name
        joints: Dict[str, Dict[str, str]] = {}

        # Find and categorize videos
        for video_file in seed_dir.glob("*.mp4"):
            video_path = str(video_file.absolute())
            video_name = video_file.stem

            # Parse video filename to extract joint name and view type
            if "_front_view" in video_name:
                joint_name = video_name.replace("_front_view", "").replace("joint_", "")
                view = "front"
            elif "_left_side_view" in video_name:
                joint_name = video_name.replace("_left_side_view", "").replace(
                    "joint_", ""
                )
                view = "left"
            elif "_right_side_view" in video_name:
                joint_name = video_name.replace("_right_side_view", "").replace(
                    "joint_", ""
                )
                view = "right"
            elif "_top_view" in video_name:
                joint_name = video_name.replace("_top_view", "").replace("joint_", "")
                view = "top"
            else:
                continue

            # Store video path by joint and view
            if joint_name not in joints:
                joints[joint_name] = {}
            joints[joint_name][view] = video_path

        if joints:
            seeds_data[seed_name] = joints

    if not seeds_data:
        print(f"No videos found for asset: {asset_name}")

    # Determine output file path
    if output_file is None:
        output_file = base_dir / f"{asset_name}_visualization.html"
    else:
        output_file = base_dir / output_file

    html_dir = output_file.parent

    # Define background colors for seed sections
    colors = [
        "#f0f9ff",
        "#fef3c7",
        "#fce7f3",
        "#f0fdf4",
        "#fef2f2",
        "#f5f3ff",
        "#fff7ed",
        "#ecfeff",
    ]

    # Generate HTML header and style
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{asset_name} - Joint Views</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            padding: 40px 20px; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .status-banner {{
            padding: 16px 24px;
            margin-bottom: 24px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .status-banner.success {{
            background: #10b981;
            color: white;
        }}
        .status-banner.failure {{
            background: #ef4444;
            color: white;
        }}
        .status-icon {{
            font-size: 20px;
        }}
        h1 {{ 
            font-size: 28px; 
            margin-bottom: 30px; 
            color: #1a1a1a;
            font-weight: 600;
        }}
        .seed-section {{ 
            margin-bottom: 16px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .seed-header {{
            padding: 20px 24px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: filter 0.2s;
        }}
        .seed-header:hover {{ filter: brightness(0.97); }}
        .seed-header h2 {{ 
            font-size: 18px;
            font-weight: 500;
            color: #1a1a1a;
        }}
        .toggle-icon {{
            font-size: 20px;
            color: #666;
            transition: transform 0.3s;
        }}
        .seed-section.collapsed .toggle-icon {{ transform: rotate(-90deg); }}
        .seed-content {{
            max-height: 10000px;
            overflow: hidden;
            transition: max-height 0.4s ease;
            background: white;
        }}
        .seed-section.collapsed .seed-content {{
            max-height: 0;
        }}
        .joint-section {{
            border-top: 1px solid #f0f0f0;
        }}
        .joint-header {{
            padding: 16px 24px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #fafafa;
            transition: background 0.2s;
        }}
        .joint-header:hover {{ background: #f5f5f5; }}
        .joint-header h3 {{ 
            font-size: 16px;
            font-weight: 500;
            color: #333;
        }}
        .joint-toggle-icon {{
            font-size: 16px;
            color: #666;
            transition: transform 0.3s;
        }}
        .joint-section.collapsed .joint-toggle-icon {{ transform: rotate(-90deg); }}
        .joint-content {{
            padding: 0 24px 24px 24px;
            max-height: 10000px;
            overflow: hidden;
            transition: max-height 0.4s ease, padding 0.4s ease;
        }}
        .joint-section.collapsed .joint-content {{
            max-height: 0;
            padding: 0 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #fafafa;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            color: #666;
            border-bottom: 2px solid #e0e0e0;
        }}
        td {{
            padding: 16px;
            border-bottom: 1px solid #f0f0f0;
            vertical-align: middle;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background: #fafafa;
        }}
        .view-name {{
            font-weight: 500;
            color: #1a1a1a;
            font-size: 14px;
        }}
        .video-container {{
            position: relative;
            width: 350px;
            height: 197px;
            background: #000;
            border-radius: 4px;
            overflow: hidden;
        }}
        video {{ 
            width: 100%;
            height: 100%;
            display: block;
        }}
        video::-webkit-media-controls {{
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .video-container:hover video::-webkit-media-controls {{
            opacity: 1;
        }}
        .empty-cell {{
            color: #999;
            font-style: italic;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{asset_name} - Joint Views</h1>
"""

    # Add status banner
    if failure_seeds:
        failure_list = ", ".join(str(s) for s in sorted(failure_seeds))
        html += f"""
        <div class="status-banner failure">
            <span class="status-icon">⚠️</span>
            <span>Failure detected in seeds: {failure_list}</span>
        </div>
"""
    else:
        html += """
        <div class="status-banner success">
            <span class="status-icon">✓</span>
            <span>All seeds passed successfully</span>
        </div>
"""

    # Generate sections for each seed
    for idx, seed_name in enumerate(sorted(seeds_data.keys())):
        color = colors[idx % len(colors)]
        html += f"""
        <div class="seed-section" style="background: {color};">
            <div class="seed-header" onclick="toggleSeed(this)">
                <h2>Seed: {seed_name}</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="seed-content">
"""

        # Generate subsections for each joint
        first_joint = True
        for joint_name in sorted(seeds_data[seed_name].keys()):
            views = seeds_data[seed_name][joint_name]
            collapsed_class = "" if first_joint else "collapsed"
            first_joint = False

            html += f"""
                <div class="joint-section {collapsed_class}">
                    <div class="joint-header" onclick="toggleJoint(this, event)">
                        <h3>Joint: {joint_name}</h3>
                        <span class="joint-toggle-icon">▼</span>
                    </div>
                    <div class="joint-content">
                        <table>
                            <thead>
                                <tr>
                                    <th>View</th>
                                    <th>MuJoCo</th>
                                    <th>Simulator 2</th>
                                    <th>Simulator 3</th>
                                </tr>
                            </thead>
                            <tbody>
"""

            # Add table rows for each view
            view_names = {
                "front": "Front View",
                "left": "Left View",
                "right": "Right View",
                "top": "Top View",
            }
            for view_type, view_label in view_names.items():
                if view_type in views:
                    rel_path = os.path.relpath(views[view_type], html_dir)
                    html += f"""
                                <tr>
                                    <td class="view-name">{view_label}</td>
                                    <td>
                                        <div class="video-container">
                                            <video loop muted playsinline controls>
                                                <source src="{rel_path}" type="video/mp4">
                                            </video>
                                        </div>
                                    </td>
                                    <td class="empty-cell">-</td>
                                    <td class="empty-cell">-</td>
                                </tr>
"""

            html += """
                            </tbody>
                        </table>
                    </div>
                </div>
"""

        html += """
            </div>
        </div>
"""

    # Add JavaScript for interactivity
    html += """
    </div>
    <script>
        // Toggle seed sections between expanded/collapsed
        function toggleSeed(header) {
            header.parentElement.classList.toggle('collapsed');
        }
        
        // Toggle joint sections between expanded/collapsed
        function toggleJoint(header, event) {
            event.stopPropagation();
            const seedContent = header.closest('.seed-content');
            const jointSections = seedContent.querySelectorAll('.joint-section');
            const currentSection = header.parentElement;
            const isCurrentlyCollapsed = currentSection.classList.contains('collapsed');
            
            // Collapse all and pause videos
            jointSections.forEach(section => {
                const videos = section.querySelectorAll('video');
                videos.forEach(video => {
                    video.pause();
                    video.currentTime = 0;
                });
                section.classList.add('collapsed');
            });
            
            // Expand clicked section and play videos if it was collapsed
            if (isCurrentlyCollapsed) {
                currentSection.classList.remove('collapsed');
                setTimeout(() => {
                    const videos = currentSection.querySelectorAll('video');
                    videos.forEach(video => {
                        video.play();
                    });
                }, 100);
            }
        }
        
        // Auto-play videos in the first non-collapsed section
        document.addEventListener('DOMContentLoaded', () => {
            document.querySelectorAll('.joint-section:not(.collapsed) video').forEach(video => {
                video.play();
            });
        });
    </script>
</body>
</html>
"""

    # Write HTML to file
    with open(output_file, "w") as f:
        f.write(html)

    print(f"HTML created at: {output_file}")
    print(f"Open: file://{output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate HTML visualization for joint animations"
    )
    parser.add_argument(
        "--asset_name", default="drawer", help="Name of the asset to visualize"
    )
    parser.add_argument(
        "--base_dir",
        default="infinigen/tools/sim/tmp",
        help="Base directory containing renders",
    )
    parser.add_argument("--output_file", help="Output HTML file name (optional)")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[],
        help="Specific seeds to include (default: all)",
    )
    parser.add_argument(
        "--failure_seeds",
        type=int,
        nargs="*",
        default=[],
        help="Seeds that failed (will be highlighted in red banner)",
    )
    args = parser.parse_args()

    generate_html(
        args.asset_name, args.base_dir, args.output_file, args.seeds, args.failure_seeds
    )
