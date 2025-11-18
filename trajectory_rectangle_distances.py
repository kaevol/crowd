#!/usr/bin/env python3
"""
轨迹矩形距离计算脚本
计算每帧中有效连线的两个矩形之间的最短距离
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import os
from itertools import combinations
from typing import List, Tuple, Optional


def get_rectangle_corners(x: float, y: float, length: float, width: float, psi_rad: float) -> np.ndarray:
    """
    计算任意方向矩形的四个顶点坐标

    Args:
        x, y: 矩形中心点坐标
        length: 矩形长度
        width: 矩形宽度
        psi_rad: 矩形旋转角度（弧度）

    Returns:
        四个顶点坐标的数组 shape=(4, 2)
    """
    # 半长和半宽
    half_length = length / 2
    half_width = width / 2

    # 矩形在局部坐标系中的四个顶点（逆时针）
    local_corners = np.array([
        [half_length, half_width],    # 右上
        [-half_length, half_width],   # 左上
        [-half_length, -half_width],  # 左下
        [half_length, -half_width]    # 右下
    ])

    # 旋转矩阵
    cos_psi = np.cos(psi_rad)
    sin_psi = np.sin(psi_rad)
    rotation_matrix = np.array([
        [cos_psi, -sin_psi],
        [sin_psi, cos_psi]
    ])

    # 旋转并平移到全局坐标系
    global_corners = local_corners @ rotation_matrix.T + np.array([x, y])

    return global_corners


def line_segment_intersection(p1: np.ndarray, p2: np.ndarray,
                               p3: np.ndarray, p4: np.ndarray) -> bool:
    """
    检查线段(p1, p2)和线段(p3, p4)是否相交

    使用向量叉积方法判断
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # 检查共线情况
    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True

    return False


def line_intersects_rectangle(p1: np.ndarray, p2: np.ndarray,
                               corners: np.ndarray) -> bool:
    """
    检查线段是否与矩形的任意边相交

    Args:
        p1, p2: 线段的两个端点
        corners: 矩形的四个顶点坐标

    Returns:
        是否相交
    """
    # 检查线段与矩形四条边的相交
    for i in range(4):
        edge_start = corners[i]
        edge_end = corners[(i + 1) % 4]
        if line_segment_intersection(p1, p2, edge_start, edge_end):
            return True
    return False


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    计算点p到线段ab的最短距离
    """
    ab = b - a
    ap = p - a

    # 线段长度的平方
    ab_squared = np.dot(ab, ab)

    if ab_squared == 0:
        # a和b重合
        return np.linalg.norm(ap)

    # 计算投影参数
    t = np.clip(np.dot(ap, ab) / ab_squared, 0, 1)

    # 最近点
    closest_point = a + t * ab

    return np.linalg.norm(p - closest_point)


def segment_to_segment_distance(a1: np.ndarray, a2: np.ndarray,
                                 b1: np.ndarray, b2: np.ndarray) -> float:
    """
    计算两条线段之间的最短距离
    """
    # 检查是否相交
    if line_segment_intersection(a1, a2, b1, b2):
        return 0.0

    # 计算四种端点到线段的距离
    distances = [
        point_to_segment_distance(a1, b1, b2),
        point_to_segment_distance(a2, b1, b2),
        point_to_segment_distance(b1, a1, a2),
        point_to_segment_distance(b2, a1, a2)
    ]

    return min(distances)


def rectangle_to_rectangle_distance(corners_a: np.ndarray,
                                     corners_b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    计算两个矩形之间的最短距离

    遍历矩形A的所有边和顶点，计算到矩形B的最短距离

    Returns:
        最短距离, 矩形A上的最近点, 矩形B上的最近点
    """
    min_distance = float('inf')
    closest_point_a = None
    closest_point_b = None

    # 遍历矩形A的所有边和矩形B的所有边
    for i in range(4):
        edge_a_start = corners_a[i]
        edge_a_end = corners_a[(i + 1) % 4]

        for j in range(4):
            edge_b_start = corners_b[j]
            edge_b_end = corners_b[(j + 1) % 4]

            # 检查边是否相交
            if line_segment_intersection(edge_a_start, edge_a_end,
                                         edge_b_start, edge_b_end):
                return 0.0, edge_a_start, edge_b_start

    # 计算矩形A的顶点到矩形B的边的距离
    for i in range(4):
        point_a = corners_a[i]
        for j in range(4):
            edge_b_start = corners_b[j]
            edge_b_end = corners_b[(j + 1) % 4]

            # 计算点到线段的最近点
            ab = edge_b_end - edge_b_start
            ap = point_a - edge_b_start
            ab_squared = np.dot(ab, ab)

            if ab_squared == 0:
                closest_on_b = edge_b_start
            else:
                t = np.clip(np.dot(ap, ab) / ab_squared, 0, 1)
                closest_on_b = edge_b_start + t * ab

            dist = np.linalg.norm(point_a - closest_on_b)
            if dist < min_distance:
                min_distance = dist
                closest_point_a = point_a.copy()
                closest_point_b = closest_on_b.copy()

    # 计算矩形B的顶点到矩形A的边的距离
    for i in range(4):
        point_b = corners_b[i]
        for j in range(4):
            edge_a_start = corners_a[j]
            edge_a_end = corners_a[(j + 1) % 4]

            # 计算点到线段的最近点
            ab = edge_a_end - edge_a_start
            ap = point_b - edge_a_start
            ab_squared = np.dot(ab, ab)

            if ab_squared == 0:
                closest_on_a = edge_a_start
            else:
                t = np.clip(np.dot(ap, ab) / ab_squared, 0, 1)
                closest_on_a = edge_a_start + t * ab

            dist = np.linalg.norm(point_b - closest_on_a)
            if dist < min_distance:
                min_distance = dist
                closest_point_a = closest_on_a.copy()
                closest_point_b = point_b.copy()

    return min_distance, closest_point_a, closest_point_b


def count_intersections(p1: np.ndarray, p2: np.ndarray,
                        rectangles: List[dict],
                        exclude_indices: set) -> int:
    """
    统计连线与多少个矩形相交

    Args:
        p1, p2: 连线的两个端点
        rectangles: 所有矩形的信息
        exclude_indices: 排除的矩形索引（即连线两端的矩形）

    Returns:
        相交的矩形数量
    """
    count = 0
    for idx, rect in enumerate(rectangles):
        if idx in exclude_indices:
            count += 1  # 连线端点所在的矩形算作相交
            continue
        if line_intersects_rectangle(p1, p2, rect['corners']):
            count += 1
    return count


def process_frame(frame_data: pd.DataFrame) -> Optional[Tuple[float, dict, dict, np.ndarray, np.ndarray]]:
    """
    处理单帧数据，找出最小距离

    Returns:
        (最小距离, 矩形A信息, 矩形B信息, 最近点A, 最近点B) 或 None
    """
    # 构建所有矩形
    rectangles = []
    for _, row in frame_data.iterrows():
        corners = get_rectangle_corners(
            row['x'], row['y'],
            row['length'], row['width'],
            row['psi_rad']
        )
        rectangles.append({
            'track_id': row['track_id'],
            'x': row['x'],
            'y': row['y'],
            'length': row['length'],
            'width': row['width'],
            'psi_rad': row['psi_rad'],
            'corners': corners,
            'center': np.array([row['x'], row['y']])
        })

    if len(rectangles) < 2:
        return None

    min_distance = float('inf')
    best_pair = None
    best_closest_points = None

    # 遍历所有矩形对
    for i, j in combinations(range(len(rectangles)), 2):
        rect_a = rectangles[i]
        rect_b = rectangles[j]

        # 检查中心点连线是否只与这两个矩形相交
        center_a = rect_a['center']
        center_b = rect_b['center']

        # 统计连线与几个矩形相交
        intersection_count = count_intersections(
            center_a, center_b,
            rectangles,
            {i, j}
        )

        # 只有当连线只与两个矩形相交时才是有效连线
        if intersection_count == 2:
            # 计算两个矩形之间的最短距离
            dist, point_a, point_b = rectangle_to_rectangle_distance(
                rect_a['corners'], rect_b['corners']
            )

            if dist < min_distance:
                min_distance = dist
                best_pair = (rect_a, rect_b)
                best_closest_points = (point_a, point_b)

    if best_pair is None:
        return None

    return (min_distance, best_pair[0], best_pair[1],
            best_closest_points[0], best_closest_points[1])


def plot_rectangles_with_distance(rect_a: dict, rect_b: dict,
                                   point_a: np.ndarray, point_b: np.ndarray,
                                   distance: float, case_id: int, frame_id: int,
                                   output_dir: str):
    """
    绘制两个矩形和它们之间的最短距离连线
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 绘制矩形A
    corners_a = rect_a['corners']
    polygon_a = Polygon(corners_a, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(polygon_a)

    # 绘制矩形B
    corners_b = rect_b['corners']
    polygon_b = Polygon(corners_b, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(polygon_b)

    # 绘制中心点
    ax.plot(rect_a['x'], rect_a['y'], 'bo', markersize=8)
    ax.plot(rect_b['x'], rect_b['y'], 'ro', markersize=8)

    # 绘制最短距离连线
    ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]],
            'g-', linewidth=2, marker='o', markersize=6)

    # 标注连线长度
    mid_point = (point_a + point_b) / 2
    ax.annotate(f'Distance: {distance:.3f}',
                xy=mid_point,
                xytext=(mid_point[0] + 1, mid_point[1] + 1),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    # 标注矩形A的信息
    info_a = (f"Track ID: {rect_a['track_id']}\n"
              f"Center: ({rect_a['x']:.2f}, {rect_a['y']:.2f})\n"
              f"Length: {rect_a['length']:.2f}\n"
              f"Width: {rect_a['width']:.2f}\n"
              f"Psi: {rect_a['psi_rad']:.3f} rad")

    # 在矩形A附近标注
    text_pos_a = corners_a[0] + np.array([1, 1])
    ax.text(text_pos_a[0], text_pos_a[1], info_a,
            fontsize=8, color='blue',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 标注矩形B的信息
    info_b = (f"Track ID: {rect_b['track_id']}\n"
              f"Center: ({rect_b['x']:.2f}, {rect_b['y']:.2f})\n"
              f"Length: {rect_b['length']:.2f}\n"
              f"Width: {rect_b['width']:.2f}\n"
              f"Psi: {rect_b['psi_rad']:.3f} rad")

    # 在矩形B附近标注
    text_pos_b = corners_b[2] + np.array([-1, -1])
    ax.text(text_pos_b[0], text_pos_b[1], info_b,
            fontsize=8, color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            ha='right', va='top')

    # 设置标题
    ax.set_title(f'Case {case_id} - Frame {frame_id}\nMin Distance: {distance:.3f}',
                 fontsize=12)

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # 添加图例
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label=f'Rectangle A (Track {rect_a["track_id"]})'),
        Line2D([0], [0], color='red', linewidth=2, label=f'Rectangle B (Track {rect_b["track_id"]})'),
        Line2D([0], [0], color='green', linewidth=2, marker='o', label='Shortest Distance')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 保存图片
    filename = f'case_{case_id}_frame_{frame_id}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(csv_path: str, output_dir: str = 'output'):
    """
    主函数

    Args:
        csv_path: 输入CSV文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # 读取数据
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 获取所有case_id
    case_ids = df['case_id'].unique()
    print(f"Found {len(case_ids)} cases")

    # 存储所有结果
    all_results = []

    # 处理每个case
    for case_id in case_ids:
        case_data = df[df['case_id'] == case_id]
        frame_ids = sorted(case_data['frame_id'].unique())

        print(f"Processing case {case_id} with {len(frame_ids)} frames...")

        for frame_id in frame_ids:
            frame_data = case_data[case_data['frame_id'] == frame_id]

            result = process_frame(frame_data)

            if result is not None:
                min_distance, rect_a, rect_b, point_a, point_b = result

                # 记录结果
                all_results.append({
                    'case_id': case_id,
                    'frame_id': frame_id,
                    'min_distance': min_distance,
                    'track_id_a': rect_a['track_id'],
                    'track_id_b': rect_b['track_id'],
                    'x_a': rect_a['x'],
                    'y_a': rect_a['y'],
                    'x_b': rect_b['x'],
                    'y_b': rect_b['y']
                })

                # 绘制图片
                plot_rectangles_with_distance(
                    rect_a, rect_b, point_a, point_b,
                    min_distance, case_id, frame_id,
                    images_dir
                )
            else:
                # 没有有效连线，记录为NaN
                all_results.append({
                    'case_id': case_id,
                    'frame_id': frame_id,
                    'min_distance': np.nan,
                    'track_id_a': np.nan,
                    'track_id_b': np.nan,
                    'x_a': np.nan,
                    'y_a': np.nan,
                    'x_b': np.nan,
                    'y_b': np.nan
                })

    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)

    # 保存CSV
    csv_output_path = os.path.join(output_dir, 'min_distances.csv')
    results_df.to_csv(csv_output_path, index=False)
    print(f"Results saved to {csv_output_path}")

    # 绘制分布图
    valid_distances = results_df['min_distance'].dropna()

    if len(valid_distances) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图
        axes[0].hist(valid_distances, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Minimum Distance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Minimum Distances (Histogram)')
        axes[0].grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = (f"Count: {len(valid_distances)}\n"
                      f"Mean: {valid_distances.mean():.3f}\n"
                      f"Std: {valid_distances.std():.3f}\n"
                      f"Min: {valid_distances.min():.3f}\n"
                      f"Max: {valid_distances.max():.3f}")
        axes[0].text(0.95, 0.95, stats_text,
                     transform=axes[0].transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 箱线图
        axes[1].boxplot(valid_distances)
        axes[1].set_ylabel('Minimum Distance')
        axes[1].set_title('Distribution of Minimum Distances (Box Plot)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        dist_plot_path = os.path.join(output_dir, 'distance_distribution.png')
        plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Distribution plot saved to {dist_plot_path}")
    else:
        print("No valid distances found to plot distribution")

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {len(all_results)}")
    print(f"Valid distances: {len(valid_distances)}")

    return results_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate minimum distances between trajectory rectangles')
    parser.add_argument('--input', '-i', type=str, default='o61.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory')

    args = parser.parse_args()

    results = main(args.input, args.output)
